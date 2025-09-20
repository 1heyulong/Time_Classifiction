import argparse
import os
import copy

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D

# =====================
#   Building Blocks
# =====================

class GaussianNoise(nn.Module):
    def __init__(self, std: float = 0.0):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class SE1d(nn.Module):
    """Squeeze-and-Excitation on token embeddings (B, N, C)."""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):  # x: (B, N, C)
        s = x.mean(dim=1)             # (B, C)
        s = self.fc2(self.act(self.fc1(s))).sigmoid()  # (B, C)
        s = s.unsqueeze(1)            # (B, 1, C)
        return x * s


class AttnPool1d(nn.Module):
    """Learnable attention pooling over tokens (sequence length N)."""
    def __init__(self, dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, N, C)
        a = self.fc2(self.act(self.fc1(self.norm(x)))).squeeze(-1)  # (B, N)
        w = torch.softmax(a, dim=1).unsqueeze(-1)                   # (B, N, 1)
        return (x * w).sum(dim=1)                                   # (B, C)


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0., use_bn_in_icb: bool = True):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.use_bn = use_bn_in_icb
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(hidden_features)
            self.bn2 = nn.BatchNorm1d(hidden_features)

    def forward(self, x):  # x: (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        x1 = self.conv1(x)
        if self.use_bn: x1 = self.bn1(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        x2 = self.conv2(x)
        if self.use_bn: x2 = self.bn2(x2)
        x2 = self.act(x2)
        x2 = self.drop(x2)

        out1 = x1 * x2
        out2 = x2 * x1

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x  # (B, N, C)


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):  # x: (B, C_in, T)
        x_out = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, C)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):  # (B, N, C)
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if args.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_layerscale: bool = True, layerscale_init: float = 1e-4, use_bn_in_icb: bool = True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, use_bn_in_icb=use_bn_in_icb)
        self.use_layerscale = use_layerscale
        if self.use_layerscale:
            self.gamma_icb = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma_asb = nn.Parameter(layerscale_init * torch.ones(dim))

    def forward(self, x):  # (B, N, C)
        if args.ICB and args.ASB:
            y = self.asb(self.norm1(x))
            if self.use_layerscale: y = y * self.gamma_asb
            y = self.icb(self.norm2(y))
            if self.use_layerscale: y = y * self.gamma_icb
            x = x + self.drop_path(y)
        elif args.ICB:
            y = self.icb(self.norm2(x))
            if self.use_layerscale: y = y * self.gamma_icb
            x = x + self.drop_path(y)
        elif args.ASB:
            y = self.asb(self.norm1(x))
            if self.use_layerscale: y = y * self.gamma_asb
            x = x + self.drop_path(y)
        return x


class TSLANet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)
        self.noise = GaussianNoise(std=args.noise_std)

        dpr = [x.item() for x in torch.linspace(0.0, args.drop_path_max, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i],
                          use_layerscale=args.use_layerscale, layerscale_init=args.layerscale_init,
                          use_bn_in_icb=args.use_bn_in_icb)
            for i in range(args.depth)
        ])

        # Optional SE
        self.use_se = args.use_se
        if self.use_se:
            self.se = SE1d(args.emb_dim, reduction=args.se_reduction)

        # Pooling head
        if args.pooling == 'attn':
            self.pool = AttnPool1d(args.emb_dim, reduction=args.attn_reduction)
        elif args.pooling == 'mean':
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling: {args.pooling}")

        self.head_drop = nn.Dropout(args.head_dropout)
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = self.noise(x)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()
        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)
        return x_masked, x_patched

    def forward_features(self, x):  # (B, C_in, T) -> (B, N, C)
        x = self.patch_embed(x)
        x = self.noise(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        if self.use_se:
            x = self.se(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.pool is None:
            x = x.mean(1)
        else:
            x = self.pool(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


# =====================
#        EMA
# =====================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        # buffers (e.g., running stats) if any
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def to(self, device):
        self.ema_model.to(device)


# =====================
#     Lightning Modules
# =====================
class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        preds, target = self.model.pretrain(data)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


# ---- Optional: SAM optimizer wrapper ----
class SAM(optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, "Invalid rho"
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups for p in group['params'] if p.grad is not None
        ]), p=2)
        scale = self.param_groups[0]['rho'] / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.sub_(self.state[p]['e_w'])  # get back to "w"
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("Call first_step and second_step explicitly")


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.use_mixup = args.mixup_alpha > 0
        self.criterion_hard = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.criterion_soft = SoftTargetCrossEntropy() if self.use_mixup else None
        self.test_preds, self.test_targets = [], []
        # EMA
        self.use_ema = args.use_ema
        if self.use_ema:
            self.ema = EMA(self.model, decay=args.ema_decay)

    def forward(self, x):
        return self.model(x)

    # --- optimizer & scheduler ---
    def configure_optimizers(self):
        if args.use_sam:
            optimizer = SAM(self.parameters(), base_optimizer=optim.AdamW, lr=args.train_lr, weight_decay=args.weight_decay, rho=args.sam_rho)
            scheduler = ReduceLROnPlateau(optimizer.base_optimizer, mode='min', factor=0.5, patience=100, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        else:
            optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    # --- hooks ---
    def on_fit_start(self):
        if self.use_ema:
            self.ema.to(self.device)

    def on_train_epoch_start(self):
        # linearly ramp DropPath by epoch
        max_drop_path = args.drop_path_max
        epoch = self.current_epoch
        max_epoch = self.trainer.max_epochs
        factor = float(epoch) / max(1, max_epoch)
        for blk in self.model.tsla_blocks:
            if hasattr(blk, 'drop_path') and isinstance(blk.drop_path, DropPath):
                blk.drop_path.drop_prob = max_drop_path * factor

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.ema.update(self.model)

    # --- mixup helper ---
    def _mixup(self, x, y):  # x:(B,C,T), y:(B,)
        lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[perm]
        num_classes = args.num_classes
        y1 = F.one_hot(y, num_classes=num_classes).float()
        y2 = F.one_hot(y[perm], num_classes=num_classes).float()
        y_mix = lam * y1 + (1 - lam) * y2
        return x, y_mix

    def _predict_model(self, mode):
        # use EMA model for val/test if enabled
        return self.ema.ema_model if (self.use_ema and mode in ["val", "test"]) else self.model

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)
        model = self._predict_model(mode)

        if mode == "train" and self.use_mixup:
            data, soft_targets = self._mixup(data, labels)
            preds = model(data)
            loss = self.criterion_soft(preds, soft_targets)
        else:
            preds = model(data)
            loss = self.criterion_hard(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).detach().cpu())
            self.test_targets.append(labels.detach().cpu())
        return loss

    # --- Lightning steps ---
    def training_step(self, batch, batch_idx):
        if args.use_sam:
            # with SAM: two forward-backward passes
            optimizer = self.optimizers()
            loss = self._calculate_loss(batch, mode="train")
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)
            loss2 = self._calculate_loss(batch, mode="train")
            self.manual_backward(loss2)
            optimizer.second_step(zero_grad=True)
            return loss2.detach()
        else:
            loss = self._calculate_loss(batch, mode="train")
            return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            test_preds = torch.cat(self.test_preds)
            test_targets = torch.cat(self.test_targets)
            cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())
            print("Confusion Matrix:\n", cm)


# =====================
#          Train
# =====================

def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[pretrain_checkpoint_callback, LearningRateMonitor("epoch"), TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(42)
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)
    return pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, LearningRateMonitor("epoch"), TQDMProgressBar(refresh_rate=500)],
        enable_model_summary=True,
        precision="32-true",
        gradient_clip_val=1.0,
        detect_anomaly=False,
        fast_dev_run=False,
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()

    if args.use_sam:
        model.automatic_optimization = False

    trainer.fit(model, train_loader, val_loader)

    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)
    return model, acc_result, f1_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--name', type=str, default='ICBpp_实验')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.3)           # stronger dropout
    parser.add_argument('--patch_size', type=int, default=60)                 # your good setting
    parser.add_argument('--drop_path_max', type=float, default=0.2)           # max DropPath prob

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False)
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=False)                # 默认关闭 ASB
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Our new toggles:
    parser.add_argument('--use_bn_in_icb', type=str2bool, default=True)
    parser.add_argument('--use_layerscale', type=str2bool, default=True)
    parser.add_argument('--layerscale_init', type=float, default=1e-4)
    parser.add_argument('--use_se', type=str2bool, default=True)
    parser.add_argument('--se_reduction', type=int, default=8)
    parser.add_argument('--pooling', type=str, default='attn', choices=['mean', 'attn'])
    parser.add_argument('--attn_reduction', type=int, default=8)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--noise_std', type=float, default=0.05)

    parser.add_argument('--use_ema', type=str2bool, default=True)
    parser.add_argument('--ema_decay', type=float, default=0.999)

    parser.add_argument('--use_sam', type=str2bool, default=False)
    parser.add_argument('--sam_rho', type=float, default=0.05)

    parser.add_argument('--mixup_alpha', type=float, default=0.0)  # 0: off

    args = parser.parse_args()

    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    run_description = f"实验描述{args.name}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"/tf_logs/0823_store_result/{args.name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    save_copy_of_files(pretrain_checkpoint_callback)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # dataset characteristics ...
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file...
    text_save_dir = "/tf_logs/textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    with open(f"{text_save_dir}/{args.model_id}.txt", 'a') as f:
        f.write(run_description + "  \n")
        f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
        f.write('acc:{}, mf1:{}' .format(acc_results, f1_results))
        f.write('\n\n')
