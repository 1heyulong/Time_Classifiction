#!/usr/bin/env python3
"""
Enhanced TSLANet training script
- Adds: temporal & frequency augmentations, mixup, NT-Xent contrastive loss (two-view), Attention Pooling, Focal Loss option,
  CosineAnnealingWarmRestarts / OneCycle schedulers, optional pretrain freeze, val_f1 checkpointing.

Dependencies: lightning (pytorch-lightning renamed 'lightning' in your original), torch, timm, torchmetrics, sklearn
Assumes you already have `dataloader.get_datasets`, `utils.get_clf_report`, `utils.save_copy_of_files`, `utils.str2bool`,
and `utils.random_masking_3D` in your repo like before.

Save as TSLANet_enhanced.py and run similarly to previous script.
"""

import argparse
import datetime
import os
from typing import Optional, Tuple

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

# External helpers from your repo
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D

# ----------------------------- Helper modules & losses -----------------------------

class AttentionPooling(nn.Module):
    """Simple attention pooling over time dimension. Input: [B, T, C] -> output [B, C]
    """
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        # x: [B, T, C]
        scores = self.attn(x)  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)
        out = (weights * x).sum(dim=1)
        return out


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: logits [B, C], targets: [B]
        logpt = -nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy loss for contrastive learning between two views.
    Expects embeddings from two views: z1, z2 [B, D]
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        # z1, z2: [B, D]
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        sim = torch.matmul(z, z.T)  # [2B, 2B]
        sim = sim / (self.temperature)

        # create mask to remove similarity with self
        labels = torch.arange(batch_size, device=z.device)
        positives = torch.cat([labels + batch_size, labels], dim=0)

        # mask self
        mask = (~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)).float()
        exp_sim = torch.exp(sim) * mask
        denom = exp_sim.sum(dim=1)

        positive_sim = torch.exp(torch.sum(z * torch.roll(z, batch_size, dims=0), dim=1) / self.temperature)
        loss = -torch.log(positive_sim / denom)
        return loss.mean()


# ----------------------------- Augmentations -----------------------------

def freq_jitter(x: torch.Tensor, sigma=0.02):
    """Apply a small multiplicative jitter to the FFT magnitude.
    x: [B, T, C] (float)
    """
    x_f = torch.fft.rfft(x, dim=1)
    mag = torch.abs(x_f)
    phase = torch.angle(x_f)
    noise = 1.0 + sigma * torch.randn_like(mag)
    mag = mag * noise
    x_f2 = mag * torch.exp(1j * phase)
    x2 = torch.fft.irfft(x_f2, n=x.shape[1], dim=1)
    return x2.type_as(x)


def temporal_crop_or_shift(x: torch.Tensor, max_shift=10):
    """Randomly shift sequences along time (circular) or crop jitter.
    x: [B, C, T] or [B, T, C] -> returns same shape as input
    """
    preserve_shape = x.shape
    if x.ndim == 3 and preserve_shape[1] < preserve_shape[2]:
        # treat as [B, C, T]
        x = x
    # convert to [B, T, C]
    if x.shape[1] == preserve_shape[-1]:
        # already [B, T, C]
        xt = x
    else:
        xt = x.transpose(1, 2)

    B, T, C = xt.shape
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        out = xt
    else:
        out = torch.roll(xt, shifts=shift, dims=1)
    # back to original
    if x.shape[1] == preserve_shape[-1]:
        return out
    else:
        return out.transpose(1, 2)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ----------------------------- Model blocks (adapted) -----------------------------

class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # input expected [B, C, T]
        x_out = self.proj(x).flatten(2).transpose(1, 2)
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

    def forward(self, x_in):
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


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        self.input_layer = nn.Linear(args.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # pooling choice
        if args.use_attn_pool:
            self.pool = AttentionPooling(args.emb_dim)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        # x: [B, C, T]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        # x: [B, P, C]
        if args.use_attn_pool:
            x = self.pool(x)  # returns [B, C]
        else:
            # AdaptiveAvgPool1d expects [B, C, L]
            x_t = x.transpose(1, 2)
            x = self.pool(x_t).squeeze(-1)
        x = self.head(x)
        return x


# ----------------------------- Lightning Module -----------------------------

class ModelTraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)

        # loss choice
        if args.use_focal:
            self.criterion = FocalLoss(gamma=args.focal_gamma, alpha=None)
        else:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

        self.contrastive = NTXentLoss(temperature=args.contrast_temp)

        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)

        if args.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_total_loss"}
            }
        elif args.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0, T_mult=1)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": 'epoch'}}
        elif args.scheduler == 'onecycle':
            # requires setting max_lr via args.train_lr
            scheduler = OneCycleLR(optimizer, max_lr=args.train_lr, total_steps=None, epochs=MAX_EPOCHS, steps_per_epoch=len(train_loader))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": 'step'}}
        else:
            return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]  # expected [B, C, T]
        labels = batch[1].to(torch.int64)

        # Two-view contrastive: produce z1, z2 embeddings (if enabled)
        if mode == 'train' and args.use_contrastive:
            # create augmented view
            x1 = data
            x2 = data.clone()
            # simple augmentations for contrastive
            if args.augment_freq:
                x2 = freq_jitter(x2, sigma=args.freq_sigma)
            if args.augment_shift:
                x2 = temporal_crop_or_shift(x2.transpose(1, 2)).transpose(1, 2) if x2.ndim==3 else x2

            # forward until pooling layer to get embeddings (use model without final head)
            z1 = self._encode(x1)
            z2 = self._encode(x2)
            c_loss = self.contrastive(z1, z2)
        else:
            c_loss = torch.tensor(0.0, device=self.device)

        # optionally mixup
        if mode == 'train' and args.use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(data, labels, args.mixup_alpha)
            preds = self.model(mixed_x)
            # mixup loss
            loss_a = self.criterion(preds, y_a)
            loss_b = self.criterion(preds, y_b)
            cls_loss = lam * loss_a + (1 - lam) * loss_b
        else:
            preds = self.model(data)
            cls_loss = self.criterion(preds, labels)

        # class weighting by empirical distribution (optional)
        if args.weight_by_freq:
            # compute simple class weighting (inverse freq)
            counts = torch.bincount(labels, minlength=args.num_classes).float()
            inv = (counts.max() - counts + 1.0)
            inv = inv / inv.sum()
            # weight positive/neg via scaling on loss per sample
            sample_weights = inv[labels].to(preds.device)
            cls_loss = (cls_loss * sample_weights).mean() if cls_loss.dim() > 0 else cls_loss

        # combine losses
        total_loss = args.cls_loss_weight * cls_loss + args.contrastive_weight * c_loss

        # metrics
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        self.log(f"{mode}_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_cls_loss", cls_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        if args.use_contrastive:
            self.log(f"{mode}_contrastive_loss", c_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())

        return total_loss

    def _encode(self, x):
        # get embedding vector before head
        # replicate TSLANet forward but return pooled embedding
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        for tsla_blk in self.model.tsla_blocks:
            x = tsla_blk(x)
        if args.use_attn_pool:
            z = self.model.pool(x)
        else:
            z = self.model.pool(x.transpose(1, 2)).squeeze(-1)
        return z

    def training_step(self, batch, batch_idx):
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


# ----------------------------- Training entrypoint -----------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TSLANetEnhanced')
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/dataset_rate_0605_realy/')
    parser.add_argument('--name', type=str, default='增强版实验')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--patch_size', type=int, default=7)

    # TSLANet components:
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # New options:
    parser.add_argument('--use_attn_pool', type=str2bool, default=True)
    parser.add_argument('--use_focal', type=str2bool, default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--use_contrastive', type=str2bool, default=True)
    parser.add_argument('--contrast_temp', type=float, default=0.2)
    parser.add_argument('--contrastive_weight', type=float, default=0.5)

    parser.add_argument('--use_mixup', type=str2bool, default=True)
    parser.add_argument('--mixup_alpha', type=float, default=0.2)

    parser.add_argument('--augment_freq', type=str2bool, default=True)
    parser.add_argument('--freq_sigma', type=float, default=0.02)
    parser.add_argument('--augment_shift', type=str2bool, default=True)

    parser.add_argument('--use_mixup_infer', type=str2bool, default=False)
    parser.add_argument('--weight_by_freq', type=str2bool, default=False)

    parser.add_argument('--cls_loss_weight', type=float, default=1.0)

    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine', 'onecycle', 'none'])
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--cosine_T0', type=int, default=10)

    parser.add_argument('--use_gpu', type=str2bool, default=True)

    args = parser.parse_args()

    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs

    # load datasets ...
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    # checkpointing
    CHECKPOINT_PATH = f"./checkpoints/{args.name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='best-{epoch}-{val_total_loss:.4f}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max'
    )

    save_copy_of_files(checkpoint_callback)

    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # logger
    tensorboard_logger = TensorBoardLogger(save_dir='./tb_logs', name=args.name)

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator='auto',
        devices=1 if args.use_gpu else None,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor('epoch'),
            TQDMProgressBar(refresh_rate=200),
            EarlyStopping(monitor='val_f1', patience=30, mode='max')
        ],
        logger=tensorboard_logger,
        num_sanity_val_steps=0
    )

    L.seed_everything(42)

    model = ModelTraining()

    trainer.fit(model, train_loader, val_loader)

    # load best
    best_path = checkpoint_callback.best_model_path
    if best_path:
        trained = ModelTraining.load_from_checkpoint(best_path)
    else:
        trained = model

    val_result = trainer.test(trained, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(trained, dataloaders=test_loader, verbose=False)

    acc_result = {"test": test_result[0].get("test_acc", None), "val": val_result[0].get("test_acc", None)}
    f1_result = {"test": test_result[0].get("test_f1", None), "val": val_result[0].get("test_f1", None)}

    print("ACC results", acc_result)
    print("F1  results", f1_result)

    text_save_dir = "./textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(f"{datetime.datetime.now()} \t {args.name} \n")
    f.write('acc:{}, mf1:{}\n'.format(acc_result, f1_result))
    f.write('\n')
    f.close()
