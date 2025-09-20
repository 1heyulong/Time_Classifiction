# tslanet_binary_shapelet.py
import argparse
import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


# -------------------------------
# Modules
# -------------------------------
class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)           # [B, C, N]
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)  # symmetric gating
        return out.transpose(1, 2)      # [B, N, C]

# 对X进行分块处理
class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 3
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, T]
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


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
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in):
        # not used when ASB=False，但保留兼容
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
            x_weighted = x_weighted + x_masked * weight_high
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        return x.to(dtype).view(B, N, C)


class TSLANet_layer(L.LightningModule):
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


# -------------------------------
# Binary Shapelet Head (Theft vs Normal)
# -------------------------------
class BinaryShapeletHead(nn.Module):
    """
    - 学习 K 个 shapelet（模板）: [K, D, L]
    - 计算滑窗余弦相似度 → 对时间取 max → 对 K 取 max：得到 s \in R (每样本一个分数)
    - 训练阶段：只用辅助损失，不直接写入 logits（避免干扰主干）
    - 推理/训练都支持：learnable gate g，根据主干置信度与 s 进行软融合
    """
    def __init__(self, embed_dim: int, num_classes: int,
                 K: int = 3, L_tokens: int = 7, metric: str = "cos", gate_hidden: int = 16):
        super().__init__()
        assert num_classes >= 2, "BinaryShapeletHead assumes at least binary classification."
        self.num_classes = num_classes
        self.K = K
        self.L = L_tokens
        self.metric = metric.lower()
        assert self.metric in ["cos", "xcorr"]
        # shapelets: [K, L, D] -> conv1d needs [K, D, L]
        self.shapelets = nn.Parameter(torch.randn(K, embed_dim, L_tokens) * 0.02)
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))
        self.gate = nn.Sequential(
            nn.Linear(2, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )
        self.register_buffer("ones_kernel", torch.ones(1, 1, L_tokens), persistent=False)
    
    @staticmethod
    def _confidence_from_logits(logits):
        # 置信度：max_softmax - second_max_softmax（更稳定于 logit 差）
        prob = logits.softmax(dim=-1)
        top2, _ = torch.topk(prob, k=2, dim=-1)
        conf = (top2[:,0] - top2[:,1]).unsqueeze(-1)  # [B,1]
        return conf

    def compute_score(self, tokens):
        """
        tokens: [B, N, D]
        返回: s ∈ [B,1] 以及时间/shapelet维度的中间图（可选返还）
        """
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2)  # [B, D, N]
        if self.metric == "cos":
            shp = F.normalize(self.shapelets, dim=(1,2), eps=1e-6)  # [K, D, L]
            num = F.conv1d(x, shp, bias=None, stride=1, padding=0)  # [B, K, N-L+1]
            x2 = x.pow(2).sum(dim=1, keepdim=True)                  # [B,1,N]
            win_l2 = torch.sqrt(F.conv1d(x2, self.ones_kernel.to(x.device), stride=1, padding=0).clamp_min(1e-6))
            sim = num / win_l2                                       # [B,K,T]
            score_time, _ = sim.max(dim=-1)                          # [B,K]
        else:
            score_map = F.conv1d(x, self.shapelets, bias=None, stride=1, padding=0)  # [B,K,T]
            score_time, _ = score_map.max(dim=-1)
        s, _ = score_time.max(dim=-1)   # [B]
        return s.unsqueeze(-1)          # [B,1]

    def aux_margin_loss(self, s_detached, labels, pos_margin=0.35, neg_margin=0.25):
        """
        s_detached: [B,1]  不回传到 tokens（只更新 shapelets）
        labels: int64
        让正样本 s ≥ m_pos，负样本 s ≤ m_neg
        """
        y = labels.view(-1,1).float()
        pos_loss = F.relu(pos_margin - s_detached) * y
        neg_loss = F.relu(s_detached - neg_margin) * (1. - y)
        # 平衡系数：根据 batch 内正负占比自动加权
        pos_w = (y.mean() + 1e-6)
        neg_w = (1 - y).mean() + 1e-6
        loss = (pos_loss.sum() / (pos_w * s_detached.numel())
               +neg_loss.sum() / (neg_w * s_detached.numel()))
        return loss

    def fuse(self, tokens, backbone_logits):
        """
        计算 s，并用可学习门控 g 融合：
        fused = (1-g)*backbone + g*shapelet_logits
        """
        s = self.compute_score(tokens)                    # [B,1]
        pos_idx = 1 if backbone_logits.shape[-1]==2 else (backbone_logits.shape[-1]-1)
        shapelet_logits = torch.zeros_like(backbone_logits)
        shapelet_logits[:, pos_idx] = self.scale * s.squeeze(-1) + self.bias
        conf = self._confidence_from_logits(backbone_logits)  # [B,1]
        g = self.gate(torch.cat([conf, s], dim=-1))           # [B,1]
        fused = (1. - g) * backbone_logits + g * shapelet_logits
        return fused, s, g




# -------------------------------
# Backbone + optional Shapelet Head
# -------------------------------
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

        dpr = [x.item() for x in torch.linspace(0.1, args.dropout_rate, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i])
            for i in range(args.depth)
        ])

        self.head = nn.Linear(args.emb_dim, args.num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # binary shapelet head
        self.use_shapelet = args.use_shapelet_head
        if self.use_shapelet:
            self.shapelet_head = BinaryShapeletHead(
                embed_dim=args.emb_dim,
                num_classes=args.num_classes,
                K=args.shapelet_per_class,
                L_tokens=args.shapelet_len_tokens,
                metric=args.shapelet_metric,
                gate_hidden=args.shapelet_gate_hidden
            )

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
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)
        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()
        for blk in self.tsla_blocks:
            x_masked = blk(x_masked)
        return x_masked, x_patched

    def forward_tokens(self, x):
        # return token features [B, N, D]
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.tsla_blocks:
            x = blk(x)
        return x
    def logits_from_tokens(self, tokens):
        cls_feat = tokens.mean(1)
        return self.head(cls_feat)

    # ------- 推理接口：默认做融合（训练时也可用） -------
    def forward(self, x):
        tokens = self.forward_tokens(x)
        logits_main = self.logits_from_tokens(tokens)
        if not self.use_shapelet:
            return logits_main
        fused, _, _ = self.shapelet_head.fuse(tokens, logits_main)
        return fused

    # ------- 训练用的辅助接口 -------
    def forward_with_aux(self, x, labels=None):
        tokens = self.forward_tokens(x)
        logits_main = self.logits_from_tokens(tokens)
        if not self.use_shapelet:
            return logits_main, None, None, tokens
        # 融合
        fused, s, g = self.shapelet_head.fuse(tokens, logits_main)
        # 辅助损失（只更新 shapelets；不反传到 tokens / backbone）
        aux_loss = None
        if labels is not None:
            with torch.no_grad():
                tokens_detached = tokens.detach()
            s_detached = self.shapelet_head.compute_score(tokens_detached)
            aux_loss = self.shapelet_head.aux_margin_loss(
                s_detached, labels,
                pos_margin=args.shapelet_pos_margin,
                neg_margin=args.shapelet_neg_margin
            )
        return fused, aux_loss, (s, g), tokens



# -------------------------------
# Lightning training
# -------------------------------
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


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.test_preds, self.test_targets = [], []

        # 是否只训练 shapelet 头（后期校准）
        self.posthoc = args.shapelet_posthoc

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # posthoc 情况下：只训练 shapelet 头的参数
        if self.posthoc and self.model.use_shapelet:
            for p in self.model.parameters(): p.requires_grad_(False)
            for p in self.model.shapelet_head.parameters(): p.requires_grad_(True)

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=args.train_lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _step(self, batch, mode="train"):
        x, labels = batch[0], batch[1].to(torch.int64)

        if self.model.use_shapelet:
            preds, aux_loss, (s, g), _ = self.model.forward_with_aux(x, labels)
        else:
            preds, aux_loss, s, g = self.model(x), None, None, None

        ce = self.criterion(preds, labels)
        loss = ce
        if aux_loss is not None:
            loss = ce + args.shapelet_aux_weight * aux_loss

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1  = self.f1(preds, labels)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc",  acc,  on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_f1",   f1,   on_step=False, on_epoch=True, prog_bar=True)

        if s is not None:
            # 监控 shapelet 行为（均值），便于诊断
            pos_mask = (labels==1)
            neg_mask = (labels==0)
            if pos_mask.any():
                self.log(f"{mode}_s_pos", s[pos_mask].mean(), on_step=False, on_epoch=True)
            if neg_mask.any():
                self.log(f"{mode}_s_neg", s[neg_mask].mean(), on_step=False, on_epoch=True)
            self.log(f"{mode}_gate_mean", g.mean(), on_step=False, on_epoch=True)

        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            test_preds = torch.cat(self.test_preds)
            test_targets = torch.cat(self.test_targets)
            cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())
            print("Confusion Matrix:\n", cm)


# -------------------------------
# Train / Eval helpers
# -------------------------------
def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH, accelerator="auto", devices=1,
        num_sanity_val_steps=0, max_epochs=PRETRAIN_MAX_EPOCHS,
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
        default_root_dir=CHECKPOINT_PATH, accelerator="auto", devices=1,
        num_sanity_val_steps=0, max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, LearningRateMonitor("epoch"), TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(42)
    if args.load_from_pretrained and pretrained_model_path:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader, val_loader)
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}
    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)
    return model, acc_result, f1_result


# -------------------------------
# Main
# -------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--name', type=str, default='ICB_Only_With_BinaryShapelet')

    # Training
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--pretrain_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Model
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=7)

    # Components
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False)
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=False)  # 你当前最佳：只保留 ICB
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Shapelet Head (binary)
    parser.add_argument('--use_shapelet_head', type=str2bool, default=True)
    parser.add_argument('--shapelet_per_class', type=int, default=3)
    parser.add_argument('--shapelet_len_tokens', type=int, default=7)
    parser.add_argument('--shapelet_metric', type=str, default='cos', choices=['cos', 'xcorr'])
    parser.add_argument('--shapelet_gate_hidden', type=int, default=16)

    
    parser.add_argument('--shapelet_aux_weight', type=float, default=0.2)  # 辅助损失权重
    parser.add_argument('--shapelet_pos_margin', type=float, default=0.35) # s>=m_pos for positives
    parser.add_argument('--shapelet_neg_margin', type=float, default=0.25) # s<=m_neg for negatives
    parser.add_argument('--shapelet_posthoc', type=str2bool, default=False) # True: 只训练 shapelet 头（冻结主干）


    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs

    run_description = f"实验描述{args.name}"
    print(f"========== {run_description} ===========")
    print("DATASET_PATH:", DATASET_PATH)

    CHECKPOINT_PATH = f"/tf_logs/shapelet_binary/{args.name}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1, filename='pretrain-{epoch}', monitor='val_loss', mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1, monitor='val_loss', mode='min'
    )

    save_copy_of_files(pretrain_checkpoint_callback)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataloaders (unchanged)
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # Dataset info
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    print("num_classes:", args.num_classes)
    if args.use_shapelet_head and args.num_classes < 2:
        raise ValueError("Binary shapelet head requires at least 2 classes (normal vs theft).")

    # Pretrain (optional)
    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    # Train & Eval
    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # Append result to text
    text_save_dir = "/tf_logs/textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    with open(f"{text_save_dir}/{args.model_id}.txt", 'a') as f:
        f.write(run_description + "  \n")
        f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
        f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
        f.write('\n\n')
