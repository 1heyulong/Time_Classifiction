import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    StochasticWeightAveraging,
    EarlyStopping,
)
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAveragePrecision,
)
from timm.models.layers import DropPath, trunc_normal_

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


# --------------------------
# Backbone modules (与原版一致，小改)
# --------------------------
class ICB(L.LightningModule):
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


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 3  # 你当前版本的设置
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, T] -> [B, N, D]
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))
        self.adaptive_filter = adaptive_filter

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
        if self.adaptive_filter:
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
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_icb=True, use_asb=True, adaptive_filter=True):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


# --------------------------
# MIL 头：patch 级logit + Top-K聚合
# --------------------------
class MILBinaryHead(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, 1)  # 输出单 logit（异常分数）

    def forward(self, x):  # x: [B, N, D]
        x = self.norm(x)
        x = self.drop(x)
        logit_patch = self.fc(x).squeeze(-1)  # [B, N]
        return logit_patch


class TSLANet_MIL(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(
                dim=args.emb_dim,
                drop=args.dropout_rate,
                drop_path=dpr[i],
                use_icb=args.ICB,
                use_asb=args.ASB,
                adaptive_filter=args.adaptive_filter
            ) for i in range(args.depth)
        ])

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

        # MIL 头
        self.patch_head = MILBinaryHead(args.emb_dim, dropout=max(0.2, args.dropout_rate))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, x):  # 返回 patch 表征 [B, N, D]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.tsla_blocks:
            x = blk(x)
        return x

    def forward(self, x):
        # 返回户级 logit 以及 patch 级 logit（训练/分析可能用得上）
        x = self.forward_backbone(x)           # [B, N, D]
        logit_patch = self.patch_head(x)       # [B, N]
        B, N = logit_patch.shape
        k = max(1, int(self.args.mil_topk_ratio * N))
        topk_vals, _ = torch.topk(logit_patch, k=k, dim=1)  # 取“最异常”的Top-K patch
        bag_logit = topk_vals.mean(dim=1)      # [B]
        return bag_logit, logit_patch


# --------------------------
# LightningModule（训练/验证/测试）
# --------------------------
class LitMIL(L.LightningModule):
    def __init__(self, args, pos_weight):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight'])
        self.args = args
        self.model = TSLANet_MIL(args)

        # 损失：二分类 BCE with logits，pos_weight = neg/pos
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # 指标（阈值相关的指标在 epoch end 用 best threshold 计算）
        self.ap = BinaryAveragePrecision()
        self.f1 = BinaryF1Score()
        self.prec = BinaryPrecision()
        self.rec = BinaryRecall()

        # 缓存用于阈值寻优与混淆矩阵
        self.val_probs, self.val_targets = [], []
        self.test_probs, self.test_targets = [], []
        self.best_threshold = 0.5  # 会在验证阶段自动更新

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.train_lr, weight_decay=self.args.weight_decay)
        # 你也可以换成 CosineWarmRestarts，这里保留 Plateau 以最小改动
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def _shared_step(self, batch, mode="train"):
        x, y = batch[0], batch[1]               # y: [B], 0/1
        bag_logit, _ = self.model(x)            # bag_logit: [B]
        loss = self.criterion(bag_logit, y.float())
        probs = torch.sigmoid(bag_logit).detach()

        # PR-AUC 用概率计算
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_ap", self.ap(probs, y), on_step=False, on_epoch=True, prog_bar=(mode!="train"), logger=True)

        if mode == "val":
            self.val_probs.append(probs.cpu())
            self.val_targets.append(y.cpu())
        elif mode == "test":
            self.test_probs.append(probs.cpu())
            self.test_targets.append(y.cpu())
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, mode="test")

    @staticmethod
    def _find_best_threshold(y_true, y_prob, beta=2.0):
        ts = np.linspace(0.01, 0.99, 99)
        scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta) for t in ts]
        return float(ts[int(np.argmax(scores))])

    def on_validation_epoch_end(self):
        if len(self.val_probs) == 0:
            return
        probs = torch.cat(self.val_probs).numpy()
        targs = torch.cat(self.val_targets).numpy()
        self.val_probs.clear(); self.val_targets.clear()

        # 按 F2 寻优，更偏召回
        best_t = self._find_best_threshold(targs, probs, beta=2.0)
        self.best_threshold = best_t
        # 用 best_t 计算阈值后的 f1/prec/rec 并记录
        preds = (probs >= best_t).astype(int)
        f1 = fbeta_score(targs, preds, beta=1.0)
        prec = (preds[targs==1].sum() / max(preds.sum(), 1)).item() if hasattr(preds, 'sum') else 0.0
        # 更稳妥使用 sklearn:
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(targs, preds, zero_division=0)
        rec = recall_score(targs, preds, zero_division=0)

        # 记录
        self.log("val_best_threshold", best_t, prog_bar=True, logger=True)
        self.log("val_f1_thr", f1, prog_bar=True, logger=True)
        self.log("val_precision_thr", prec, prog_bar=False, logger=True)
        self.log("val_recall_thr", rec, prog_bar=False, logger=True)

    def on_test_epoch_end(self):
        if len(self.test_probs) == 0:
            return
        probs = torch.cat(self.test_probs).numpy()
        targs = torch.cat(self.test_targets).numpy()
        self.test_probs.clear(); self.test_targets.clear()

        preds = (probs >= self.best_threshold).astype(int)
        cm = confusion_matrix(targs, preds)
        print(f"Using threshold={self.best_threshold:.3f}")
        print("Confusion Matrix:\n", cm)


# --------------------------
# 训练脚本
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default='data/hhar')
    parser.add_argument('--name', type=str, default='MIL_TOPK')

    # Training
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    # Model
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=90)  # 你当前最优
    parser.add_argument('--mil_topk_ratio', type=float, default=0.05)  # Top-K占比（可调 1%~10%）
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs

    CHECKPOINT_PATH = f"/tf_logs/{args.name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # Callbacks: 用 PR-AUC 作为监控更贴合少数类
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_ap',
        mode='max',
        filename='best-{epoch:03d}-{val_ap:.4f}'
    )
    lr_cb = LearningRateMonitor("epoch")
    prog_cb = TQDMProgressBar(refresh_rate=500)
    swa = StochasticWeightAveraging(swa_lrs=1e-4)
    # 可选早停（基于PR-AUC）
    early_stop = EarlyStopping(monitor="val_ap", mode="max", patience=30, verbose=True)

    save_copy_of_files(checkpoint_callback)  # 你原来的备份函数

    # 固定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(42)

    # 数据
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    args.num_classes = 2  # 二分类
    args.class_names = ['0', '1']
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    # 统计类别比例，算 pos_weight (label=1 为“异常”，少数类)
    y_train = train_loader.dataset.y_data
    counts = np.bincount(y_train)
    num_pos = counts[1] if len(counts) > 1 else 0
    num_neg = counts[0] if len(counts) > 0 else 0
    pos_weight = (num_neg / max(num_pos, 1)) if num_pos > 0 else 1.0
    print(f"Train counts: neg={num_neg}, pos={num_pos}, pos_weight={pos_weight:.3f}")

    # Trainer
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, lr_cb, prog_cb, swa, early_stop],
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    model = LitMIL(args, pos_weight=pos_weight)
    trainer.fit(model, train_loader, val_loader)

    # 用最优权重评估
    best_path = trainer.checkpoint_callback.best_model_path
    print("Best ckpt:", best_path)
    model = LitMIL.load_from_checkpoint(best_path, args=args, pos_weight=pos_weight)

    # 验证与测试
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    # 计算并打印基于阈值的 F1/Prec/Rec（由 on_*_epoch_end 打印混淆矩阵）
    print("val metrics:", val_result)
    print("test metrics:", test_result)

    # 如果你需要分类报告 get_clf_report，需要其能接受二分类概率与阈值；
    # 如当前实现仅支持多类，需要做适配（可用 sklearn.classification_report）。
    # get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)


if __name__ == "__main__":
    main()