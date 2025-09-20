# tslanet_minor_shapelet_train.py
# ===============================================
# 直接运行示例：
# python tslanet_minor_shapelet_train.py \
#   --data_path data/hhar \
#   --name ICB_ShapeletMinor \
#   --ICB True --ASB False \
#   --use_shapelet_head True \
#   --shapelet_len 3 --K_normal 4 --K_anomaly 6 \
#   --shapelet_init kmeans_anomaly_only \
#   --patch_size 60 --emb_dim 128 --depth 2 \
#   --num_epochs 300 --batch_size 32 \
#   --train_lr 1e-3 --weight_decay 5e-4
# ===============================================

import argparse
import os
import numpy as np
from typing import Tuple

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

# === 你自己的模块：保持不变 ===
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


# =============== 基础模块：ICB / PatchEmbed ===============
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # -> [B, D, T]
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2)
        return out.transpose(1, 2)  # -> [B, T, D]


class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, L]
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, T, D]


# =============== 少类 Shapelet Head ===============
class ShapeletAttentionHeadMinor(nn.Module):
    """
    两个 shapelet 池：Normal & Anomaly
    - shapelet 维度：[K, D, L]；与 patch 特征 [B, T, D] 在 D 上对应，沿 T 方向做 conv1d 匹配。
    """
    def __init__(self, D, num_classes=2, L=3, K_normal=4, K_anomaly=4):
        super().__init__()
        self.L = L
        self.D = D
        self.Kn = K_normal
        self.Ka = K_anomaly

        self.shapelets_normal = nn.Parameter(torch.randn(K_normal, D, L))
        self.shapelets_anomaly = nn.Parameter(torch.randn(K_anomaly, D, L))
        nn.init.xavier_uniform_(self.shapelets_normal)
        nn.init.xavier_uniform_(self.shapelets_anomaly)

        self.classifier = nn.Linear(K_normal + K_anomaly, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)

    @torch.no_grad()
    def init_with_centers(self, centers_normal: torch.Tensor = None, centers_anomaly: torch.Tensor = None):
        """
        用外部给定的中心初始化 shapelet 池。形状必须是 [K, D, L]。
        """
        if centers_normal is not None and centers_normal.shape == self.shapelets_normal.shape:
            self.shapelets_normal.copy_(centers_normal)
        if centers_anomaly is not None and centers_anomaly.shape == self.shapelets_anomaly.shape:
            self.shapelets_anomaly.copy_(centers_anomaly)

    def _conv_sim(self, x_bdt, shapelets_kdl):
        # x_bdt: [B, D, T] ; shapelets_kdl: [K, D, L]
        B, D, T = x_bdt.shape
        K, D2, L = shapelets_kdl.shape
        assert D == D2, "D mismatch between features and shapelets"
        sims = []
        for k in range(K):
            w = shapelets_kdl[k].unsqueeze(0)  # [1, D, L]
            sim = F.conv1d(x_bdt, w, stride=1)  # [B, 1, T-L+1]
            sims.append(sim.squeeze(1))         # [B, T']
        return torch.stack(sims, dim=1)         # [B, K, T']

    def forward(self, x_btd):
        # x_btd: [B, T, D]
        x_bdt = x_btd.transpose(1, 2)  # -> [B, D, T]
        sims_n = self._conv_sim(x_bdt, self.shapelets_normal)   # [B, Kn, T']
        sims_a = self._conv_sim(x_bdt, self.shapelets_anomaly)  # [B, Ka, T']

        feat_n = sims_n.max(dim=-1)[0]  # [B, Kn]
        feat_a = sims_a.max(dim=-1)[0]  # [B, Ka]
        feat = torch.cat([feat_n, feat_a], dim=-1)  # [B, Kn+Ka]
        logits = self.classifier(feat)              # [B, 2]
        return logits, feat_n, feat_a


# =============== 主干：TSLANet（ICB-only + DropPath + LS） ===============
class TSLANet(nn.Module):
    def __init__(self, seq_len, num_channels, num_classes=2,
                 emb_dim=128, depth=2, patch_size=60, dropout_rate=0.15,
                 use_shapelet_head=False, shapelet_len=3, K_normal=4, K_anomaly=6):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0.1, dropout_rate, depth)]
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                ICB(emb_dim, int(emb_dim * 3), drop=dropout_rate),
                DropPath(dpr[i])
            ) for i in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        self.use_shapelet_head = use_shapelet_head
        if use_shapelet_head:
            self.shapelet_head = ShapeletAttentionHeadMinor(
                D=emb_dim, num_classes=num_classes,
                L=shapelet_len, K_normal=K_normal, K_anomaly=K_anomaly
            )
        else:
            self.head = nn.Linear(emb_dim, num_classes)

    def encode(self, x):
        """返回 patch-level 表征 [B, T, D]（供 shapelet 匹配/初始化使用）"""
        x = self.patch_embed(x)        # [B, T, D]
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)
        return x                        # [B, T, D]

    def forward(self, x):
        x = self.encode(x)
        if self.use_shapelet_head:
            logits, feat_n, feat_a = self.shapelet_head(x)
            return logits, feat_n, feat_a
        else:
            return self.head(x.mean(dim=1))


# =============== Lightning 模型封装（训练/验证/测试） ===============
class LitModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.model = TSLANet(
            seq_len=args.seq_len,
            num_channels=args.num_channels,
            num_classes=args.num_classes,
            emb_dim=args.emb_dim,
            depth=args.depth,
            patch_size=args.patch_size,
            dropout_rate=args.dropout_rate,
            use_shapelet_head=args.use_shapelet_head,
            shapelet_len=args.shapelet_len,
            K_normal=args.K_normal,
            K_anomaly=args.K_anomaly
        )

        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.test_preds, self.test_targets = [], []

    # ------------- shapelet 初始化（可选）-------------
    @torch.no_grad()
    def _collect_windows(self, x_btd, L, max_windows_per_series=4):
        """
        x_btd: [B, T, D] -> 返回 [N, D, L]
        这里随机抽取几个窗口，避免生成过多切片。
        """
        B, T, D = x_btd.shape
        if T < L:
            return torch.empty(0, D, L, device=x_btd.device)
        starts_all = torch.randint(0, T - L + 1, (B, max_windows_per_series), device=x_btd.device)
        out = []
        for b in range(B):
            for s in starts_all[b].tolist():
                out.append(x_btd[b, s:s+L, :].transpose(0, 1))  # [D, L]
        if len(out) == 0:
            return torch.empty(0, D, L, device=x_btd.device)
        return torch.stack(out, dim=0)  # [N, D, L]

    @torch.no_grad()
    def init_shapelets_from_loader(self, train_loader, device):
        """
        三种初始化模式：
        - random：不做任何事
        - kmeans_both：对两类分别做 KMeans
        - kmeans_anomaly_only：只对 anomaly（label=1）做 KMeans，normal 保持随机或也做 KMeans（可选）
        """
        if not self.args.use_shapelet_head:
            return

        mode = self.args.shapelet_init
        if mode == "random":
            print("[ShapeletInit] random (skip)")
            return

        print(f"[ShapeletInit] {mode} ...")
        self.model.eval()
        device = device or self.device

        Lw = self.args.shapelet_len
        max_batches = self.args.shapelet_init_max_batches
        max_windows_per_series = self.args.shapelet_init_windows_per_series
        max_samples_per_class = self.args.shapelet_init_samples_per_class

        feats_norm, feats_anom = [], []
        seen_batches = 0

        for batch in train_loader:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device).long()
            with torch.no_grad():
                x_btd = self.model.encode(data)  # [B, T, D]
                win_bdl = self._collect_windows(x_btd, L=Lw, max_windows_per_series=max_windows_per_series)  # [N, D, L]
            if win_bdl.numel() == 0:
                continue

            # 根据 labels 将窗口分到不同池（简单起见：本 batch 的窗口全按样本标签划分）
            # 更精细做法：逐样本拆分，但当前随机窗口数很少，影响不大
            B = data.size(0)
            idx = 0
            for b in range(B):
                cls = int(labels[b].item())
                take = min(max_windows_per_series, max(0, x_btd.size(1) - Lw + 1))
                if take == 0:
                    continue
                sel = win_bdl[idx:idx+take]  # [take, D, L]
                idx += take
                if cls == 1:
                    feats_anom.append(sel.cpu())
                else:
                    feats_norm.append(sel.cpu())

            seen_batches += 1
            if (seen_batches >= max_batches) or \
               (sum(x.shape[0] for x in feats_anom) >= max_samples_per_class) or \
               (sum(x.shape[0] for x in feats_norm) >= max_samples_per_class):
                break

        def _stack_limit(lst, limit):
            if len(lst) == 0:
                return None
            arr = torch.cat(lst, dim=0)  # [N, D, L]
            if arr.size(0) > limit:
                idx = torch.randperm(arr.size(0))[:limit]
                arr = arr[idx]
            return arr

        arr_norm = _stack_limit(feats_norm, max_samples_per_class)
        arr_anom = _stack_limit(feats_anom, max_samples_per_class)

        # 执行 KMeans（在 CPU 上）
        def _kmeans_centers(arr, K):
            if arr is None or arr.size(0) < K:
                return None
            X = arr.reshape(arr.size(0), -1).numpy()
            km = KMeans(n_clusters=K, n_init='auto', random_state=42)
            km.fit(X)
            centers = torch.from_numpy(km.cluster_centers_).float().reshape(K, arr.size(1), arr.size(2))
            return centers

        centers_n = None
        centers_a = None

        if mode == "kmeans_both":
            centers_n = _kmeans_centers(arr_norm, self.args.K_normal)
            centers_a = _kmeans_centers(arr_anom, self.args.K_anomaly)
        elif mode == "kmeans_anomaly_only":
            centers_a = _kmeans_centers(arr_anom, self.args.K_anomaly)
            # 可选：也对 normal 做 kmeans；这里保持随机初始化即可
            # centers_n = _kmeans_centers(arr_norm, self.args.K_normal)

        self.model.shapelet_head.init_with_centers(
            centers_normal=None if centers_n is None else centers_n.to(device),
            centers_anomaly=None if centers_a is None else centers_a.to(device)
        )
        print("[ShapeletInit] done.")

    # ------------- Lightning 标准接口 -------------
    def forward(self, x):
        return self.model(x)

    def _contrastive_minor_loss(self, feat_n, feat_a, labels):
        # 仅在少类样本上启用 margin 对比：max(sim_anom) - max(sim_norm) >= m
        anom_mask = (labels == 1)
        if anom_mask.sum() == 0:
            return torch.tensor(0., device=self.device)
        pos = feat_a[anom_mask].max(dim=1)[0].mean()  # anomaly 组的最大相似度（平均）
        neg = feat_n[anom_mask].max(dim=1)[0].mean()
        margin = self.args.shapelet_margin
        return F.relu(margin - (pos - neg))

    def training_step(self, batch, batch_idx):
        data, labels = batch
        labels = labels.long()

        if self.args.use_shapelet_head:
            preds, feat_n, feat_a = self.model(data)
            ce = self.criterion(preds, labels)
            # 对比损失（只针对少类）
            cl = self._contrastive_minor_loss(feat_n, feat_a, labels)
            # 线性预热
            warm = min(1.0, (self.current_epoch + 1) / max(1, self.args.shapelet_contrast_warmup))
            loss = ce + warm * self.args.shapelet_contrast_weight * cl
        else:
            preds = self.model(data)
            loss = self.criterion(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        labels = labels.long()
        if self.args.use_shapelet_head:
            preds, _, _ = self.model(data)
        else:
            preds = self.model(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, labels = batch
        labels = labels.long()
        if self.args.use_shapelet_head:
            preds, _, _ = self.model(data)
        else:
            preds = self.model(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)
        self.test_preds.append(preds.argmax(dim=-1).cpu())
        self.test_targets.append(labels.cpu())
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            y_pred = torch.cat(self.test_preds)
            y_true = torch.cat(self.test_targets)
            cm = confusion_matrix(y_true.numpy(), y_pred.numpy())
            print("Confusion Matrix:\n", cm)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.train_lr, weight_decay=self.args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }


# ================= 主训练流程（基本不改你的外层脚手架） =================
def pretrain_model(args, CHECKPOINT_PATH, train_loader, val_loader, pretrain_checkpoint_callback):
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

    # 简化：用 MSE 重构做个 placeholder（如需真的预训练再补齐）
    class DummyPretrain(L.LightningModule):
        def __init__(self, args):
            super().__init__()
            self.model = TSLANet(args.seq_len, args.num_channels, args.num_classes,
                                 args.emb_dim, args.depth, args.patch_size, args.dropout_rate,
                                 use_shapelet_head=False)

        def training_step(self, batch, _):
            x, _ = batch
            z = self.model.encode(x)
            loss = (z ** 2).mean()
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)

    L.seed_everything(42)
    model = DummyPretrain(args)
    trainer.fit(model, train_loader, val_loader)
    return pretrain_checkpoint_callback.best_model_path


def train_and_test(args, CHECKPOINT_PATH, train_loader, val_loader, test_loader,
                   checkpoint_callback, pretrain_checkpoint_callback):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, LearningRateMonitor("epoch"), TQDMProgressBar(refresh_rate=500)],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)

    lit = LitModel(args)

    # （可选）使用训练集做 shapelet 初始化（不改数据逻辑，仅遍历几批）
    if args.use_shapelet_head and args.shapelet_init != "random":
        lit.init_shapelets_from_loader(train_loader, device=None)

    trainer.fit(lit, train_loader, val_loader)

    # 加载最好验证指标的权重再评估
    best = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, args=args)
    val_result = trainer.test(best, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(best, dataloaders=test_loader, verbose=False)

    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(best, test_loader, CHECKPOINT_PATH, args.class_names)
    return best, acc_result, f1_result


# ========================== 入口 ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--name', type=str, default='ICB_ShapeletMinor')

    # Train
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Model
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=60)

    # Switches
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=False)  # 保留开关但不使用
    parser.add_argument('--adaptive_filter', type=str2bool, default=False)
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False)

    # Shapelet Head 开关与超参
    parser.add_argument('--use_shapelet_head', type=str2bool, default=True)
    parser.add_argument('--shapelet_len', type=int, default=3)     # token级 shapelet 长度
    parser.add_argument('--K_normal', type=int, default=4)
    parser.add_argument('--K_anomaly', type=int, default=6)
    parser.add_argument('--shapelet_init', type=str, default='kmeans_anomaly_only',
                        choices=['random', 'kmeans_both', 'kmeans_anomaly_only'])
    parser.add_argument('--shapelet_init_max_batches', type=int, default=30)
    parser.add_argument('--shapelet_init_windows_per_series', type=int, default=4)
    parser.add_argument('--shapelet_init_samples_per_class', type=int, default=1024)

    # 少类对比损失
    parser.add_argument('--shapelet_contrast_weight', type=float, default=0.5)
    parser.add_argument('--shapelet_contrast_warmup', type=int, default=5)  # epoch 线性预热
    parser.add_argument('--shapelet_margin', type=float, default=1.0)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)
    run_description = f"实验描述 {args.name}"
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"/tf_logs/0901result/{args.name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1, filename='pretrain-{epoch}',
        monitor='val_loss', mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1,
        monitor='val_loss', mode='min'
    )

    save_copy_of_files(pretrain_checkpoint_callback)

    # 保持确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据：逻辑不改
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # 数据集属性
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    # （可选）预训练
    if args.load_from_pretrained and args.pretrain_epochs > 0:
        best_model_path = pretrain_model(args, CHECKPOINT_PATH, train_loader, val_loader, pretrain_checkpoint_callback)
    else:
        best_model_path = ''

    # 训练 + 测试
    model, acc_results, f1_results = train_and_test(
        args, CHECKPOINT_PATH,
        train_loader, val_loader, test_loader,
        checkpoint_callback, pretrain_checkpoint_callback
    )
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # 结果记录
    text_save_dir = "/tf_logs/textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    with open(f"{text_save_dir}/{args.model_id}.txt", 'a') as f:
        f.write(run_description + "  \n")
        f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
        f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
        f.write('\n\n')
