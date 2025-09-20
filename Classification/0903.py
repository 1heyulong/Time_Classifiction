# 0902TSLANet_rawshapelet_gate.py
"""
TSLANet + Raw Shapelet (KMeans init on anomaly windows) + Fusion Gate
Usage example:
python 0902TSLANet_rawshapelet_gate.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name ICB_RawShapeletGate \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --shapelet_len 30 --K_anomaly 10 \
  --num_epochs 500 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --shapelet_init kmeans_anomaly_only --finetune_shapelets True
"""
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

# your existing modules (must be present)
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool

# -------------------------------
# ICB block (之前的ICB模块,进行保留)
# -------------------------------

class ICBBlock(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, D] -> transpose to [B, D, N] for conv1d across tokens/time
        x = x.transpose(1, 2)
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)
        return out.transpose(1, 2)


# -------------------------------
# TSLBackbone: 先进行分割,再加位置参数,再进入ICB模块,最后全局池化
# -------------------------------
class TSLANetBackbone(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim, depth, dropout_rate):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(num_channels, emb_dim, kernel_size=patch_size, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)

        dpr = [x for x in torch.linspace(0.1, dropout_rate, depth)]
        self.blocks = nn.ModuleList([
            ICBBlock(emb_dim, int(emb_dim * 3), drop=dropout_rate)
            for _ in range(depth)
        ])
        self.head_in_dim = emb_dim
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_raw):
        # x_raw: [B, C, T]
        x = self.proj(x_raw).flatten(2).transpose(1, 2)  # [B, N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = x + blk(x)
        # global pooled feature
        feat = x.mean(1)  # [B, D]
        return feat


# -------------------------------
# Raw Shapelet head: 来学习shapelet 形状的卷积核
# -------------------------------
class RawShapeletHead(nn.Module):
    def __init__(self, shapelets_np, emb_dim, finetune=True):
        """
        shapelets_np: numpy array [K, C, L_raw] or torch tensor
        emb_dim: projection dim (same as backbone embedding dim)
        finetune: whether the conv kernels are learnable
        """
        super().__init__()
        if isinstance(shapelets_np, np.ndarray):
            tensor = torch.from_numpy(shapelets_np.astype(np.float32))
        else:
            tensor = shapelets_np.float()
        K, C, L = tensor.shape
        # conv1d kernels expect shape [out_channels, in_channels, kernel_size]
        self.shapelet_kernels = nn.Parameter(tensor.clone(), requires_grad=finetune)  # [K, C, L]
        self.proj = nn.Linear(K, emb_dim)
        # small normalization layer to stabilize scale differences
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self, x_raw):
        # x_raw: [B, C, T]
        # conv1d expects [B, C, T], weight [K, C, L] -> out [B, K, T']
        sims = F.conv1d(x_raw, self.shapelet_kernels, stride=1)  # [B, K, T']
        feat = sims.max(dim=-1)[0]  # [B, K]
        # optional norm
        feat = feat / (feat.norm(dim=-1, keepdim=True) + self.eps)
        return self.proj(feat)  # [B, emb_dim]


# -------------------------------
# Combined model with fusion gate将TSLANet和Raw Shapelet特征进行融合,我感觉不是很靠谱
# -------------------------------
class CombinedModel(nn.Module):
    def __init__(self, backbone: TSLANetBackbone, shapelets_np, num_classes=2, finetune_shapelets=True):
        super().__init__()
        self.backbone = backbone
        emb_dim = backbone.head_in_dim
        self.raw_head = RawShapeletHead(shapelets_np, emb_dim, finetune=finetune_shapelets)

        # gate: input 2*D -> hidden -> 2 scores
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 2)
        )

        # final classifier (on fused feature)
        self.fused_fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x_raw):
        # x_raw: [B, C, T]
        base_feat = self.backbone(x_raw)   # [B, D]原始TSLANet的特征
        raw_feat = self.raw_head(x_raw)    # [B, D]使用shapelet卷积核得到的特征
        gate_in = torch.cat([base_feat, raw_feat], dim=-1)  # [B, 2D]
        gate_logits = self.gate_net(gate_in)  # [B, 2]
        gate_w = torch.softmax(gate_logits, dim=-1)  # [B,2]
        w_base = gate_w[:, 0:1]
        w_raw = gate_w[:, 1:2]
        fused = w_base * base_feat + w_raw * raw_feat  # [B, D]
        logits = self.fused_fc(fused)  # [B, num_classes]
        return logits, base_feat, raw_feat, gate_w


# -------------------------------
# Helper: init shapelets (KMeans on anomaly windows)
# -------------------------------
def init_shapelets_from_raw(train_loader, K_anomaly=10, shapelet_len=30,
                            max_batches=200, windows_per_series=4, max_windows_total=5000, seed=42):
    """
    Collect windows from anomaly sequences (label==1) then KMeans.
    Returns centers: np.array [K_anomaly, C, shapelet_len]
    """
    np.random.seed(seed)
    anomaly_windows = []
    used_batches = 0
    for batch in train_loader:
        x_batch, y_batch = batch  # assume x_batch: [B, C, T], y_batch: [B]
        x_np = x_batch.cpu().numpy()
        y_np = y_batch.cpu().numpy()
        B = x_np.shape[0]
        for i in range(B):
            if int(y_np[i]) == 1:
                seq = x_np[i]  # [C, T]
                T = seq.shape[1]
                step = max(1, shapelet_len // 2)
                for s in range(0, T - shapelet_len + 1, step):
                    win = seq[:, s:s + shapelet_len]  # [C, L]
                    if win.shape[1] == shapelet_len:
                        anomaly_windows.append(win.astype(np.float32))
                        if len(anomaly_windows) >= max_windows_total:
                            break
                if len(anomaly_windows) >= max_windows_total:
                    break
        used_batches += 1
        if used_batches >= max_batches:
            break

    if len(anomaly_windows) == 0:
        print("[ShapeletInit] WARNING: No anomaly windows found. Returning random kernels.")
        # Create random if none found - shape: [K_anomaly, C, L]
        # Need to infer C from train_loader dataset if possible
        sample_x, _ = next(iter(train_loader))
        C = sample_x.shape[1]
        centers = np.random.randn(K_anomaly, C, shapelet_len).astype(np.float32) * 0.02
        return centers

    windows_arr = np.stack(anomaly_windows, axis=0)  # [N, C, L]
    N = windows_arr.shape[0]
    C = windows_arr.shape[1]
    L = windows_arr.shape[2]
    print(f"[ShapeletInit] Collected {N} anomaly windows for clustering (C={C}, L={L})")

    # flatten windows for clustering
    X = windows_arr.reshape(N, -1)
    n_clusters = min(K_anomaly, N)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.reshape(n_clusters, C, L).astype(np.float32)
    print(f"[ShapeletInit] KMeans done. Returning {centers.shape[0]} centers.")
    return centers  # [K, C, L]


# -------------------------------
# Lightning module wrapping model/training
# -------------------------------

class LitModel(L.LightningModule):
    def __init__(self, args, shapelets_np):
        super().__init__()
        self.save_hyperparameters(ignore=['shapelets_np'])
        # Prepare backbone + combined model
        backbone = TSLANetBackbone(
            seq_len=args.seq_len,
            patch_size=args.patch_size,
            num_channels=args.num_channels,
            emb_dim=args.emb_dim,
            depth=args.depth,
            dropout_rate=args.dropout_rate
        )
        self.model = CombinedModel(backbone, shapelets_np, num_classes=args.num_classes,
                                   finetune_shapelets=args.finetune_shapelets)
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        logits, *_ = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _shared_step(self, batch, mode="train"):
        x, y = batch
        y = y.long()
        logits, base_feat, raw_feat, gate_w = self.model(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1(logits, y)

        # optional: log gate stats
        gate_mean = gate_w.mean(dim=0)  # [2]
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        self.log(f"{mode}_f1", f1, prog_bar=True)
        self.log(f"{mode}_gate_base", float(gate_mean[0]), on_step=False, on_epoch=True)
        self.log(f"{mode}_gate_raw", float(gate_mean[1]), on_step=False, on_epoch=True)

        if mode == "test":
            self.test_preds.append(preds.cpu())
            self.test_targets.append(y.cpu())
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="test")

    def on_test_epoch_end(self):
        if len(self.test_preds) > 0:
            y_pred = torch.cat(self.test_preds).numpy()
            y_true = torch.cat(self.test_targets).numpy()
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix (test):")
            print(cm)


# -------------------------------
# full training flow
# -------------------------------
def train_and_eval(args):
    # load datasets (unchanged)
    train_loader, val_loader, test_loader = get_datasets(args.data_path, args)

    # dataset meta
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.class_names = [str(i) for i in range(args.num_classes)]

    # init shapelets (raw-space) if needed
    if args.shapelet_init in ("kmeans_anomaly_only", "kmeans_both"):
        print("[Main] Initializing raw-space shapelets via KMeans...")
        # only anomaly used for centers by default (kmeans_anomaly_only)
        centers = init_shapelets_from_raw(train_loader,
                                          K_anomaly=args.K_anomaly,
                                          shapelet_len=args.shapelet_len,
                                          max_batches=args.shapelet_init_max_batches,
                                          windows_per_series=args.shapelet_init_windows_per_series,
                                          max_windows_total=args.shapelet_init_samples_per_class,
                                          seed=args.seed)
    else:
        # random
        print("[Main] Using random shapelet kernels.")
        sample_x, _ = next(iter(train_loader))
        C = sample_x.shape[1]
        centers = np.random.randn(args.K_anomaly, C, args.shapelet_len).astype(np.float32) * 0.02

    # create lit model
    lit = LitModel(args, centers)

    # optional: save a copy of the script and configs
    CHECKPOINT_PATH = os.path.join(args.checkpoint_root, args.name)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, save_top_k=1, monitor="val_loss", mode="min"
    )
    save_copy_of_files(checkpoint_callback)

    lrmon = LearningRateMonitor("epoch")
    prog = TQDMProgressBar(refresh_rate=500)

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, lrmon, prog]
    )

    L.seed_everything(args.seed)
    trainer.fit(lit, train_loader, val_loader)

    # load best and test
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt is not None and best_ckpt != "":
        best = LitModel.load_from_checkpoint(best_ckpt, args=args, shapelets_np=centers)
    else:
        best = lit

    test_res = trainer.test(best, dataloaders=test_loader, verbose=False)
    val_res = trainer.test(best, dataloaders=val_loader, verbose=False)

    acc_result = {"test": test_res[0].get("test_acc", None) if test_res else None,
                  "val": val_res[0].get("test_acc", None) if val_res else None}
    f1_result = {"test": test_res[0].get("test_f1", None) if test_res else None,
                 "val": val_res[0].get("test_f1", None) if val_res else None}

    # produce classification report / artifacts
    get_clf_report(best, test_loader, CHECKPOINT_PATH, args.class_names)

    # log results to text
    text_save_dir = os.path.join(CHECKPOINT_PATH, "textFiles")
    os.makedirs(text_save_dir, exist_ok=True)
    with open(os.path.join(text_save_dir, f"{args.model_id}.txt"), "a") as f:
        f.write(f"Run: {args.name}\n")
        f.write(f"Acc: {acc_result}\nF1: {f1_result}\n\n")

    return best, acc_result, f1_result


# -------------------------------
# argument parsing & main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='RAW_SHAPLET')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--name', type=str, default='ICB_RawShapeletGate')

    # training
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # model
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=60)

    # shapelet init / control
    parser.add_argument('--use_shapelet', type=str2bool, default=True)
    parser.add_argument('--shapelet_len', type=int, default=30)
    parser.add_argument('--K_anomaly', type=int, default=10)
    parser.add_argument('--shapelet_init', type=str, default='kmeans_anomaly_only',
                        choices=['random', 'kmeans_anomaly_only', 'kmeans_both'])
    parser.add_argument('--finetune_shapelets', type=str2bool, default=True)
    parser.add_argument('--shapelet_init_max_batches', type=int, default=200)
    parser.add_argument('--shapelet_init_windows_per_series', type=int, default=4)
    parser.add_argument('--shapelet_init_samples_per_class', type=int, default=5000)

    # logging / checkpoint
    parser.add_argument('--checkpoint_root', type=str, default='/tf_logs/shapelet_raw_gate')

    # misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    args = parser.parse_args()

    # quick dataset pass to set meta needed by model
    train_loader, val_loader, test_loader = get_datasets(args.data_path, args)
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.class_names = [str(i) for i in range(args.num_classes)]

    # set batch_size if loader expects parser's batch_size (some get_datasets rely on args)
    # but we already passed args into get_datasets; ensure consistency
    print("Dataset loaded. seq_len:", args.seq_len, "channels:", args.num_channels, "num_classes:", args.num_classes)

    # train & eval
    best_model, acc_results, f1_results = train_and_eval(args)
    print("ACC results", acc_results)
    print("F1 results", f1_results)
