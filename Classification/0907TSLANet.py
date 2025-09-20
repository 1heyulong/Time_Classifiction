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

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool

# =====================================================
# ICB Block
# =====================================================
class ICBBlock(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, N]
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)
        return out.transpose(1, 2)  # [B, N, D]

# =====================================================
# Patch Embedding
# =====================================================
class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
        return x

# =====================================================
# Raw Shapelet Head (直接替代 pooling)
# =====================================================
class RawShapeletHead(nn.Module):
    def __init__(self, shapelets_np, emb_dim, finetune=True):
        super().__init__()
        if isinstance(shapelets_np, np.ndarray):
            tensor = torch.from_numpy(shapelets_np.astype(np.float32))
        else:
            tensor = shapelets_np.float()
        K, C, L = tensor.shape
        self.shapelet_kernels = nn.Parameter(tensor.clone(), requires_grad=finetune)  # [K, C, L]
        self.proj = nn.Linear(K, emb_dim)
        self.register_buffer("eps", torch.tensor(1e-6))

    def forward(self, x_raw):
        sims = F.conv1d(x_raw, self.shapelet_kernels, stride=1)  # [B, K, T']
        feat = sims.max(dim=-1)[0]  # [B, K]
        feat = feat / (feat.norm(dim=-1, keepdim=True) + self.eps)
        return self.proj(feat)  # [B, D]

# =====================================================
# Backbone 
# =====================================================
class TSLANetBackbone(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim, depth, dropout_rate,
                 shapelets_np, finetune_shapelets=True):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([ICBBlock(emb_dim, emb_dim * 3, drop=dropout_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        self.shapelet_head = RawShapeletHead(shapelets_np, emb_dim, finetune=finetune_shapelets)
        self.head_in_dim = emb_dim
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_raw):
        # patch-level features
        x = self.patch_embed(x_raw)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = x + blk(x)
        x = self.norm(x)  # [B, N, D]
        return x.mean(1)  # global pooled feature


class FusionModel(nn.Module):
    def __init__(self, backbone, shapelets_np, num_classes=2, finetune_shapelets=True):
        super().__init__()
        emb_dim = backbone.head_in_dim
        self.backbone = backbone
        self.shapelet_head = RawShapeletHead(shapelets_np, emb_dim, finetune=finetune_shapelets)
        self.classifier = nn.Linear(emb_dim * 2, num_classes)

    def forward(self, x_raw):
        global_feat = self.backbone(x_raw)   # [B, D]
        local_feat = self.shapelet_head(x_raw)  # [B, D]
        fused = torch.cat([global_feat, local_feat], dim=-1)  # [B, 2D]
        logits = self.classifier(fused)
        return logits, global_feat, local_feat



# =====================================================
# Init Shapelets (KMeans on anomaly windows)
# =====================================================
def init_shapelets_from_raw(train_loader, K_anomaly=10, shapelet_len=30,
                            max_batches=200, max_windows_total=5000, seed=42):
    np.random.seed(seed)
    anomaly_windows = []
    for bi, (x_batch, y_batch) in enumerate(train_loader):
        x_np, y_np = x_batch.cpu().numpy(), y_batch.cpu().numpy()
        for i in range(len(y_np)):
            if int(y_np[i]) == 1:
                seq = x_np[i]  # [C, T]
                T = seq.shape[1]
                step = max(1, shapelet_len // 2)
                for s in range(0, T - shapelet_len + 1, step):
                    win = seq[:, s:s + shapelet_len]
                    if win.shape[1] == shapelet_len:
                        anomaly_windows.append(win.astype(np.float32))
                        if len(anomaly_windows) >= max_windows_total:
                            break
        if bi >= max_batches or len(anomaly_windows) >= max_windows_total:
            break

    if len(anomaly_windows) == 0:
        print("[ShapeletInit] WARNING: No anomaly windows, random init.")
        sample_x, _ = next(iter(train_loader))
        C = sample_x.shape[1]
        return np.random.randn(K_anomaly, C, shapelet_len).astype(np.float32) * 0.02

    windows_arr = np.stack(anomaly_windows, axis=0)
    N, C, L = windows_arr.shape
    print(f"[ShapeletInit] Collected {N} windows (C={C}, L={L})")

    X = windows_arr.reshape(N, -1)
    n_clusters = min(K_anomaly, N)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.reshape(n_clusters, C, L).astype(np.float32)
    print(f"[ShapeletInit] KMeans done. {centers.shape[0]} centers.")
    return centers

# =====================================================
# Lightning Module
# =====================================================
class LitModel(L.LightningModule):
    def __init__(self, args, shapelets_np):
        super().__init__()
        self.save_hyperparameters(ignore=['shapelets_np'])
        self.backbone = TSLANetBackbone(args.seq_len, args.patch_size, args.num_channels,
                                        args.emb_dim, args.depth, args.dropout_rate,
                                        shapelets_np, finetune_shapelets=args.finetune_shapelets)
        self.fc = nn.Linear(self.backbone.head_in_dim, args.num_classes)
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        self.test_preds, self.test_targets = [], []
        self.model = FusionModel(self.backbone, shapelets_np, args.num_classes, finetune_shapelets=args.finetune_shapelets)

    def forward(self, x):
        logits, *_ = self.model(x)
        return logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=100, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def _shared_step(self, batch, mode="train"):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1(logits, y)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        self.log(f"{mode}_f1", f1, prog_bar=True)
        if mode == "test":
            self.test_preds.append(preds.cpu())
            self.test_targets.append(y.cpu())
        return loss

    def training_step(self, b, i): return self._shared_step(b, "train")
    def validation_step(self, b, i): return self._shared_step(b, "val")
    def test_step(self, b, i): return self._shared_step(b, "test")

    def on_test_epoch_end(self):
        if self.test_preds:
            y_pred = torch.cat(self.test_preds).numpy()
            y_true = torch.cat(self.test_targets).numpy()
            print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# =====================================================
# Training Loop
# =====================================================
def train_and_eval(args):
    train_loader, val_loader, test_loader = get_datasets(args.data_path, args)
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.class_names = [str(i) for i in range(args.num_classes)]

    centers = init_shapelets_from_raw(train_loader, K_anomaly=args.K_anomaly,
                                      shapelet_len=args.shapelet_len, seed=args.seed)

    lit = LitModel(args, centers)
    CHECKPOINT_PATH = os.path.join(args.checkpoint_root, args.name)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=CHECKPOINT_PATH, save_top_k=1, monitor="val_loss", mode="min")
    save_copy_of_files(ckpt_cb)

    trainer = L.Trainer(default_root_dir=CHECKPOINT_PATH, accelerator="auto", devices=1,
                        max_epochs=args.num_epochs, callbacks=[ckpt_cb, LearningRateMonitor("epoch"),
                                                               TQDMProgressBar(refresh_rate=500)])
    L.seed_everything(args.seed)
    trainer.fit(lit, train_loader, val_loader)

    best = LitModel.load_from_checkpoint(ckpt_cb.best_model_path, args=args, shapelets_np=centers)
    test_res = trainer.test(best, dataloaders=test_loader, verbose=False)
    val_res = trainer.test(best, dataloaders=val_loader, verbose=False)

    get_clf_report(best, test_loader, CHECKPOINT_PATH, args.class_names)
    return best, {"test": test_res[0]["test_acc"], "val": val_res[0]["test_acc"]}, \
           {"test": test_res[0]["test_f1"], "val": val_res[0]["test_f1"]}

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TSLANet_ShapeletReplace')
    parser.add_argument('--data_path', type=str, default='data/hhar')
    parser.add_argument('--name', type=str, default='ICB_ShapeletReplace')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=60)
    parser.add_argument('--shapelet_len', type=int, default=30)
    parser.add_argument('--K_anomaly', type=int, default=10)
    parser.add_argument('--finetune_shapelets', type=str2bool, default=True)
    parser.add_argument('--checkpoint_root', type=str, default='/tf_logs/0906')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    args = parser.parse_args()

    model, acc_res, f1_res = train_and_eval(args)
    print("ACC:", acc_res)
    print("F1 :", f1_res)
