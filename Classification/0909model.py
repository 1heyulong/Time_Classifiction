# 0905_TSLANet_MIL.py
"""
TSLANet + MIL (instance scorer + top-k pooling) + Focal Loss + optional pseudo-anomaly augmentation

Usage:
python 0905_TSLANet_MIL.py \
  --data_path /hy-tmp/0716_realdata/ \
  --name MIL_topk3_gamma2 \
  --patch_size 60 --emb_dim 128 --depth 2 \
  --num_epochs 300 --batch_size 32 \
  --train_lr 1e-3 --weight_decay 5e-4 \
  --mil_topk 3 --focal_gamma 2.0 --pseudo_anomaly True
"""
import argparse, os, random
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool


from sklearn.metrics import f1_score, accuracy_score

def find_best_threshold(probs, labels):
    """扫描阈值找最大 F1"""
    best_thr, best_f1 = 0.5, -1
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1

def evaluate_with_threshold(model, loader, threshold):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(model.device), y.to(model.device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= threshold).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    return acc, f1




# -------------------
# ICB block
# -------------------
class ICBBlock(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,D,N]
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)
        return out.transpose(1, 2)

# -------------------
# Patch embedding
# -------------------
class PatchEmbed(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim):
        super().__init__()
        stride = patch_size // 2
        self.num_patches = int((seq_len - patch_size) / stride + 1)
        self.proj = nn.Conv1d(num_channels, emb_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B,N,D]
        return x

# -------------------
# TSLANet backbone (no mean pooling)
# -------------------
class TSLANetBackbone(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim, depth, dropout_rate):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([
            ICBBlock(emb_dim, emb_dim * 3, drop=dropout_rate) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head_in_dim = emb_dim
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)  # [B,N,D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.norm(x)  # [B,N,D] (patch-level features)

# -------------------
# Instance scorer + MIL pooling
# -------------------
class InstanceMILHead(nn.Module):
    def __init__(self, emb_dim, topk=1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, 1)
        )
        self.topk = topk

    def forward(self, x_patches):  # [B,N,D]
        logits = self.scorer(x_patches).squeeze(-1)  # [B,N]
        if self.topk == 1:
            bag_logits, _ = logits.max(dim=1)  # [B]
        else:
            topk_vals, _ = logits.topk(self.topk, dim=1)
            bag_logits = topk_vals.mean(dim=1)  # [B]
        return bag_logits, logits

# -------------------
# Focal Loss
# -------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss
                      if self.alpha is not None else (1 - pt) ** self.gamma * bce_loss)
        if self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# -------------------
# Optional pseudo anomaly augment
# -------------------
def inject_pseudo_anomaly(x, prob=0.15, scale=0.3):
    # x: [B,C,T]
    B,C,T = x.shape
    for i in range(B):
        if random.random() < prob:
            s = random.randint(0, T//2)
            e = min(T, s + random.randint(10,50))
            x[i,:,s:e] = x[i,:,s:e] * scale
    return x

# -------------------
# Lightning wrapper
# -------------------
class LitMIL(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TSLANetBackbone(args.seq_len, args.patch_size,
                                        args.num_channels, args.emb_dim,
                                        args.depth, args.dropout_rate)
        self.mil_head = InstanceMILHead(args.emb_dim, topk=args.mil_topk)
        self.f1 = MulticlassF1Score(num_classes=2)
        self.criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        self.test_preds, self.test_targets = [], []
        self.args = args

    def forward(self, x):
        x_patches = self.backbone(x)
        bag_logits, _ = self.mil_head(x_patches)
        return bag_logits

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.args.train_lr,
                                weight_decay=self.args.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                         patience=50, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

    def _shared_step(self, batch, mode):
        x,y = batch
        if self.args.pseudo_anomaly and mode=="train":
            x = inject_pseudo_anomaly(x.clone())
        bag_logits = self(x)
        loss = self.criterion(bag_logits, y)
        preds = (torch.sigmoid(bag_logits) > 0.5).long()
        acc = (preds==y).float().mean()
        f1 = self.f1(preds, y)
        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        self.log(f"{mode}_f1", f1, prog_bar=True)
        if mode=="test":
            self.test_preds.append(preds.cpu())
            self.test_targets.append(y.cpu())
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch,"train")
    def validation_step(self, batch, batch_idx): self._shared_step(batch,"val")
    def test_step(self, batch, batch_idx): self._shared_step(batch,"test")

    def on_test_epoch_end(self):
        if self.test_preds:
            y_pred = torch.cat(self.test_preds).numpy()
            y_true = torch.cat(self.test_targets).numpy()
            print("Confusion Matrix:\n", confusion_matrix(y_true,y_pred))

# -------------------
# main training flow
# -------------------
def train_and_eval(args):
    train_loader,val_loader,test_loader = get_datasets(args.data_path,args)
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.class_names = [str(i) for i in range(args.num_classes)]
    lit = LitMIL(args)

    ckpt_dir = os.path.join(args.checkpoint_root,args.name)
    os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,save_top_k=1,monitor="val_loss",mode="min")
    save_copy_of_files(ckpt_cb)

    trainer = L.Trainer(default_root_dir=ckpt_dir, accelerator="auto", devices=1,
                        max_epochs=args.num_epochs,
                        callbacks=[ckpt_cb,LearningRateMonitor("epoch"),TQDMProgressBar(refresh_rate=500)])
    L.seed_everything(args.seed)
    trainer.fit(lit,train_loader,val_loader)
    best = LitMIL.load_from_checkpoint(ckpt_cb.best_model_path,args=args)

    best.eval()
    val_probs, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(best.device)
            logits = best(x)
            val_probs.append(torch.sigmoid(logits).cpu().numpy())
            val_labels.append(y.numpy())
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)

    best_thr, best_val_f1 = find_best_threshold(val_probs, val_labels)
    print(f"[Val] Best threshold {best_thr:.2f}, F1={best_val_f1:.4f}")

    # ---- 用最佳阈值在 val/test 上评估 ----
    val_acc, val_f1 = evaluate_with_threshold(best, val_loader, best_thr)
    test_acc, test_f1 = evaluate_with_threshold(best, test_loader, best_thr)

    print("Final Results:")
    print({"val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1, "thr": best_thr})

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data_path",type=str,default="data/hhar")
    p.add_argument("--name",type=str,default="MIL_exp")
    p.add_argument("--num_epochs",type=int,default=300)
    p.add_argument("--batch_size",type=int,default=32)
    p.add_argument("--train_lr",type=float,default=1e-3)
    p.add_argument("--weight_decay",type=float,default=5e-4)
    p.add_argument("--emb_dim",type=int,default=128)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--dropout_rate",type=float,default=0.15)
    p.add_argument("--patch_size",type=int,default=60)
    p.add_argument("--mil_topk",type=int,default=3)
    p.add_argument("--focal_gamma",type=float,default=2.0)
    p.add_argument("--focal_alpha",type=float,default=None)
    p.add_argument("--pseudo_anomaly",type=str2bool,default=False)
    p.add_argument("--checkpoint_root",type=str,default="/tf_logs/mil_runs")
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    train_and_eval(args)
