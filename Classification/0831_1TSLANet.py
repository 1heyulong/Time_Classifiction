# tslanet_shapelet_attention.py
"""
TSLANet + Shapelet-Attention Head (ShapeFormer-style)
- Keep your data loading and training pipeline unchanged.
- Replace classification head with Shapelet-Attention fusion (optional).
- Default: keep ICB True, ASB False (your best baseline).
"""

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
# Basic building blocks (kept very close to your original)
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
        x = x.transpose(1, 2)  # [B, C, N]
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)
        return out.transpose(1, 2)  # [B, N, C]


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, T]
        x_out = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]
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
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        return adaptive_mask.unsqueeze(-1)

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
            x_weighted += x_masked * weight_high
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype).view(B, N, C)
        return x


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
# Shapelet-Attention Head (ShapeFormer-style)
# -------------------------------
class ShapeletAttentionHead(nn.Module):
    """
    Shapelet-attention head that:
      - Maintains K shapelet templates of length L (in token space).
      - For each shapelet, computes similarity (via conv1d) against all subsequences of the tokens.
      - Soft-attends over positions to produce a shapelet-specific representation.
      - Aggregate across K and fuse with global representation (mean pooled).
    Configurable args:
      - K: number of shapelets
      - L: length in tokens (must be <= N)
      - agg: how to aggregate K (mean / max / learned)
      - fuse: concatenation or gated fusion
    """
    def __init__(self, embed_dim, num_shapelets=6, shapelet_len=7, agg='mean', fuse_mode='concat', gate=False):
        super().__init__()
        self.D = embed_dim
        self.K = num_shapelets
        self.L = shapelet_len
        assert agg in ('mean', 'max', 'proj')
        self.agg = agg
        assert fuse_mode in ('concat', 'gated')
        self.fuse_mode = fuse_mode
        self.gate = gate

        # shapelets: K x D x L (conv kernels)
        self.shapelets = nn.Parameter(torch.randn(self.K, self.D, self.L) * 0.02)
        trunc_normal_(self.shapelets, std=0.02)

        # if 'proj' aggregate, learn weights to combine K -> D
        if self.agg == 'proj':
            self.proj_agg = nn.Linear(self.K * self.D, self.D)

        # fusion layer after concatenation
        if self.fuse_mode == 'concat':
            self.fusion_proj = nn.Linear(self.D * 2, self.D)
        elif self.fuse_mode == 'gated' or self.gate:
            # gate: g = sigmoid(W [global; shapelet])
            self.gate_net = nn.Sequential(
                nn.Linear(self.D * 2, self.D),
                nn.GELU(),
                nn.Linear(self.D, 1),
                nn.Sigmoid()
            )

        # scaling for attention scores
        self.register_buffer('scale', torch.tensor(1.0 / (self.D * self.L) ** 0.5))

    def forward(self, tokens):
        """
        tokens: [B, N, D] token embeddings from backbone
        returns: shapelet_rep [B, D], attn_map [B, K, T']
        """
        B, N, D = tokens.shape
        assert D == self.D, f"embed dim mismatch: {D} vs {self.D}"
        if N < self.L:
            # pad tokens to at least L (rare but safe)
            pad = self.L - N
            tokens = F.pad(tokens, (0, 0, 0, pad))  # pad time dim at end
            B, N, D = tokens.shape

        # tokens -> [B, D, N] for conv1d
        x = tokens.transpose(1, 2)  # [B, D, N]

        # conv1d with K kernels shaped [K, D, L] yields [B, K, T'] where T' = N-L+1
        scores = F.conv1d(x, self.shapelets, bias=None, stride=1, padding=0)  # [B, K, T']
        # scale and softmax across time
        scores = scores * self.scale
        attn = F.softmax(scores, dim=-1)  # [B, K, T']

        # compute subsequence embeddings: unfold tokens to [B, D, T', L], mean over L -> [B, D, T']
        subseqs = x.unfold(dimension=2, size=self.L, step=1)  # [B, D, T', L]
        subseqs_mean = subseqs.mean(dim=-1)  # [B, D, T']

        # compute shapelet-specific representation: einsum attn(B,K,T') x subseqs_mean(B,D,T') -> [B, K, D]
        rep = torch.einsum('bkt, bdt -> bkd', attn, subseqs_mean)  # [B, K, D]

        # aggregate across K
        if self.agg == 'mean':
            rep_agg = rep.mean(dim=1)  # [B, D]
        elif self.agg == 'max':
            rep_agg, _ = rep.max(dim=1)
        else:  # proj
            rep_flat = rep.view(B, -1)
            rep_agg = self.proj_agg(rep_flat)

        # final normalized shapelet rep
        rep_agg = F.layer_norm(rep_agg, (self.D,))

        return rep_agg, attn  # attn useful for visualization / debugging

# -------------------------------
# TSLANet backbone with Shapelet-Attention fusion
# -------------------------------
class TSLANet(L.LightningModule):
    def __init__(self):
        super().__init__()
        # patch embedding
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

        # original classifier head replaced by a light one; after fusion we'll map to classes
        self.cls_head = nn.Linear(args.emb_dim, args.num_classes)

        # shapelet attention head (optional)
        self.use_shapelet = args.use_shapelet_attn
        if self.use_shapelet:
            # num shapelets K: absolute number or per-class? choose absolute for now
            self.shapelet_head = ShapeletAttentionHead(
                embed_dim=args.emb_dim,
                num_shapelets=args.shapelet_K,
                shapelet_len=args.shapelet_L,
                agg=args.shapelet_agg,
                fuse_mode=args.shapelet_fuse,
                gate=args.shapelet_gate
            )
            # if fuse_mode concat then cls_in_dim = 2*D else D
            cls_in_dim = args.emb_dim * (2 if args.shapelet_fuse == 'concat' else 1)
            # projection from fused features to logits
            self.fused_proj = nn.Linear(cls_in_dim, args.num_classes)

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

    def forward_tokens(self, x):
        # x: [B, C, T]
        tokens = self.patch_embed(x)  # [B, N, D]
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)
        for blk in self.tsla_blocks:
            tokens = blk(tokens)
        return tokens  # [B, N, D]

    def forward(self, x):
        tokens = self.forward_tokens(x)                # [B, N, D]
        global_feat = tokens.mean(dim=1)               # [B, D]
        logits_main = self.cls_head(global_feat)       # [B, C]

        if not self.use_shapelet:
            return logits_main

        # shapelet representation + attn map
        shapelet_feat, attn_map = self.shapelet_head(tokens)  # [B, D], [B, K, T']
        # fuse
        if args.shapelet_fuse == 'concat':
            fused = torch.cat([global_feat, shapelet_feat], dim=-1)  # [B, 2D]
            logits = self.fused_proj(fused)
        else:  # 'gated' fusion (learnable gate)
            # gate: small MLP on [global; shapelet]
            gate_input = torch.cat([global_feat, shapelet_feat], dim=-1)
            g = torch.sigmoid(nn.Linear(self.cls_head.in_features * 2, 1).to(global_feat.device)(gate_input))  # scalar per sample
            shapelet_logits = self.cls_head(shapelet_feat)
            logits = (1. - g) * logits_main + g * shapelet_logits

        # expose attn_map via attribute for debugging/visualization if needed
        self.last_shapelet_attn = attn_map.detach().cpu() if attn_map is not None else None
        return logits

# -------------------------------
# Lightning training module (keeps your loss/logging)
# -------------------------------
class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        loss = self.criterion(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # expose some shapelet diagnostics if available
        if self.model.use_shapelet and hasattr(self.model, 'last_shapelet_attn'):
            # log mean attention strengths (pos/neg) - requires labels in batch
            attn = getattr(self.model, 'last_shapelet_attn', None)
            if attn is not None:
                # attn: cpu tensor [B, K, T']
                try:
                    attn_mean = attn.mean(dim=-1).mean(dim=-1)  # [B]
                    pos_mask = labels.cpu() == 1
                    neg_mask = labels.cpu() == 0
                    if pos_mask.any():
                        self.log(f"{mode}_shapelet_attn_pos", float(attn_mean[pos_mask].mean()), on_step=False, on_epoch=True)
                    if neg_mask.any():
                        self.log(f"{mode}_shapelet_attn_neg", float(attn_mean[neg_mask].mean()), on_step=False, on_epoch=True)
                except Exception:
                    pass

        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

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

# -------------------------------
# Training / Helpers (identical pattern to your scripts)
# -------------------------------
def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None

    L.seed_everything(42)
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
    parser.add_argument('--name', type=str, default='ICB_ShapeletAttention')

    # Training args
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Model args (keep defaults aligned with your best)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=7)

    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=False)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # Shapelet-attn options
    parser.add_argument('--use_shapelet_attn', type=str2bool, default=True)
    parser.add_argument('--shapelet_K', type=int, default=6, help='total number of shapelets (templates)')
    parser.add_argument('--shapelet_L', type=int, default=7, help='shapelet length in tokens (<= num tokens)')
    parser.add_argument('--shapelet_agg', type=str, default='mean', choices=['mean','max','proj'])
    parser.add_argument('--shapelet_fuse', type=str, default='concat', choices=['concat','gated'])
    parser.add_argument('--shapelet_gate', type=str2bool, default=False)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs

    run_description = f"实验描述{args.name}"
    print(f"========== {run_description} ===========")
    print("DATASET_PATH:", DATASET_PATH)

    CHECKPOINT_PATH = f"/tf_logs/shapelet_attention/{args.name}"
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

    # Train & eval
    model, acc_results, f1_results = train_model('')
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # persist results
    text_save_dir = "/tf_logs/textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    with open(f"{text_save_dir}/{args.model_id}.txt", 'a') as f:
        f.write(run_description + "  \n")
        f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
        f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
        f.write('\n\n')
