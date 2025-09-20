import argparse
import random
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# change these imports to wherever your functions live
from TSLANetshiyan import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sliding windows helper -> returns [B, C, num_windows, window_size]
def sliding_windows(x, window_size, step):
    B, C, T = x.shape
    num_windows = (T - window_size) // step + 1
    windows = []
    for s in range(0, num_windows * step, step):
        windows.append(x[:, :, s:s + window_size].unsqueeze(2))
    return torch.cat(windows, dim=2)  # [B, C, num_windows, window_size]


# ---------------------------
# Blocks: ICB, PatchEmbed, TSLA backbone (patch-level output)
# ---------------------------
class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, D] -> conv1d expects [B, D, N]
        x = x.transpose(1, 2)
        x1 = self.drop(self.act(self.conv1(x)))
        x2 = self.drop(self.act(self.conv2(x)))
        out = self.conv3(x1 * x2 + x2 * x1)
        return out.transpose(1, 2)  # [B, N, D]


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=60, in_chans=1, embed_dim=128):
        super().__init__()
        stride = max(1, patch_size // 2)
        self.num_patches = (seq_len - patch_size) // stride + 1
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        # x: [B, C, T] -> [B, N, D]
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.asb = None  # not using ASB in current simplified block; keep optional
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        return x + self.drop_path(self.icb(self.norm2(x)))


class TSLANetBackbone(nn.Module):
    def __init__(self, seq_len, patch_size, num_channels, emb_dim, depth, dropout_rate):
        super().__init__()
        self.patch_embed = PatchEmbed(seq_len, patch_size, num_channels, emb_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.pos_drop = nn.Dropout(dropout_rate)
        dpr = [x.item() for x in torch.linspace(0.1, dropout_rate, depth)]
        self.blocks = nn.ModuleList([TSLANet_layer(dim=emb_dim, drop=dropout_rate, drop_path=dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(emb_dim)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_in_dim = emb_dim

    def forward(self, x_raw):
        # returns patch-level features and pooled global feature
        # x_raw: [B, C, T]
        x = self.patch_embed(x_raw)  # [B, N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        global_feat = x.mean(1)  # [B, D]
        return x, global_feat  # patch-level [B,N,D], global [B,D]


# ---------------------------
# Statistical branch (per-patch stats -> conv -> vector)
# ---------------------------
class StatisBranch(nn.Module):
    def __init__(self, seq_len, patch_size, out_dim=128):
        super().__init__()
        stride = max(1, patch_size // 2)
        self.num_patches = (seq_len - patch_size) // stride + 1
        # convnet to summarize patch-level statistics
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.ReLU()
        )
        self.out_dim = out_dim

    def forward(self, x_raw, patch_size):
        # x_raw: [B, C, T]
        step = max(1, patch_size // 2)
        x_patch = sliding_windows(x_raw, patch_size, step)  # [B, C, num_patches, window]
        # per-patch stats across window dim: mean/std/max_abs_diff, then average across channels
        patch_mean = x_patch.mean(dim=-1)  # [B, C, num_patches]
        patch_std = x_patch.std(dim=-1)    # [B, C, num_patches]
        diff = x_patch[:, :, :, 1:] - x_patch[:, :, :, :-1]
        diff_max = diff.abs().max(dim=-1)[0]  # [B, C, num_patches-1]
        # pad diff_max to have num_patches dimension (duplicate last)
        if diff_max.shape[-1] < patch_mean.shape[-1]:
            diff_max = F.pad(diff_max, (0, 1), mode='replicate')
        # aggregate across channels to get 3 channels
        mean_agg = patch_mean.mean(dim=1, keepdim=True)  # [B,1,num_patches]
        std_agg = patch_std.mean(dim=1, keepdim=True)
        diff_agg = diff_max.mean(dim=1, keepdim=True)
        stats = torch.cat([mean_agg, std_agg, diff_agg], dim=1)  # [B,3,num_patches]
        out = self.net(stats)  # [B, out_dim]
        return out


# ---------------------------
# Fusion modules
# ---------------------------
class FusionModule(nn.Module):
    def __init__(self, emb_dim, stat_dim, fusion_type="gate_nn"):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == "learned_scalar":
            # learnable scalar alpha in (0,1)
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.classifier = nn.Linear(emb_dim, 1)  # use fused dim = emb_dim (we assume same dim)
            # if stat_dim != emb_dim, project stat to emb_dim externally
        elif fusion_type == "gate_nn":
            # gate network uses concatenated [global, stat] -> scalar gate in (0,1)
            self.gate = nn.Sequential(
                nn.Linear(emb_dim + stat_dim, (emb_dim + stat_dim) // 2),
                nn.GELU(),
                nn.Linear((emb_dim + stat_dim) // 2, 1)
            )
            self.proj_stat = nn.Linear(stat_dim, emb_dim) if stat_dim != emb_dim else nn.Identity()
            self.classifier = nn.Linear(emb_dim, 1)
        elif fusion_type == "mlp_concat":
            # concat and MLP
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim + stat_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU()
            )
            self.classifier = nn.Linear(emb_dim, 1)
        else:
            raise ValueError("Unknown fusion_type")

    def forward(self, global_feat, stat_feat):
        # global_feat: [B, Dg], stat_feat: [B, Ds]
        if self.fusion_type == "learned_scalar":
            # require same dims: if stat dim differs, caller must project
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * global_feat + (1 - alpha) * stat_feat
            logits = self.classifier(fused)
            return logits.squeeze(-1), alpha.detach().cpu().item()
        elif self.fusion_type == "gate_nn":
            stat_proj = self.proj_stat(stat_feat)
            gate_logit = self.gate(torch.cat([global_feat, stat_feat], dim=-1))
            gate = torch.sigmoid(gate_logit)  # [B,1]
            fused = gate * global_feat + (1 - gate) * stat_proj
            logits = self.classifier(fused)
            return logits.squeeze(-1), gate.squeeze(-1)
        else:  # mlp_concat
            fused = self.mlp(torch.cat([global_feat, stat_feat], dim=-1))
            logits = self.classifier(fused)
            return logits.squeeze(-1), None


# ---------------------------
# Pseudo anomaly augment (optional)
# ---------------------------
def inject_pseudo_anomaly(x, prob=0.12, scale=0.25, min_len=5, max_len=60):
    # x: [B, C, T]
    B, C, T = x.shape
    for i in range(B):
        if random.random() < prob:
            s = random.randint(0, max(0, T - min_len))
            l = random.randint(min_len, min(max_len, T - s))
            x[i, :, s:s + l] = x[i, :, s:s + l] * scale
    return x


# ---------------------------
# Full Model wrapper
# ---------------------------
class ModelWithStatFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = TSLANetBackbone(seq_len=args.seq_len, patch_size=args.patch_size,
                                        num_channels=args.num_channels, emb_dim=args.emb_dim,
                                        depth=args.depth, dropout_rate=args.dropout_rate)
        self.statis = StatisBranch(seq_len=args.seq_len, patch_size=args.patch_size, out_dim=args.stat_dim)
        # if stat_dim != emb_dim and fusion_type == learned_scalar we project stat to emb_dim
        self.stat_proj_for_scalar = nn.Linear(args.stat_dim, args.emb_dim) if (args.fusion_type == "learned_scalar" and args.stat_dim != args.emb_dim) else nn.Identity()
        self.fusion = FusionModule(emb_dim=args.emb_dim, stat_dim=args.stat_dim, fusion_type=args.fusion_type)

    def forward(self, x_raw):
        # x_raw: [B, C, T]
        patch_feats, global_feat = self.backbone(x_raw)
        stat_feat = self.statis(x_raw, patch_size=args.patch_size)
        # project if needed
        if isinstance(self.stat_proj_for_scalar, nn.Linear):
            stat_proj = self.stat_proj_for_scalar(stat_feat)
        else:
            stat_proj = stat_feat
        logits, gate = self.fusion(global_feat, stat_proj)
        return logits, gate, global_feat, stat_feat


# ---------------------------
# Training / evaluation utilities
# ---------------------------
def find_best_threshold(probs, labels):
    best_thr, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        f = f1_score(labels, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, t
    return best_thr, best_f1

def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    probs_list = []
    labels_list = []
    gates = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, gate, _, _ = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(y.numpy())
            if gate is not None:
                gates.append(gate.detach().cpu().numpy())
    probs = np.concatenate(probs_list)
    labels = np.concatenate(labels_list)
    preds = (probs >= threshold).astype(int)
    return {
        "acc": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "cm": confusion_matrix(labels, preds),
        "probs": probs,
        "labels": labels,
        "gates": np.concatenate(gates) if len(gates) > 0 else None
    }


# ---------------------------
# Main train loop
# ---------------------------
def train_and_eval(args):
    # load data (unchanged)
    train_loader, val_loader, test_loader = get_datasets(args.data_path, args)

    # metadata
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]
    args.class_names = [str(i) for i in range(args.num_classes)]

    # build model
    model = ModelWithStatFusion(args).to(device)

    # criterion: BCE with optional pos_weight to handle imbalance
    # pos_ratio = np.mean(train_loader.dataset.y_data == 1)
    pos_ratio = np.mean(train_loader.dataset.y_data.cpu().numpy() == 1)
    pos_weight = torch.tensor((1.0 - pos_ratio) / (pos_ratio + 1e-8)).to(device) if args.use_pos_weight else None
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True) if args.use_plateau else optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_T, eta_min=1e-6)

    # checkpoint dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_desc = f"{args.name}_{timestamp}"
    CHECKPOINT_PATH = args.checkpoint_root
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    # save_copy_of_files(CHECKPOINT_PATH)  # your utils expects path or callback; adjust if necessary

    best_val_f1 = -1.0
    best_model_path = os.path.join(CHECKPOINT_PATH, "best_model.pt")

    # tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=os.path.join(CHECKPOINT_PATH, "tb"))
    except Exception:
        tb = None

    set_seed(args.seed)
    for epoch in range(args.num_epochs):
        model.train()
        losses = []
        all_probs = []
        all_labels = []
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device).float()
            optimizer.zero_grad()
            if args.pseudo_anomaly and random.random() < args.pseudo_prob:
                x = inject_pseudo_anomaly(x, prob=args.pseudo_prob, scale=args.pseudo_scale)
            logits, gate, _, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(y.detach().cpu().numpy().tolist())
        avg_train_loss = np.mean(losses)
        train_preds = (np.array(all_probs) >= 0.5).astype(int)
        train_f1 = f1_score(all_labels, train_preds, zero_division=0)
        train_acc = accuracy_score(all_labels, train_preds)

        # validation step (collect probs for thresholding)
        model.eval()
        val_losses = []
        val_probs_list = []
        val_labels_list = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y_cpu = y.cpu().numpy()
                logits, gate, _, _ = model(x)
                loss = criterion(logits, torch.tensor(y_cpu, dtype=torch.float32, device=device))
                val_losses.append(loss.item())
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.append(y_cpu)
        avg_val_loss = np.mean(val_losses)
        val_probs = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)

        # scheduler step
        if args.use_plateau:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # choose threshold on val
        best_thr_epoch, best_f1_epoch = find_best_threshold(val_probs, val_labels)
        val_preds = (val_probs >= best_thr_epoch).astype(int)
        val_f1_epoch = f1_score(val_labels, val_preds, zero_division=0)
        val_acc_epoch = accuracy_score(val_labels, val_preds)

        # logging
        print(f"[{epoch}/{args.num_epochs}] train_loss={avg_train_loss:.4f} train_f1={train_f1:.4f} val_loss={avg_val_loss:.4f} val_f1={val_f1_epoch:.4f} thr={best_thr_epoch:.3f}")
        if tb:
            tb.add_scalar("Loss/train", avg_train_loss, epoch)
            tb.add_scalar("Loss/val", avg_val_loss, epoch)
            tb.add_scalar("F1/train", train_f1, epoch)
            tb.add_scalar("F1/val", val_f1_epoch, epoch)
            tb.add_scalar("Acc/val", val_acc_epoch, epoch)

        # save by val_f1
        if val_f1_epoch > best_val_f1:
            best_val_f1 = val_f1_epoch
            torch.save({"model_state": model.state_dict(), "args": vars(args)}, best_model_path)
            print(f"Saved best model (val_f1={best_val_f1:.4f}) -> {best_model_path}")

    # after training: load best and perform final evaluation with threshold search
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # val threshold search (again)
    all_val_probs = []
    all_val_labels = []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, gate, _, _ = model(x)
            all_val_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_val_labels.append(y.cpu().numpy())
    all_val_probs = np.concatenate(all_val_probs)
    all_val_labels = np.concatenate(all_val_labels)
    best_thr, best_val_f1 = find_best_threshold(all_val_probs, all_val_labels)
    print(f"Best threshold found on val: {best_thr:.3f} (val_f1={best_val_f1:.4f})")

    # test evaluation
    test_res = evaluate_model(model, test_loader, device, threshold=best_thr)
    print("Test results:", test_res["acc"], test_res["f1"], test_res["prec"], test_res["rec"])
    print("Confusion matrix:\n", test_res["cm"])

    # save classification report/artifacts if you have get_clf_report
    try:
        get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)
    except Exception:
        pass

    # write results
    with open(os.path.join(CHECKPOINT_PATH, "results.txt"), "a") as f:
        f.write(f"run: {args.name}\n")
        f.write(f"best_val_f1: {best_val_f1:.4f}, best_thr: {best_thr:.3f}\n")
        f.write(f"test_acc: {test_res['acc']:.4f}, test_f1: {test_res['f1']:.4f}, prec: {test_res['prec']:.4f}, rec: {test_res['rec']:.4f}\n")
        f.write(f"cm:\n{test_res['cm']}\n\n")

    if tb:
        tb.close()

    return model, test_res


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--name', type=str, default='stat_fusion')
    parser.add_argument('--checkpoint_root', type=str, default='/tf_logs/stat_fusion')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=60)
    parser.add_argument('--stat_dim', type=int, default=128)
    parser.add_argument('--fusion_type', type=str, default='gate_nn', choices=['learned_scalar','gate_nn','mlp_concat'])
    parser.add_argument('--module', type=str, default='TSLA_Statis')  # kept for compatibility
    parser.add_argument('--pseudo_anomaly', type=str2bool, default=False)
    parser.add_argument('--pseudo_prob', type=float, default=0.12)
    parser.add_argument('--pseudo_scale', type=float, default=0.25)
    parser.add_argument('--use_plateau', type=str2bool, default=True)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--cosine_T', type=int, default=50)
    parser.add_argument('--use_pos_weight', type=str2bool, default=True)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # set seed and small adjustments
    set_seed(args.seed)

    # attach dataloader batch_size into their get_datasets if needed (some implementations use args.batch_size)
    # We'll assume get_datasets reads args.batch_size
    import types
    # ensure args visible in module-level functions requiring it
      # used by some submodules if needed
    args = args

    model, test_res = train_and_eval(args)
