import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, random_masking_3D


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
    def __init__(self, seq_len, patch_size, in_chans, embed_dim):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, hparams):
        super().__init__()
        self.hparams = hparams
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
        return adaptive_mask.unsqueeze(-1)

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.hparams.adaptive_filter:
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2

        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        return x.to(dtype).view(B, N, C)


def local_perturbation(x, perturb_rate=0.1):
    B, C, T = x.shape
    num_perturb = int(T * perturb_rate)
    for i in range(B):
        idx = torch.randperm(T)[:num_perturb]
        noise = torch.randn(C, num_perturb).to(x.device) * 0.1
        x[i, :, idx] += noise
    return x


def sliding_window(x, crop_size=512):
    B, C, T = x.shape
    if crop_size >= T:
        return x
    start_idx = np.random.randint(0, T - crop_size)
    return x[:, :, start_idx:start_idx + crop_size]


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(z, z.T)
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * B, -1)
    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=-1)
    loss = -torch.log(positives / denominator)
    return loss.mean()


class TSLANet_layer(L.LightningModule):
    def __init__(self, hparams, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.hparams_layer = hparams
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim, hparams=self.hparams_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        if self.hparams_layer.ICB and self.hparams_layer.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.hparams_layer.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.hparams_layer.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


class TSLANet(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.patch_embed = PatchEmbed(
            seq_len=self.hparams.seq_len, patch_size=self.hparams.patch_size,
            in_chans=self.hparams.num_channels, embed_dim=self.hparams.emb_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hparams.emb_dim))
        self.pos_drop = nn.Dropout(p=self.hparams.dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, self.hparams.dropout_rate, self.hparams.depth)]
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(hparams=self.hparams, dim=self.hparams.emb_dim, drop=self.hparams.dropout_rate, drop_path=dpr[i])
            for i in range(self.hparams.depth)
        ])
        self.head = nn.Linear(self.hparams.emb_dim, self.hparams.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_feat=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        x = x.mean(1)
        return x if return_feat else self.head(x)


class model_training(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = TSLANet(hparams=self.hparams)
        self.f1 = MulticlassF1Score(num_classes=self.hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss(
            weight=self.hparams.class_weights,
            label_smoothing=self.hparams.label_smoothing
        )
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.train_lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=0.2, patience=10, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def _calculate_loss(self, batch, mode="train"):
        data, labels = batch[0], batch[1].to(torch.int64)
        if mode == "train":
            data1 = sliding_window(data.clone(), crop_size=int(data.shape[-1] * 0.8))
            data2 = local_perturbation(data.clone(), perturb_rate=0.1)
            data1 = F.interpolate(data1, size=self.hparams.seq_len)
            data2 = F.interpolate(data2, size=self.hparams.seq_len)
            feat1 = self.model(data1, return_feat=True)
            feat2 = self.model(data2, return_feat=True)
            preds = self.model(data)
            loss_cls = self.criterion(preds, labels)
            loss_cl = nt_xent_loss(feat1, feat2)
            total_loss = loss_cls + self.hparams.loss_rate * loss_cl
        else:
            preds = self.model(data)
            total_loss = self.criterion(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)
        self.log(f"{mode}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, "test")

    def on_test_epoch_end(self):
        if self.test_preds:
            preds = torch.cat(self.test_preds).numpy()
            targets = torch.cat(self.test_targets).numpy()
            cm = confusion_matrix(targets, preds)
            print("Confusion Matrix:\n", cm)