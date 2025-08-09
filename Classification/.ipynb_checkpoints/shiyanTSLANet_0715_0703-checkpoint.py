import argparse
import datetime
import os
import stumpy

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D
from transformers import get_cosine_schedule_with_warmup

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


class MultiScaleTemporalBlock(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.branch1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_features, hidden_features, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(in_features, hidden_features, kernel_size=7, padding=3)

        self.combine = nn.Conv1d(hidden_features * 3, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):  # x: [B, T, C]
        x = x.transpose(1, 2)
        out1 = self.act(self.branch1(x))
        out2 = self.act(self.branch2(x))
        out3 = self.act(self.branch3(x))
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.combine(out)
        out = out.transpose(1, 2)
        return out


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if args.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


def generate_simulated_anomaly(x_normal, min_length=7, max_length=90, anomaly_type='random'):
    """
    在正常样本上生成模拟异常
    x_normal: 正常样本 [C, T]
    anomaly_type: 'reduction'(窃电模拟), 'spike'(异常高耗), 'flat'(稳定低耗)
    """
    C, T = x_normal.shape
    x_anomaly = x_normal.clone()
    
    # 随机选择异常时间段
    length = random.randint(min_length, min(max_length, T))
    start = random.randint(0, T - length)
    end = start + length
    
    if anomaly_type == 'reduction':  # 窃电模式
        reduction_factor = random.uniform(0.2, 0.7)
        x_anomaly[:, start:end] *= reduction_factor
        # 添加平滑过渡
        transition = min(5, length//4)
        for i in range(transition):
            factor = reduction_factor + (1 - reduction_factor) * i / transition
            x_anomaly[:, start+i] *= factor
            x_anomaly[:, end-1-i] *= factor
            
    elif anomaly_type == 'spike':  # 异常高耗
        spike_factor = random.uniform(1.5, 3.0)
        x_anomaly[:, start:end] *= spike_factor
        
    elif anomaly_type == 'flat':  # 稳定低耗
        base_level = torch.mean(x_normal[:, start:end], dim=1, keepdim=True)
        flat_factor = random.uniform(0.3, 0.8)
        x_anomaly[:, start:end] = base_level * flat_factor
    
    return x_anomaly


def compute_matrix_profile_features(x, window_size=30):
    """
    计算Matrix Profile特征
    x: 输入序列 [B, C, T]
    返回: MP特征 [B, 4] (min_mp, mean_mp, std_mp, num_discords)
    """
    B, C, T = x.shape
    mp_features = torch.zeros(B, 4)
    
    for i in range(B):
        # 取主要通道计算MP
        ts = x[i, 0].cpu().numpy()
        mp = stumpy.stump(ts, m=window_size)[:, 0]
        
        # 提取特征统计量
        min_mp = np.nanmin(mp)
        mean_mp = np.nanmean(mp)
        std_mp = np.nanstd(mp)
        
        # 检测显著异常片段
        threshold = mean_mp + 2 * std_mp
        num_discords = np.sum(mp > threshold)
        
        mp_features[i] = torch.tensor([min_mp, mean_mp, std_mp, num_discords])
    
    return mp_features


def local_perturbation(x, perturb_rate=0.1):
    """
    对输入张量 x 应用局部扰动。
    perturb_rate 表示扰动比例（例如 0.1 表示扰动 10%）
    x: Tensor of shape (B, C, T)
    """
    B, C, T = x.shape
    num_perturb = int(T * perturb_rate)  # 计算需要扰动的时间步数
    for i in range(B):
        idx = torch.randperm(T)[:num_perturb]  # 随机选择需要扰动的时间步
        noise = torch.randn(C, num_perturb).to(x.device) * 0.1  # 生成扰动噪声
        x[i,:,idx] += noise
    
    return x


def sliding_window(x, crop_size=512):
    """
    对输入张量 x 应用滑动窗口操作。
    window_size: 窗口大小
    x: Tensor of shape (B, C, T)
    """
    B, C, T = x.shape
    if crop_size >= T:
        return x
    start_idx = np.random.randint(0, T - crop_size)  # 随机选择起始位置
    return x[:,:,start_idx:start_idx + crop_size]


def segment_swap(x, period=7):
    B, C, T = x.shape
    if T < 2 * period:
        return x
    
    num_segments = T // period
    segment_indices = list(range(num_segments))

    # 随机选择两个不同周期段交换
    idx1, idx2 = random.sample(segment_indices, 2)

    # 构建交换后的序列
    swapped_x = x.clone()
    seg1 = x[:, :, idx1*period:(idx1+1)*period].clone()
    seg2 = x[:, :, idx2*period:(idx2+1)*period].clone()
    
    swapped_x[:, :, idx1*period:(idx1+1)*period] = seg2
    swapped_x[:, :, idx2*period:(idx2+1)*period] = seg1

    return swapped_x


def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # Conc
    similarity_matrix = torch.matmul(z, z.T)

    # 保留正负样本的相似度
    labels = torch.cat([torch.arange(B), torch.arange(B) + B], dim=0).to(z.device)
    mask = torch.eye(2 * B, dtype=torch.bool).to(z.device)
    similarity_matrix = similarity_matrix[~mask].view(2 * B, -1)

    positives = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    denominator = torch.sum(torch.exp(similarity_matrix / temperature), dim=-1)
    loss = -torch.log(positives / denominator)
    return loss.mean()


class AbnormalAttentionPooling(L.LightningModule):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        # x: [B, N, C]
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)  # [B, N, 1]
        x_weighted = (x*attn_weights).sum(dim=1)  # [B, C]

        return x_weighted


class TopKPooling(L.LightningModule):
    def __init__(self, in_features, k_ratio=0.2):
        super().__init__()
        self.k_ratio = k_ratio
        self.score_fn = nn.Linear(in_features, 1)

    def forward(self, x):  # x: [B, N, C]
        score = self.score_fn(x).squeeze(-1)  # [B, N]
        k = int(self.k_ratio * x.size(1))
        topk_idx = score.topk(k, dim=1).indices  # [B, K]
        topk_idx = topk_idx.unsqueeze(-1).expand(-1, -1, x.size(2))  # [B, K, C]
        x_topk = torch.gather(x, 1, topk_idx)  # [B, K, C]
        return x_topk.mean(1)  # [B, C]


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if args.mstb:
            self.icb = MultiScaleTemporalBlock(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        else:
            self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)


    def forward(self, x):
        # Check if both ASB and ICB are true
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


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

        self.input_layer = nn.Linear(args.patch_size, args.emb_dim)

        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i])
            for i in range(args.depth)]
        )
        if args.topk_pool:
            self.pooling = TopKPooling(in_features=args.emb_dim, k_ratio=0.2)
        else:
            self.pooling = AbnormalAttentionPooling(in_features=args.emb_dim)
        # Classifier head
        self.head = nn.Linear(args.emb_dim, args.num_classes)

        # init weights
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
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x, return_feat=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        # x = x.mean(1)
        x = self.pooling(x)  # [B, D]
        if return_feat:
            return x
        else:
            return self.head(x)


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

        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

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
        self.criterion = LabelSmoothingCrossEntropy()
        # 用于储存测试集得预测和真实标签
        self.test_preds = []
        self.test_targets = []


        # MP特征分类头
        if args.use_mp_features:
            self.mp_head = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, args.num_classes))


    def forward(self, x, mp_features=None):
        base_output = self.model(x)

        if args.use_mp_features and mp_features is not None:
            mp_output = self.mp_head(mp_features)
            # 融合主模型和MP特征输出
            return 0.7 * base_output + 0.3 * mp_output
        return base_output

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
    #     return optimizer


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)

        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        if mode == "train":
            data1 = sliding_window(data.clone(), crop_size=int(data.shape[-1]*0.8))
            data2 = local_perturbation(data.clone(), perturb_rate=0.1)
            data3 = segment_swap(data.clone(), period=args.patch_size)
            # 转换为模型输入尺寸
            data1 = F.interpolate(data1, size=args.seq_len)
            data2 = F.interpolate(data2, size=args.seq_len)
            data3 = F.interpolate(data3, size=args.seq_len)
            # 模型前向
            feat1 = self.model.forward(data1, return_feat=True)  # B, D
            feat2 = self.model.forward(data2, return_feat=True)  # B, D
            feat3 = self.model.forward(data3, return_feat=True)  # B, D

            preds = self.model.forward(data)
            loss_cls = self.criterion(preds, labels)
            loss_cl_13 = nt_xent_loss(feat1, feat3, temperature=0.5)
            loss_cl_23 = nt_xent_loss(feat2, feat3, temperature=0.5)
            total_loss = loss_cls + args.loss_rate * (loss_cl_13 + loss_cl_23)  # 平衡系数可调


        else:
            preds = self.model(data)
            # 计算分类损失
            total_loss = self.criterion(preds, labels)

        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)

        # Logging for both step and epoch
        self.log(f"{mode}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())
            
        return total_loss

    def training_step(self, batch, batch_idx):
        data, labels = batch
        if args.use_simulated_anomaly:
            normal_mask = (labels == 0)
            normal_data = data[normal_mask]
            
            if len(normal_data) > 0:
                # 为每个正常样本生成1-3个异常变体
                num_variants = random.randint(1, 3)
                augmented_data = []
                augmented_labels = []
                
                for _ in range(num_variants):
                    anomaly_type = random.choice(['reduction', 'spike', 'flat'])
                    for sample in normal_data:
                        augmented_data.append(generate_simulated_anomaly(
                            sample, 
                            min_length=args.min_anomaly_length,
                            max_length=args.max_anomaly_length,
                            anomaly_type=anomaly_type
                        ))
                        augmented_labels.append(torch.tensor(1))  # 异常标签
                
                # 添加到训练数据
                augmented_data = torch.stack(augmented_data)
                augmented_labels = torch.stack(augmented_labels)
                data = torch.cat([data, augmented_data], dim=0)
                labels = torch.cat([labels, augmented_labels], dim=0)
        
        # 计算MP特征
        mp_features = None
        if args.use_mp_features:
            mp_features = compute_matrix_profile_features(data, args.mp_window_size)
        
        # 模型前向
        outputs = self.forward(data, mp_features)
        loss = self.criterion(outputs, labels)
        # loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        # 在所有测试批次完成后，生成混淆矩阵
        if len(self.test_preds) > 0:
            test_preds = torch.cat(self.test_preds)
            test_targets = torch.cat(self.test_targets)
            
            # 生成混淆矩阵
            cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())
            print("Confusion Matrix:\n", cm)


def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs

    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    model = model_pretraining()
    trainer.fit(model, train_loader, val_loader)

    return pretrain_checkpoint_callback.best_model_path


def train_model(pretrained_model_path):
    # # 创建 TensorBoardLogger
    # tensorboard_logger = TensorBoardLogger(
    #     save_dir=args.checkpoints,  # 日志保存路径
    #     name="tensorboard_logs"     # 日志文件夹名称
    # )
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        num_sanity_val_steps=0,
        max_epochs=MAX_EPOCHS,
        # logger=tensorboard_logger,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    L.seed_everything(42)  # To be reproducible
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()

    trainer.fit(model, train_loader, val_loader)

    # Load the best checkpoint after training
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}

    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, acc_result, f1_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/0712_realdata/')
    parser.add_argument('--name', type=str, default='随机测试专用')
    
    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--loss_rate', type=float, default=0.1)
    parser.add_argument('--topk_pool', type=str2bool, default=False)
    parser.add_argument('--mstb', type=str2bool, default=False)
    # 在参数解析部分添加
    parser.add_argument('--use_simulated_anomaly', type=str2bool, default=False, 
                        help='是否使用模拟异常数据增强')
    parser.add_argument('--use_mp_features', type=str2bool, default=False, 
                        help='是否使用Matrix Profile特征')
    parser.add_argument('--min_anomaly_length', type=int, default=7, 
                        help='模拟异常的最小持续时间')
    parser.add_argument('--max_anomaly_length', type=int, default=90, 
                        help='模拟异常的最大持续时间')
    parser.add_argument('--mp_window_size', type=int, default=30, 
                        help='Matrix Profile计算窗口大小')

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=90)
    
    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint
    run_description = f"{args.name}"
    print(f"========== {run_description} ===========")


    CHECKPOINT_PATH = f"/tf_logs/store_result/{args.name}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='val_loss',
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Save a copy of this file and configs file as a backup
    save_copy_of_files(pretrain_checkpoint_callback)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load datasets ...
    train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    print("Dataset loaded ...")

    # Get dataset characteristics ...
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file...
    text_save_dir = f"/tf_logs/Classification/0704textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n')
    f.write('\n')
    f.close()


