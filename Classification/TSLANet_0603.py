import argparse
import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torchmetrics import ConfusionMatrix

from timm.loss import LabelSmoothingCrossEntropy
from timm.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D

from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryAccuracy, BinaryF1Score


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


def Sliding_window(x, window_size, step_size):
    """
    实现滑动窗口操作。
    :param x: 输入张量，形状为 (batch_size, channels, seq_len)
    :param window_size: 窗口大小
    :param step_size: 滑动步长
    :return: 滑动窗口后的张量，形状为 (batch_size, num_windows, window_size)
    """
    batch_size, channels, seq_len = x.shape
    # 计算可以生成的完整窗口数量
    num_windows = (seq_len - window_size) // step_size + 1

    # 初始化存储窗口的列表
    windows = []

    # 提取每个窗口
    for i in range(0, num_windows * step_size, step_size):
        windows.append(x[:, :, i:i + window_size].unsqueeze(2))  # 提取窗口并添加维度

    # 合并窗口，形状为 (batch_size, channels, num_windows, window_size)
    return torch.cat(windows, dim=2)


class Statis_layer(L.LightningModule):
    def __init__(self):
        super().__init__()
        # (1)定义使用变量
        # (num, 1, seq_len) -> (num, num_patches, patch_size)
        stride = args.patch_size // 2
        num_patches = (args.seq_len - args.patch_size) // stride + 1

        self.num_patches = num_patches

        # (2)定义需要用到的模型
        self.conv1 = nn.Conv1d(
            in_channels=(num_patches-1),      # 输入通道数
            out_channels=num_patches,       # 输出通道数
            kernel_size=1,      # 卷积核大小
            stride=1,           # 步长
            bias=True           # 是否使用偏置项
        )

        self.conv2 = nn.Sequential(
            # 第一层：扩大通道数，捕捉局部模式
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出长度: (371 - 2)//2 + 1 = 186

            # 第二层：加深网络，提取抽象特征
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # 输出长度: 93

            # 第三层：进一步压缩时序维度
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化，输出 [300, 128, 1]
            nn.Flatten()              # 最终特征: [300, 128]
        )

    def forward(self, x):


        # self.num_patches的选择:[7,30,90,180,360],不同的对应不同的长度
        window_size = args.patch_size
        step_size = args.patch_size // 2
        # num_epochs = (args.seq_len - window_size) // step_size + 1  (343)
        x_patch = Sliding_window(x, window_size, step_size)# (num, 1, num_patches, window_size)

        #(1):(num, 1, 1024) -> (num, num_patch, self.patch_size),按照周期将数据重排列
        # x_patch = x.reshape(x.shape[0], -1, self.patch_size)
        #(2):计算均值(num, 1, num_patches, window_size) -> (num, 1, num_patches)
        x_patch_mean = torch.mean(x_patch, dim=-1)

        #(3):计算标准差(num, 1, num_patches, window_size) -> (num, 1, num_patches)
        # x_patch_std = torch.std(x_patch, dim=-1, keepdim=True)
        x_patch_std = torch.std(x_patch, dim=-1)
        # (num, 2, num_patches)
        statistics = torch.cat([x_patch_mean, x_patch_std], dim=1)

        #  (num, 1, num_patches, window_size) -> (num, 1, num_patches-1, window_size)
        diff_x = torch.diff(x_patch, n=1, axis=2)
        # (num, 1, num_patches-1, window_size) -> (num, 1, num_patches-1)
        diff_x_patch_max = torch.max(diff_x, dim=-1).values
        diff_x_patch_max = diff_x_patch_max.transpose(1, 2)
        # diff_x_2_transposed = diff_x_2.permute(0, 2, 1)  # 调整维度
        diff_x_2 = self.conv1(diff_x_patch_max)# (num, num_patches, 1)
        diff_x_2 = diff_x_2.transpose(1, 2)# (num, 1, num_patches)
        # (num, 3, num_patches)
        x3 = torch.cat([statistics, diff_x_2], dim=1)  # 拼接
        # x_3_transposed = x_3.permute(0, 2, 1)  # 调整维度
        # x_4 = self.conv2(x_3)
        # x_4 = x_4.transpose(1, 2) 
        # x_4 = self.linear(x_4) 
        x4 = self.conv2(x3)
        # x4 = self.conv3(x3)
        return x4



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

        # 统计特征提取输出维度 (num, 3, num_patches)
        self.statis_layer = Statis_layer()

        # Classifier head规定输入维度和输出维度,因为损失函数的缘故,此处用args.num_classes-1
        if args.criterion == 'cross_entropy':
            self.head = nn.Linear(args.emb_dim, args.num_classes)
            self.head2 = nn.Linear(128, args.num_classes)
        elif args.criterion == 'bce':
            self.head = nn.Linear(args.emb_dim, args.num_classes-1)
            self.head2 = nn.Linear(128, args.num_classes-1)
        self.head3 = nn.Linear(1034, args.num_classes)


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

    def forward(self, x):
        x_statis = self.statis_layer(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x) + self.head2(x_statis)
        return x


class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

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
        self.confmat = ConfusionMatrix(task="binary", num_classes=args.num_classes).to(self.device)
        
        # self.fc = nn.Linear(2, 1)  # 输出形状(N,1)
        if args.criterion == 'cross_entropy':
            self.criterion = LabelSmoothingCrossEntropy()
            self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        elif args.criterion == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
            self.acc = BinaryAccuracy()
            self.f1 = BinaryF1Score()
            self.precision = BinaryPrecision()
            self.recall = BinaryRecall()
            self.auroc = BinaryAUROC()

        # 用于储存测试集得预测和真实标签
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        if args.criterion == 'cross_entropy':
            loss = self.criterion(preds, labels)
            acc = (preds.argmax(dim=-1) == labels).float().mean()
            f1 = self.f1(preds, labels)
        elif args.criterion == 'bce':
            # realy_labels = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).float()
            # change_preds = self.fc(preds)
            # 计算部分：
            loss = self.criterion(preds.squeeze(1), labels.float())  # 形状(N,) vs (N,)
            pred_labels = (torch.sigmoid(preds) > 0.3).long().squeeze(1)
            acc = self.acc(pred_labels, labels)
            f1 = self.f1(pred_labels, labels)
            precision = self.precision(pred_labels, labels)
            recall = self.recall(pred_labels, labels)
            auroc = self.auroc(torch.sigmoid(preds), labels)


        # Logging for both step and epoch
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_precision", precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_recall", recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
       
       
        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())
            
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        # 在所有测试批次完成后，生成混淆矩阵
        if len(self.test_preds) > 0:
            device = next(self.model.parameters()).device
            targets = torch.cat(self.test_targets).to(device)
            preds = torch.cat(self.test_preds).to(device)
            
            # 生成混淆矩阵
            cm = self.confmat(targets, preds)
            print(f"\nConfusion Matrix (device={device}):\n", cm.cpu().numpy())


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
    precision_result = {"test": test_result[0]["test_precision"],"val": val_result[0]["test_precision"]}
    recall_result = {"test": test_result[0]["test_recall"],"val": val_result[0]["test_recall"]}
    auroc_result = {"test": test_result[0]["test_auroc"],"val": val_result[0]["test_auroc"]}


    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, acc_result, f1_result, precision_result, recall_result, auroc_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/dataset_rate0.2')
    parser.add_argument('--name', type=str, default='T_0602文件随机测试专用')
    
    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--criterion', type=str, default='bce', choices=['cross_entropy', 'bce'], help='Loss function to use')


    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.3)# 0.15
    parser.add_argument('--patch_size', type=int, default=7)

    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint
    run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}___"
    run_description += f"ASB_{args.ASB}__AF_{args.adaptive_filter}__ICB_{args.ICB}__preTr_{args.load_from_pretrained}_"
    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    print(f"========== {run_description} ===========")
    run_description = f"0603{args.name}"

    CHECKPOINT_PATH = f"/TSLANet/Classification/store_result/{run_description}"
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

    model, acc_result, f1_result, precision_result, recall_result, auroc_result = train_model(best_model_path)
    print("ACC results", acc_result)
    print("F1  results", f1_result)
    print("Precision results", precision_result)
    print("Recall results", recall_result)
    print("AUROC results", auroc_result)

    # append result to a text file...
    text_save_dir = "/TSLANet/Classification/textFiles"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    f.write('acc:{}, mf1:{}, pre:{}, reca:{}, auroc:{}'.format(acc_result, f1_result, precision_result, recall_result, auroc_result))
    # 保存混淆矩阵
    if hasattr(model, 'test_preds') and hasattr(model, 'test_targets'):
        test_preds = torch.cat(model.test_preds)
        test_targets = torch.cat(model.test_targets)
        cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write('\n')

    f.write('\n')
    f.close()
