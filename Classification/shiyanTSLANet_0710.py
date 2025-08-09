import argparse
import datetime
import os

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

from TSLANetshiyan import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D
from transformers import get_cosine_schedule_with_warmup



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



class ResidualBlock(L.LightningModule):
    """TCN残差块与您原有代码兼容"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              dilation=dilation, padding=(kernel_size-1)*dilation//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(out + residual)
    
class LSTM_TCN(nn.Module):
    """LSTM-TCN特征提取模块替换原有的TSLANet_layer"""
    def __init__(self, input_dim=1, lstm_units=64, tcn_filters=[32, 32, 16], kernel_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True, bidirectional=False)
        self.tcn_blocks = nn.ModuleList([
            ResidualBlock(lstm_units + input_dim, tcn_filters[0], kernel_size, dilation=1),
            ResidualBlock(tcn_filters[0], tcn_filters[1], kernel_size, dilation=2),
            ResidualBlock(tcn_filters[1], tcn_filters[2], kernel_size, dilation=4)
        ])
        self.dense = nn.Linear(tcn_filters[2], 128)

    def forward(self, x):
        # 输入x形状: [batch, seq_len, 1]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_units]
        lstm_out = lstm_out.transpose(1, 2)  # [batch, lstm_units, seq_len]
        
        # 拼接原始输入和LSTM输出
        x_reshaped = x.transpose(1, 2)  # [batch, 1, seq_len]
        concat = torch.cat([x_reshaped, lstm_out], dim=1)  # [batch, lstm_units+1, seq_len]
        
        # TCN处理
        tcn_out = concat
        for block in self.tcn_blocks:
            tcn_out = block(tcn_out)
        tcn_out = F.relu(tcn_out)
        
        # 全局平均池化 + 全连接
        pooled = torch.mean(tcn_out, dim=2)  # [batch, tcn_filters[-1]]
        return self.dense(pooled)  # [batch, 128]

class DCNN(nn.Module):
    """适配长序列的DCNN模块（输入形状 [batch, 1, num_weeks=147, 7]）"""
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # [batch, 32, 147, 7]
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                         # [batch, 32, 73, 7]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [batch, 64, 73, 7]
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),                         # [batch, 64, 36, 7]
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [batch, 128, 36, 7]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                  # [batch, 128, 1, 1]
        )
        self.dense = nn.Linear(128, 128)

    def forward(self, x):
        out = self.conv_blocks(x).squeeze(-1).squeeze(-1)  # [batch, 128]
        return self.dense(out)

class DualTimeFusionModel(nn.Module):
    """主模型替换原有的TSLANet"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.lstm_tcn = LSTM_TCN()
        self.dcnn = DCNN()
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),  # LSTM-TCN和DCNN各输出128维
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        self.week_pad = nn.ConstantPad1d((0, 7 - 1034 % 7), 0)  # 填充不足一周的部分
    def forward(self, x):
        # 输入x形状: [batch, channels, seq_len]
        # 假设channels=1（单变量时序），seq_len=28（4周数据）
        
        # 一维日尺度处理
        x_1d = x.transpose(1, 2)  # [batch, seq_len, 1]
        tcn_feat = self.lstm_tcn(x_1d)  # [batch, 128]
        
        # 二维周尺度处理
        x_padded = self.week_pad(x) 
        x_2d = x_padded.unfold(2, 7, 7)

        dcnn_feat = self.dcnn(x_2d)  # [batch, 128]
        
        # 特征融合与分类
        fused = torch.cat([tcn_feat, dcnn_feat], dim=1)  # [batch, 256]
        return self.classifier(fused)  # [batch, num_classes]


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = DualTimeFusionModel(num_classes=args.num_classes)
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()
        # 用于储存测试集得预测和真实标签
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

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
        loss = self._calculate_loss(batch, mode="train")
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
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/0627dataset_1/')
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


    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=90)
    
    # TSLANet components:
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


    CHECKPOINT_PATH = f"/TSLANet/Classification/store_result/{args.name}"
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


    best_model_path = ''

    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    # append result to a text file...
    text_save_dir = f"/TSLANet/Classification/0702textFiles_{args.name}"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n')
    f.write('\n')
    f.close()
