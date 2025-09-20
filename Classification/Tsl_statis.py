import argparse
import datetime
import os

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger


import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torch.nn.functional as F

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score

from TSLANetshiyan import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D, Inception_Block_V1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix





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


# 滑动时间序列分割工具
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


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=7, in_chans=1, embed_dim=128):
        super().__init__()
        stride = patch_size // 2  # 3
        num_patches = (seq_len - patch_size) // stride + 1
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


class TSLANet_layer(nn.Module):
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


class Statis_layer(nn.Module):
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


class TSLANet(nn.Module):
    def __init__(self):
        super().__init__()
        # 分割切块
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        # 位置编码\正则化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout)


        # 构建多层的TSLANet函数
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # 统计特征提取输出维度 (num, 3, num_patches)
        self.statis_layer = Statis_layer()

        # Classifier head规定输入维度和输出维度,因为损失函数的缘故,此处用args.num_classes-1
        self.head = nn.Linear(args.emb_dim, args.num_classes-1)
        self.head2 = nn.Linear(128, args.num_classes-1)
        self.head3 = nn.Linear(1034, args.num_classes-1)



        # 初始化权重, 为了快速训练使用
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


    def forward(self, x):

        if args.module == 'Statis':
            x_statis = self.statis_layer(x)
            # x_statis.shape (num, 128, )
            x = self.head2(x_statis)

        elif args.module == 'TSLA':
            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for tsla_blk in self.tsla_blocks:
                x = tsla_blk(x)
            x = x.mean(1)
            x = self.head(x)

        elif args.module == 'TSLA_Statis':
            x_statis = self.statis_layer(x)

            x = self.patch_embed(x)
            x = x + self.pos_embed
            x = self.pos_drop(x)

            for tsla_blk in self.tsla_blocks:
                x = tsla_blk(x)
            x = x.mean(1)
            x = self.head(x) + self.head2(x_statis)

        else:
            raise ValueError("Invalid module type. Choose from 'Statis', 'TSLA', 'TSLA_Statis', 'other'.")

        return x
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/dataset_rate0.2/')
    parser.add_argument('--start', type=str2bool, default=False, help='用于控制学习率调整函数使用')
    parser.add_argument('--name', type=str, default='描述模型')
    parser.add_argument('--batch_size', type=int, default=64)

    # Model parameters:
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2, help="控制神经网络深度")
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=7, help="时间序列切割的窗口大小")
    parser.add_argument('--module', type=str, default='TSLA_Statis', help="选择模型模块: 'Statis', 'TSLA', 'TSLA_Statis'")

    # TSLANet_layer components:
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint
    run_description = f"model{args.module}"
    run_description += f"_time{datetime.datetime.now().strftime('%H_%M')}_{args.name}"
    print(f"========== {run_description} ===========")
    
    CHECKPOINT_PATH = f"/hy-tmp/store_result/time_0603/{run_description}"

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

    model = TSLANet().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)#定义优化器
    if args.start:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # criterion = LabelSmoothingCrossEntropy()
    criterion = nn.BCEWithLogitsLoss()#定义损失函数

    # 训练、验证和测试逻辑
    best_val_loss = float('inf')
    best_model_path = f"/hy-tmp/store_pt/{args.module}_best_model_time_{datetime.datetime.now().strftime('%H_%M')}.pt"
   
    # 初始化 TensorBoardLogger
    tensorboard_logger = TensorBoardLogger(
        save_dir=CHECKPOINT_PATH,  # 日志保存路径
        name="tensorboard_logs"    # 日志文件夹名称
    )

    # 初始化 F1 计算器
    train_f1_metric = BinaryF1Score().to('cuda')
    val_f1_metric = BinaryF1Score().to('cuda')

    train_acc_metric = torchmetrics.Accuracy(task="binary").to('cuda')
    val_acc_metric = torchmetrics.Accuracy(task="binary").to('cuda')

    for epoch in range(args.num_epochs):

        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to('cuda'), y.to('cuda')
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 计算 F1 系数
            preds = (output > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, train_preds)  # 使用 sklearn 计算 Accuracy
        current_lr = optimizer.param_groups[0]['lr']

        # 记录训练指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Train", avg_train_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Train", train_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Train", train_acc, epoch)
        tensorboard_logger.experiment.add_scalar("LearningRate", current_lr, epoch)  # 记录学习率

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {train_acc:.4f}")


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_preds, val_labels = [], []
            for x, y in val_loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = model(x)
                loss = criterion(output, y.unsqueeze(1).float())
                val_loss += loss.item()

                # 计算 F1 系数
                preds = (output > 0.5).float()

                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = f1_score(val_labels, val_preds)  # 使用 sklearn 计算 F1
        avg_val_acc = accuracy_score(val_labels, val_preds)  # 使用 sklearn 计算 Accuracy

        # 记录验证指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Validation", avg_val_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Validation", avg_val_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1: {avg_val_f1:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

        # 动态调整学习率
        scheduler.step()

        # 保存验证损失最小的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"新的模型的验证损失: {best_val_loss:.4f}")

    # 测试阶段
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0.0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to('cuda'), y.to('cuda')
            output = model(x)
            loss = criterion(output, y.unsqueeze(1).float())
            test_loss += loss.item()
            ## 如果损失函数是criterion = LabelSmoothingCrossEntropy()
            # preds = torch.argmax(output, dim=1)
            # test_preds.extend(preds.cpu().numpy())
            # test_labels.extend(y.cpu().numpy())

            ## 如果损失函数是criterion = nn.BCEWithLogitsLoss()
            preds = (output > 0.5).float()  # 将 logits 转换为二值预测
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    # 计算测试集指标 ACC, F1, Precision, Recall, cm
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds)
    test_recall = recall_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # 保存测试结果到文本文件
    results_dir = "/TSLANet/Classification/result_log"
    os.makedirs(results_dir, exist_ok=True)  # 创建目录（如果不存在）
    results_file = os.path.join(results_dir, "result_log.txt")

    with open(results_file, "a") as f:
        f.write(f"模型描述_{run_description}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}")
        f.write(f"Test F1 Score: {test_f1:.4f}")
        f.write(f"Test Precision: {test_precision:.4f}")
        f.write(f"Test Recall: {test_recall:.4f}")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write("\n")
        f.write("\n")

    print(f"测试结果已保存到 {results_file}")






