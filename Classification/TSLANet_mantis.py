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

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D, Inception_Block_V1



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
        # 分割切块
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches

        # 位置编码\正则化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)


        # 构建多层的TSLANet函数
        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]  # stochastic depth decay rule
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i])
            for i in range(args.depth)]
        )

        # 统计特征提取输出维度 (num, 3, num_patches)
        self.statis_layer = Statis_layer()

        # Classifier head规定输入维度和输出维度,因为损失函数的缘故,此处用args.num_classes-1
        self.head = nn.Linear(args.emb_dim, args.num_classes-1)
        self.head2 = nn.Linear(128, args.num_classes-1)
        self.head3 = nn.Linear(1034, args.num_classes-1)
        # 新的模型组件, 但是他好像也是需要原始函数的输入
        self.Times_Block = nn.ModuleList([
            TimesBlock()
            for _ in range(args.depth)]
        )

        self.VAEmodel = VAEmodel()

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
        """
        # 统计特征的组件
        x_statis = self.statis_layer(x)
        x = self.head2(x_statis)
        """

        """
        # TSLANet的组件
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        x = x.mean(1)
        x = self.head(x)
        """
        """
        # x = self.patch_embed(x)
        # x = x + self.pos_embed
        # x = self.pos_drop(x)
        # print(x.shape)
        for Times_Block in self.Times_Block:
            x = Times_Block(x)
        # print(x.shape)
        x = x.mean(1)
        # print(x.shape,'llllll')
        x = x.unsqueeze(1)
        # x = self.head3(x)
        """
        
        """        
        x = self.patch_embed(x)
        # print('qqqqqqqqqqqq',x.shape)# torch.Size([64, 343, 128])
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # print('qqqqqqqqqqqq',x.shape)# torch.Size([64, 343, 128])
        x = self.VAEmodel(x)
        """
        x_statis = self.statis_layer(x)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        x = x.mean(1)
        x = self.head(x) + self.head2(x_statis)

        return x
    

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self):
        super(TimesBlock, self).__init__()
        self.seq_len = args.seq_len
        # self.pred_len = args.pred_len
        self.k = args.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(args.d_model, args.d_ff,
                               num_kernels=args.num_kernels),
            nn.GELU(),
            Inception_Block_V1(args.d_ff, args.d_model,
                               num_kernels=args.num_kernels)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if T % period != 0:
                pad_len = period - (T % period)
                padding = torch.zeros([B, pad_len, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
                length = T + pad_len
            else:
                length = T
                out = x
            # reshape
            # print(i,'\n',out.shape,'\n',period,'\n',length,'\n',length / period)
            # reshape: [B, T, N] → [B, T//p, p, N] → [B, N, T//p, p]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # print(out.shape)
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        res = res.permute(0, 2, 1).contiguous()

        return res

# 多层次特征融合的,
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 分支1: 小核捕捉局部细节
        self.branch1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        # 分支2: 中核捕捉中等范围模式
        self.branch2 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        # 分支3: 大核捕捉全局特征
        self.branch3 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        
    def forward(self, x):
        # 各分支输出形状: [B, 16, 371] （假设padding保持长度）
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        # 沿通道维度拼接 -> [B, 16*3=48, 371]
        return torch.cat([out1, out2, out3], dim=1)


class VAEmodel(nn.Module):
    def __init__(self):
        super(VAEmodel, self).__init__()
        # self.config = config
        self.l_win = args.patch_size
        self.n_channel = args.num_channels
        # self.code_size = args.emb_dim
        self.code_size = 1
        self.num_hidden_units = 512

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.num_hidden_units // 16, (3, self.n_channel), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 8, (3, self.n_channel), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 4, (3, self.n_channel), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(),
            nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units, (4, self.n_channel), stride=(1, 1))
        )
        # Define mean and std layers for latent space
        self.fc_mean = nn.Linear(self.num_hidden_units, self.code_size)
        self.fc_std = nn.Linear(self.num_hidden_units, self.code_size)

        # Decoder
        self.decoder_fc = nn.Linear(self.code_size, self.num_hidden_units)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.num_hidden_units, self.num_hidden_units // 4, (4, self.n_channel)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 4, self.num_hidden_units // 8, (3, self.n_channel), stride=(2, 1), padding=(1, 0)),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.num_hidden_units // 8, self.n_channel, (3, self.n_channel), stride=(2, 1), padding=(1, 0)),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder
        x = x.unsqueeze(1)  # Add channel dimension
        encoded = self.encoder(x)
        # print(encoded.shape,'ttttttttt')
        encoded = encoded.view(encoded.size(0), -1)  # Flatten
        # print(encoded.shape,'ttttttttt')
        # mu = self.fc_mean(encoded)
        # logvar = self.fc_std(encoded)
        mu = torch.mean(encoded, dim=-1).unsqueeze(1)
        logvar = torch.std(encoded, dim=-1).unsqueeze(1)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        # print(z.shape,'zzzzzzzzzzzz')
        # Decoder
        z = self.decoder_fc(z)
        z = z.view(z.size(0), self.num_hidden_units, 1, 1)  # Reshape for ConvTranspose2d
        reconstructed = self.decoder(z).view(x.size(0), -1).mean(dim=-1, keepdim=True)

        return reconstructed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/test_0429/dataset5/')
    parser.add_argument('--start', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='未定义名称')

    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--para', type=str2bool, default=True)
    
    # TSLANet components:
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)



    # TimesNet parameters:
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=1)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--pred_len', type=int, default=0)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint
    run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}___"
    run_description += f"ASB_{args.ASB}__AF_{args.adaptive_filter}__ICB_{args.ICB}__preTr_{args.load_from_pretrained}_"
    run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    print(f"========== {run_description} ===========")
    run_description = f"0515_1"

    CHECKPOINT_PATH = f"/TSLANet/Classification/lightning_logs/{run_description}"



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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)#定义优化器
    if args.start:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    # criterion = LabelSmoothingCrossEntropy()
    criterion = nn.BCEWithLogitsLoss()#定义损失函数

    # 训练、验证和测试逻辑
    best_val_loss = float('inf')
    best_model_path = f"best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
   
    # 初始化 TensorBoardLogger
    tensorboard_logger = TensorBoardLogger(
        save_dir=CHECKPOINT_PATH,  # 日志保存路径
        name=f"{args.name}"    # 日志文件夹名称
    )

    # 初始化 F1 计算器
    train_f1_metric = BinaryF1Score().to('cuda')
    val_f1_metric = BinaryF1Score().to('cuda')

    train_acc_metric = torchmetrics.Accuracy(task="binary").to('cuda')
    val_acc_metric = torchmetrics.Accuracy(task="binary").to('cuda')

    for epoch in range(args.num_epochs):

        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        train_acc = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to('cuda'), y.to('cuda')
            optimizer.zero_grad()
            output = model(x)
            # print(output.shape,'ooooo')
            loss = criterion(output, y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 计算 F1 系数
            preds = (output > 0.5).float()
            train_f1 += train_f1_metric(preds, y.unsqueeze(1)).item()
            train_acc += train_acc_metric(preds, y.unsqueeze(1)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_f1 = train_f1 / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录训练指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Train", avg_train_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Train", avg_train_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        tensorboard_logger.experiment.add_scalar("LearningRate", current_lr, epoch)  # 记录学习率

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}, Train Accuracy: {avg_train_acc:.4f}")


        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to('cuda'), y.to('cuda')
                output = model(x)
                loss = criterion(output, y.unsqueeze(1).float())
                val_loss += loss.item()

                # 计算 F1 系数
                preds = (output > 0.5).float()
                val_f1 += val_f1_metric(preds, y.unsqueeze(1)).item()
                val_acc += val_acc_metric(preds, y.unsqueeze(1)).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)


        # 记录验证指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Validation", avg_val_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Validation", avg_val_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}, Validation F1: {avg_val_f1:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

        # # print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}")
        # print(f"Epoch {epoch}: Validation Loss: {avg_val_loss:.4f}, Validation F1: {avg_val_f1:.4f}")

        # 动态调整学习率
        scheduler.step(avg_val_loss)

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

    # 计算测试集指标
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    test_precision = precision_score(test_labels, test_preds, average='weighted')
    test_recall = recall_score(test_labels, test_preds, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # 保存测试结果到文本文件
    results_dir = "/TSLANet/Classification/result_log"
    os.makedirs(results_dir, exist_ok=True)  # 创建目录（如果不存在）
    results_file = os.path.join(results_dir, "result_log.txt")

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(f"ICB:{args.ICB},ASB:{args.ASB},Ir优化器:{args.start},实验描述:{args.name}每次实验之前, 需要自己想想怎么描述实验结果:\n")
        f.write(f"Test Accuracy: {test_acc:.4f} Test F1 Score: {test_f1:.4f} Test Precision: {test_precision:.4f} Test Recall: {test_recall:.4f}\n")
        f.write("\n")
        f.write("\n")

    print(f"测试结果已保存到 {results_file}")



    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_id', type=str, default='TEST')
    # parser.add_argument('--data_path', type=str, default=r'data/hhar')

    # # Training parameters:
    # parser.add_argument('--num_epochs', type=int, default=100)
    # parser.add_argument('--pretrain_epochs', type=int, default=50)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--train_lr', type=float, default=1e-3)
    # parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # # Model parameters:
    # parser.add_argument('--emb_dim', type=int, default=128)
    # parser.add_argument('--depth', type=int, default=2)
    # parser.add_argument('--masking_ratio', type=float, default=0.4)
    # parser.add_argument('--dropout_rate', type=float, default=0.15)
    # parser.add_argument('--patch_size', type=int, default=8)

    # # TSLANet components:
    # parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    # parser.add_argument('--ICB', type=str2bool, default=True)
    # parser.add_argument('--ASB', type=str2bool, default=True)
    # parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    # args = parser.parse_args()
    # DATASET_PATH = args.data_path
    # MAX_EPOCHS = args.num_epochs
    # print(DATASET_PATH)

    # # load from checkpoint
    # run_description = f"{os.path.basename(args.data_path)}_dim{args.emb_dim}_depth{args.depth}___"
    # run_description += f"ASB_{args.ASB}__AF_{args.adaptive_filter}__ICB_{args.ICB}__preTr_{args.load_from_pretrained}_"
    # run_description += f"{datetime.datetime.now().strftime('%H_%M_%S')}"
    # print(f"========== {run_description} ===========")
    # run_description = f"0512_2"

    # CHECKPOINT_PATH = f"/TSLANet/Classification/lightning_logs/{run_description}"
    # pretrain_checkpoint_callback = ModelCheckpoint(
    #     dirpath=CHECKPOINT_PATH,
    #     save_top_k=1,
    #     filename='pretrain-{epoch}',
    #     monitor='val_loss',
    #     mode='min'
    # )

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=CHECKPOINT_PATH,
    #     save_top_k=1,
    #     monitor='val_loss',
    #     mode='min'
    # )

    # # Save a copy of this file and configs file as a backup
    # save_copy_of_files(pretrain_checkpoint_callback)

    # # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # # load datasets ...
    # train_loader, val_loader, test_loader = get_datasets(DATASET_PATH, args)
    # print("Dataset loaded ...")

    # # Get dataset characteristics ...
    # args.num_classes = len(np.unique(train_loader.dataset.y_data))
    # args.class_names = [str(i) for i in range(args.num_classes)]
    # args.seq_len = train_loader.dataset.x_data.shape[-1]
    # args.num_channels = train_loader.dataset.x_data.shape[1]

    # if args.load_from_pretrained:
    #     best_model_path = pretrain_model()
    # else:
    #     best_model_path = ''

    # model, acc_results, f1_results = train_model(best_model_path)
    # print("ACC results", acc_results)
    # print("F1  results", f1_results)

    # # append result to a text file...
    # text_save_dir = "textFiles"
    # os.makedirs(text_save_dir, exist_ok=True)
    # f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    # f.write(run_description + "  \n")
    # f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    # f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    # f.write('\n')
    # f.write('\n')
    # f.close()
