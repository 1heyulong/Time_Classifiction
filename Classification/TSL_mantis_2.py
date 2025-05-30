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
import torchmetrics

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import BinaryF1Score
from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

        # Classifier head
        # self.head = nn.Linear(args.emb_dim, args.num_classes)
        self.head = nn.Linear(args.emb_dim, args.num_classes-1)

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
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--start', type=str2bool, default=False)
    parser.add_argument('--name', type=str, default='描述模型使用')
    
    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)

    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

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
    run_description = f"模型描述:{args.name}"

    CHECKPOINT_PATH = f"/TSLANet/Classification/store_result/{run_description}"

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
    best_model_path = f"/TSLANet/Classification/storept/best_model_{args.name}.pth"
   
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
        # train_f1 = 0.0
        # train_acc = 0.0
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
            # train_f1 += train_f1_metric(preds, y.unsqueeze(1)).item()
            # train_acc += train_acc_metric(preds, y.unsqueeze(1)).item()

        avg_train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        train_acc = accuracy_score(train_labels, train_preds)  # 使用 sklearn 计算 Accuracy
        # avg_train_f1 = train_f1 / len(train_loader)
        # avg_train_acc = train_acc / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录训练指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Train", avg_train_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Train", train_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Train", train_acc, epoch)
        tensorboard_logger.experiment.add_scalar("LearningRate", current_lr, epoch)  # 记录学习率

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Accuracy: {train_acc:.4f}")


        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_acc = 0.0

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
        avg_val_f1 = f1_score(val_labels, val_preds, average='weighted')  # 使用 sklearn 计算 F1
        avg_val_acc = accuracy_score(val_labels, val_preds)  # 使用 sklearn 计算 Accuracy
        # avg_val_f1 = val_f1 / len(val_loader)
        # avg_val_acc = val_acc / len(val_loader)


        # 记录验证指标到 TensorBoard
        tensorboard_logger.experiment.add_scalar("Loss/Validation", avg_val_loss, epoch)
        tensorboard_logger.experiment.add_scalar("F1/Validation", avg_val_f1, epoch)
        tensorboard_logger.experiment.add_scalar("Accuracy/Validation", avg_val_acc, epoch)

        print(f"Validation Loss: {avg_val_loss:.4f}, Validation F1: {avg_val_f1:.4f}, Validation Accuracy: {avg_val_acc:.4f}")

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

    with open(results_file, "a") as f:
        f.write(f"每次实验之前, 需要自己想想怎么描述实验结果:{args.name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Precision: {test_precision:.4f}\n")
        f.write(f"Test Recall: {test_recall:.4f}\n")
        f.write("\n")
        f.write("\n")

    print(f"测试结果已保存到 {results_file}")

