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
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from dataloader import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        # 以下为新加内容
        self.gate = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.tensor(1e-2))  # 残差缩放


    # def forward(self, x):
    #     x = x.transpose(1, 2)
    #     x1 = self.conv1(x)
    #     x1_1 = self.act(x1)
    #     x1_2 = self.drop(x1_1)

    #     x2 = self.conv2(x)
    #     x2_1 = self.act(x2)
    #     x2_2 = self.drop(x2_1)

    #     out1 = x1 * x2_2
    #     out2 = x2 * x1_2

    #     x = self.conv3(out1 + out2)
    #     x = x.transpose(1, 2)
    #     return x
    def forward(self, x):
        x = x.transpose(1, 2)          # [B, C, N]
        x1 = self.drop(self.act(self.conv1(x)))  # [B, H, N]
        x2 = self.drop(self.act(self.conv2(x)))  # [B, H, N]

        gated = x2 * self.gate(x1) + x1 * self.gate(x2)  # 稳定的双向门控
        y = self.conv3(gated)            # [B, C, N]
        y = (self.gamma * y).transpose(1, 2)  # 残差缩放后回到 [B, N, C]
        return y

class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 3
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
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

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
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)
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

        self.pre_head_norm = nn.LayerNorm(args.emb_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=max(0.2, args.dropout_rate)),
            nn.Linear(args.emb_dim, args.num_classes)
        )

        self.att_fc1 = nn.Linear(args.emb_dim, args.emb_dim // 2)
        self.att_fc2 = nn.Linear(args.emb_dim // 2, 1)
        self.tanh = nn.Tanh()
        self.pre_head_norm = nn.LayerNorm(args.emb_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=max(0.2, args.dropout_rate)),
            nn.Linear(args.emb_dim, args.num_classes)
        )

        # # Classifier head
        # self.head = nn.Linear(args.emb_dim, args.num_classes)

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
        att = self.att_fc2(self.tanh(self.att_fc1(x))).squeeze(-1)  # [B, N]
        att = torch.softmax(att, dim=1)
        # 加权聚合
        z = torch.einsum('bn,bnd->bd', att, x)
        z = self.pre_head_norm(z)
        logits = self.head(z)
        return logits

        # x = x.mean(1)
        # x = self.pre_head_norm(x)
        # x = self.head(x)
        # return x


class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = TSLANet()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes, average="macro")
        self.precision = MulticlassPrecision(num_classes=args.num_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=args.num_classes, average="macro")

        self.register_buffer("ce_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.criterion = nn.CrossEntropyLoss(weight=self.ce_weights)
        # 用于储存测试集得预测和真实标签
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    # def configure_optimizers(self):
    #     optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    #     scheduler = ReduceLROnPlateau(
    #         optimizer, 
    #         mode='min', 
    #         factor=0.5, 
    #         patience=10, 
    #         verbose=True
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_loss"  # 监控验证集损失
    #         }
    #     }
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6  # T_0=首次周期长度(epochs)，T_mult=周期倍增
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"  # 每个 epoch 调整
            }
        }



    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]  # expected [B, C, T]
        labels = batch[1].to(torch.int64)

        preds = self.model.forward(data)
        
        loss = self.criterion(preds, labels)
        f1 = self.f1(preds, labels)
        prec = self.precision(preds, labels)
        rec = self.recall(preds, labels)

        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_precision", prec, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{mode}_recall", rec, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        if mode == "test":
            self.test_preds.append(preds.argmax(dim=-1).cpu())
            self.test_targets.append(labels.cpu())
        return loss

    # def _calculate_loss(self, batch, mode="train"):
    #     data = batch[0]
    #     labels = batch[1].to(torch.int64)

    #     preds = self.model.forward(data)
    #     loss = self.criterion(preds, labels)
    #     acc = (preds.argmax(dim=-1) == labels).float().mean()
    #     f1 = self.f1(preds, labels)

    #     # Logging for both step and epoch
    #     self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     if mode == "test":
    #         self.test_preds.append(preds.argmax(dim=-1).cpu())
    #         self.test_targets.append(labels.cpu())
            
    #     return loss

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


from lightning.pytorch.callbacks import EarlyStopping, StochasticWeightAveraging
swa = StochasticWeightAveraging(swa_lrs=1e-4)

def train_model():
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
            TQDMProgressBar(refresh_rate=500),
            swa
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

    # acc_result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"], "val": val_result[0]["test_f1"]}
    prec_result = {"test": test_result[0].get("test_precision"), "val": val_result[0].get("test_precision")}
    recall_result = {"test": test_result[0].get("test_recall"), "val": val_result[0].get("test_recall")}


    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)

    return model, f1_result, prec_result, recall_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='TEST')
    parser.add_argument('--data_path', type=str, default=r'data/hhar')
    parser.add_argument('--name', type=str, default='随机测试专用')
    
    # Training parameters:
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    
    # Model parameters:
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=7)

    # TSLANet components:
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    args = parser.parse_args()
    DATASET_PATH = args.data_path
    MAX_EPOCHS = args.num_epochs
    print(DATASET_PATH)

    # load from checkpoint
    run_description = f"实验描述{args.name}"
    print(f"========== {run_description} ===========")


    CHECKPOINT_PATH = f"/tf_logs/{args.name}"
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

    class_counts = np.bincount(train_loader.dataset.y_data)
    num_samples = len(train_loader.dataset.y_data)
    # 反频率权重: w_c = N / (2 * N_c)
    class_weights = num_samples / (2.0 * class_counts + 1e-8)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)


    model, f1_result, prec_result, recall_result = train_model()
    print("F1  results", f1_result)
    print("Precision results", prec_result)
    print("Recall results", recall_result)

    # append result to a text file...
    text_save_dir = "/tf_logs/textFiles/"
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"TSLANet_{os.path.basename(args.data_path)}_l_{args.depth}" + "  \n")
    f.write('mf1:{}, prec:{}, rec:{}'.format(f1_result, prec_result, recall_result))
    f.write('\n')
    f.write('\n')
    f.close()
