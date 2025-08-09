from TSLANetshiyan import get_datasets
import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def extract_global_features(x):
    """提取每个序列的全局统计特征（用于KDE）"""
    x = x.squeeze(1).cpu().numpy()  # (B, T)
    features = []
    for seq in x:
        features.append([
            np.mean(seq), np.std(seq),
            np.max(seq), np.min(seq),
            np.mean(np.diff(seq)),  # 趋势
            np.std(np.diff(seq))    # 波动
        ])
    return np.array(features)  # (B, 6)


def extract_local_features(x, window_size=128, step=64):
    """提取每个序列的局部滑窗特征（用于热力图）"""
    B, C, T = x.shape
    x = x.squeeze(1).cpu().numpy()  # [B, T]
    num_windows = (T - window_size) // step + 1
    local_feats = []

    for i in range(B):
        seq = x[i]
        feats = []
        for start in range(0, T - window_size + 1, step):
            win = seq[start:start + window_size]
            feats.append([
                np.mean(win), np.std(win),
                np.max(win), np.min(win),
                np.mean(np.diff(win)), np.std(np.diff(win))
            ])
        local_feats.append(feats)  # (num_windows, 6)
    return np.array(local_feats)  # (B, num_windows, 6)


def plot_kde_comparison(train_feat, test_feat, feature_names, save_dir):
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(6, 4))
        sns.kdeplot(train_feat[:, i], label='Train', fill=True)
        sns.kdeplot(test_feat[:, i], label='Test', fill=True)
        plt.title(f"Global Feature Distribution - {name}")
        plt.xlabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'global_kde_{name}.png'))
        plt.close()


def plot_heatmap_diff(train_local_feat, test_local_feat, feature_names, save_dir):
    """
    输入：[B, W, 6]，输出每个特征在每个窗口上的平均值对比热力图
    """
    mean_train = np.mean(train_local_feat, axis=0)  # [W, 6]
    mean_test = np.mean(test_local_feat, axis=0)    # [W, 6]
    diff = mean_test - mean_train  # [W, 6]

    for i, name in enumerate(feature_names):
        plt.figure(figsize=(6, 3))
        sns.heatmap(diff[:, i][None, :], cmap="coolwarm", annot=False,
                    xticklabels=False, yticklabels=[name], cbar=True)
        plt.title(f"Local Feature Shift - {name} (Test - Train)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'local_diff_heatmap_{name}.png'))
        plt.close()


def collect_label_1_samples(dataloader, max_samples=256):
    data = []
    for batch in dataloader:
        x, y = batch
        label_1_idx = (y == 1)
        if label_1_idx.any():
            x_label1 = x[label_1_idx]
            data.append(x_label1)
            if len(torch.cat(data)) >= max_samples:
                break
    return torch.cat(data)[:max_samples]


def analyze_data_distribution(train_loader, test_loader, save_dir="./dist_analysis", max_samples=256):
    """
    分析训练集和测试集的分布差异（全局KDE + 局部热力图）
    """
    os.makedirs(save_dir, exist_ok=True)
    feature_names = ['mean', 'std', 'max', 'min', 'diff_mean', 'diff_std']
    # ✅ 只提取标签为 1 的样本
    train_data = collect_label_1_samples(train_loader, max_samples)
    test_data = collect_label_1_samples(test_loader, max_samples)


    # Step 2: 提取全局特征
    train_global_feat = extract_global_features(train_data)
    test_global_feat = extract_global_features(test_data)
    plot_kde_comparison(train_global_feat, test_global_feat, feature_names, save_dir)

    # Step 3: 提取局部特征
    train_local_feat = extract_local_features(train_data)
    test_local_feat = extract_local_features(test_data)
    plot_heatmap_diff(train_local_feat, test_local_feat, feature_names, save_dir)

    print(f"[√] 分布对比图已保存到: {save_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=90)
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/0716_realdata/')
    args = parser.parse_args()

    train_loader, test_loader, _ = get_datasets(args.data_path, args)
    analyze_data_distribution(train_loader, test_loader, save_dir="/dist_analysis/2")
    print("数据分布分析完成！")