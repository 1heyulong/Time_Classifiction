import argparse
import os
import optuna
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 从您的核心脚本中导入必要的模块
from shiyanTSLANet_0712_0702_2 import model_training
from dataloader import get_datasets

def objective(trial, args_cli):
    """
    Optuna调用的目标函数, 运行一次完整的训练和评估。
    """
    # 1. 定义超参数搜索空间
    hparams = argparse.Namespace()
    hparams.train_lr = trial.suggest_float("train_lr", 1e-5, 1e-2, log=True)
    hparams.dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    hparams.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    hparams.loss_rate = trial.suggest_float("loss_rate", 0.05, 0.5)
    hparams.depth = trial.suggest_int("depth", 2, 4)
    hparams.emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 192])

    # 2. 从命令行或默认值获取固定参数
    hparams.data_path = args_cli.data_path
    hparams.batch_size = args_cli.batch_size
    hparams.num_epochs = args_cli.num_epochs
    hparams.patch_size = 90
    hparams.label_smoothing = 0.1
    hparams.ICB = True
    hparams.ASB = True
    hparams.adaptive_filter = True

    # 3. 数据加载和预处理
    L.seed_everything(42)
    train_loader, val_loader, _ = get_datasets(hparams.data_path, hparams)
    
    y_train_numpy = train_loader.dataset.y_data.cpu().numpy()

    hparams.num_classes = len(np.unique(train_loader.dataset.y_data))
    hparams.seq_len = train_loader.dataset.x_data.shape[-1]
    hparams.num_channels = train_loader.dataset.x_data.shape[1]


    y_train_integers = y_train_numpy.astype(int)
    unique_classes = np.unique(y_train_integers)
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_classes,
        y=y_train_numpy
    )
    hparams.class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 4. 模型训练
    early_stopping = EarlyStopping(monitor="val_f1", patience=15, mode="max", verbose=False)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=hparams.num_epochs,
        callbacks=[early_stopping],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    model = model_training(hparams=hparams)
    
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"Trial failed with error: {e}. Pruning trial.")
        raise optuna.exceptions.TrialPruned()

    # 5. 返回要优化的目标值 (验证集F1分数)
    val_f1 = trainer.callback_metrics.get("val_f1", 0.0).item()
    return val_f1

def train_with_best_params(best_params):
    """使用找到的最佳参数来完整地训练一次，并保存模型。"""
    print("\n--- Training final model with best parameters ---")
    
    hparams = argparse.Namespace(**best_params) # 将字典转为Namespace
    
    # 设置固定的参数
    hparams.data_path = '/hy-tmp/0712_realdata/'
    hparams.batch_size = 32
    hparams.num_epochs = 500 # 可以用更多的epoch来训练最终模型
    hparams.patch_size = 90
    hparams.label_smoothing = 0.1
    hparams.ICB = True
    hparams.ASB = True
    hparams.adaptive_filter = True
    
    # 数据加载
    L.seed_everything(42)
    train_loader, val_loader, test_loader = get_datasets(hparams.data_path, hparams)
    y_train_numpy = train_loader.dataset.y_data.cpu().numpy()
    unique_classes = np.unique(y_train_numpy)

    hparams.num_classes = len(np.unique(train_loader.dataset.y_data))
    hparams.seq_len = train_loader.dataset.x_data.shape[-1]
    hparams.num_channels = train_loader.dataset.x_data.shape[1]
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_numpy)
    hparams.class_weights = torch.tensor(class_weights, dtype=torch.float)

    # 设置回调，包括模型保存
    CHECKPOINT_PATH = "./best_model_checkpoint"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        filename='best-model-{epoch}-{val_f1:.4f}',
        save_top_k=1,
        monitor='val_f1',
        mode='max'
    )
    early_stopping = EarlyStopping(monitor="val_f1", patience=25, mode="max", verbose=True)

    # 训练最终模型
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=True, # 为最终模型开启日志
        enable_progress_bar=True,
    )

    model = model_training(hparams=hparams)
    trainer.fit(model, train_loader, val_loader)

    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print("Testing best model...")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')

if __name__ == "__main__":
    # 解析命令行参数, 可用于做对比试验
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=50, help='Number of hyperparameter tuning trials.')
    parser.add_argument('--data_path', type=str, default=r'/hy-tmp/0712_realdata/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=500, help='Epochs for each trial, can be less than final training.')
    args_cli = parser.parse_args()

    # 创建一个匿名函数来传递命令行参数
    objective_fn = lambda trial: objective(trial, args_cli)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_fn, n_trials=args_cli.n_trials)

    print("\n+++++++++++++++++++++++++++++++++++++++")
    print("Hyperparameter search finished.")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Max val_f1): {best_trial.value:.4f}")
    print("  Best Parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print("+++++++++++++++++++++++++++++++++++++++")

    # 使用找到的最佳参数进行最终的训练
    train_with_best_params(best_trial.params)