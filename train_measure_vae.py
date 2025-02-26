import click
import torch
import os
import shutil
import numpy as np
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.image import NonUniformImage

from MeasureVAE.measure_vae import MeasureVAE
from MeasureVAE.vae_trainer import VAETrainer
from MeasureVAE.vae_tester import VAETester
from data.dataloaders.bar_dataset import *
from utils.helpers import *

# ✅ 确保 PyTorch 多进程正确初始化
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    torch.serialization.add_safe_globals([TensorDataset])

print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("当前设备:", torch.cuda.current_device())
print("设备名称:", torch.cuda.get_device_name(0))

@click.command()
@click.option('--batch_size', default=64, help='Training batch size')
@click.option('--num_epochs', default=50, help='Number of training epochs')
@click.option('--train/--test', default=False, help='Train or test the model')
def main(batch_size, num_epochs, train):

    # ✅ 1️⃣ 确保 `datasets/` 目录清理
    dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    if os.path.exists(dataset_dir):
        print("⚠️  发现旧数据集，删除 `datasets/` 目录...")
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # ✅ 2️⃣ 重新加载数据集
    print("📥 重新加载 `FolkNBarDataset` 数据集...")
    is_short = True  # ✅ **确保数据集小一些**
    #is_short = False
    num_bars = 1

    folk_dataset_train = FolkNBarDataset(
        dataset_type='train',
        is_short=is_short,
        num_bars=num_bars
    )
    folk_dataset_test = FolkNBarDataset(
        dataset_type='test',
        is_short=is_short,
        num_bars=num_bars
    )

    # ✅ 3️⃣ 创建数据加载器
    train_dataloader, val_dataloader, test_dataloader = folk_dataset_train.data_loaders(
        batch_size=batch_size, split=(0.7, 0.2)
    )

    # ✅ **打印数据集大小，确认是否正确**
    print(f"✅ 训练数据集总大小: {len(train_dataloader.dataset)} ")
    print(f"✅ 训练批次数: {len(train_dataloader)} (batch_size={batch_size})")

    # ✅ 4️⃣ 初始化 VAE 模型
    model = MeasureVAE(
        dataset=folk_dataset_train,
        note_embedding_dim=10,
        metadata_embedding_dim=2,
        num_encoder_layers=2,
        encoder_hidden_size=512,
        encoder_dropout_prob=0.5,
        latent_space_dim=256,
        num_decoder_layers=2,
        decoder_hidden_size=512,
        decoder_dropout_prob=0.5,
        has_metadata=False
    )

    # ✅ 5️⃣ 将模型移动到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ 使用设备: {device}")

    # ✅ 6️⃣ 训练或测试模型
    if train:
        print("🚀 训练模式启动...")

        trainer = VAETrainer(
            dataset=folk_dataset_train,
            model=model,
            lr=1e-4,
            has_reg_loss=True,
            reg_type='rhy_complexity',
            reg_dim=0
        )

        # ✅ 这里修复 `train_model()` 的参数，删除 `data_loader`
        trainer.train_model(
            batch_size=batch_size,
            num_epochs=num_epochs,
            plot=True,
            log=True,
        )

# **************************************************  评估  **************************************************************************************
    else:
        print("🧪 测试模式启动...")
        model.load()
        model.to(device)
        model.eval()

        tester = VAETester(
            dataset=folk_dataset_test,
            model=model,
            has_reg_loss=True,
            reg_type='rhy_complexity',
            reg_dim=0
        )

        print("🎯 评估模型...")
        tester.test_model(batch_size=batch_size)

if __name__ == '__main__':
    main()
