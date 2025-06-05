#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト(Pytorch Lightning版)
特徴量；MFCC
識別器；CNN

pytorch-lightning
    Docs: https://pytorch-lightning.readthedocs.io/
LightningModule
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html
Trainer
    Docs: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    API Refference: https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html
"""

import argparse
import os

import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchaudio
import torchmetrics
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class my_MLP(pl.LightningModule):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.model = self.create_model(input_channels, output_dim)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.confm = torchmetrics.ConfusionMatrix(10, normalize="true")

    def create_model(self, input_channels, output_dim):
        """
        MLPモデルの構築
        Args:
            input_channels: チャネル数
            output_dim: 出力次元
        Returns:
            model: 定義済みモデル
        """
        model = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),  # 64x128x128
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 64x64x64
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 64x64x64
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # 64x32x32
            torch.nn.Flatten(),  # 65536
            torch.nn.Linear(64 * 16 * 16, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, output_dim),
        )
        # モデル構成の表示
        print(model)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/acc",
            self.train_acc(pred, y),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("val/acc", self.val_acc(pred, y), prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_id=None):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_fn(pred, y)
        self.log("test/acc", self.test_acc(pred, y), prog_bar=True, logger=True)
        return {"pred": torch.argmax(pred, dim=-1), "target": y}

    def test_epoch_end(self, outputs) -> None:
        # 混同行列を tensorboard に出力
        preds = torch.cat([tmp["pred"] for tmp in outputs])
        targets = torch.cat([tmp["target"] for tmp in outputs])
        confusion_matrix = self.confm(preds, targets)
        df_cm = pd.DataFrame(
            confusion_matrix.cpu().numpy(), index=range(10), columns=range(10)
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="gray_r").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return self.optimizer


class FSDD(Dataset):
    def __init__(self, path_list, label) -> None:
        super().__init__()
        self.features = self.feature_extraction(path_list)
        self.label = label

    def feature_extraction(self, path_list):
        """
        wavファイルのリストから特徴抽出を行いリストで返す
        扱う特徴量はMFCC13次元の平均（0次は含めない）
        Args:
            root: dataset が存在するディレクトリ
            path_list: 特徴抽出するファイルのパスリスト
        Returns:
            features: 特徴量
        """
        n_mfcc = 64
        n_time = 64
        datasize = len(path_list)
        features = torch.zeros(datasize, 1, n_mfcc, n_time)
        transform = torchaudio.transforms.MFCC(
            n_mfcc=32, melkwargs={"n_mels": 64, "n_fft": 512}
        )
        for i, path in enumerate(path_list):
            # data.shape==(channel,time)
            data, _ = torchaudio.load(os.path.join(root, path))
            mfcc = transform(data[0]).unsqueeze(0).unsqueeze(0)
            mfcc = torch.nn.functional.interpolate(
                mfcc, size=(n_mfcc, n_time), mode="bilinear", align_corners=False
            )
            mfcc = mfcc.squeeze(0)

            features[i] = mfcc
        return features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        return self.features[index], self.label[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv(os.path.join(root, "training.csv"))

    # Dataset の作成
    train_dataset = FSDD(training["path"].values, training["label"].values)

    # Train/Validation 分割
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size], torch.Generator().manual_seed(20200616)
    )

    if args.path_to_truth:
        # Test Dataset の作成
        test = pd.read_csv(args.path_to_truth)
        test_dataset = FSDD(test["path"].values, test["label"].values)
    else:
        test_dataset = None

    # DataModule の作成
    datamodule = pl.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=4,
    )

    # モデルの構築
    model = my_MLP(input_channels=train_dataset[0][0].shape[0], output_dim=10)

    # 学習の設定
    trainer = pl.Trainer(max_epochs=50, gpus=0)

    # モデルの学習
    trainer.fit(model=model, datamodule=datamodule)

    # バリデーション
    trainer.validate(model=model, datamodule=datamodule)

    if args.path_to_truth:
        # テスト
        trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
