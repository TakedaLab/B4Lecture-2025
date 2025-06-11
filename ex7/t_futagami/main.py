#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう.

特徴量: MFCCの平均（0次項含まず）
識別器: MLP
"""

from __future__ import division, print_function

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


def my_MLP(
    input_shape,
    output_dim,
    units1=256,
    units2=256,
    dropout1=0.2,
    dropout2=0.2,
    l2_rate=0.01,
):
    """
    MLPモデルの構築.

    Args:
        input_shape: 入力の形
        output_dim: 出力次元
        units1: 1層目のユニット数
        units2: 2層目のユニット数
        dropout1: 1層目のドロップアウト率
        dropout2: 2層目のドロップアウト率
        l2_rate: L2正則化の係数 (デフォルトは0.01)
    Returns:
        model: 定義済みモデル
    """
    model = Sequential()

    model.add(
        Dense(
            units1, input_dim=input_shape, kernel_regularizer=regularizers.l2(l2_rate)
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(dropout1))

    model.add(Dense(units2, kernel_regularizer=regularizers.l2(l2_rate)))
    model.add(Activation("relu"))
    model.add(Dropout(dropout2))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    return model


def extract_mfcc(data, n_mfcc=100, n_segments=4):  # n_mfcc は基本MFCC係数の数
    """
    音声データからMFCCとそのデルタ、ダブルデルタ特徴量を抽出する.

    MFCC、デルタMFCC、ダブルデルタMFCCを計算し、時間軸で平均化.
    これら全ての特徴量を連結し、1次元の特徴ベクトルとする.
    Args:
        data_sr_tuples: (音声波形データ, サンプリングレート) のタプルのリスト
        n_mfcc: 抽出する基本MFCC係数の数 (例: 13, 20, 100)
        n_segments: 音声信号を分割するセグメント数 (デフォルトは4)
    Returns:
        features_array: 抽出された特徴量のNumPy配列
    """
    features_list = []
    for audio_signal in data:
        mfcc_orig = librosa.feature.mfcc(y=audio_signal, n_mfcc=n_mfcc)
        # デルタ特徴量を計算 (入力と同じ形状でパディングされて返される)
        delta_mfcc = librosa.feature.delta(mfcc_orig, width=5)
        delta2_mfcc = librosa.feature.delta(mfcc_orig, width=5, order=2)

        # 0次MFCCは除外
        mfcc_orig = mfcc_orig[1:, :]  # 0次MFCCを除外
        delta_mfcc = delta_mfcc[1:, :]  # 0次デルタMFCCを除外
        delta2_mfcc = delta2_mfcc[1:, :]  # 0次ダブルデルタMFCCを除外

        features = []
        # 音声信号をn_segments個のセグメントに分割して平均を取る
        for feature_all in [mfcc_orig, delta_mfcc, delta2_mfcc]:
            segments_feature = np.array_split(feature_all, n_segments, axis=1)
            segment_means = [np.mean(segment, axis=1) for segment in segments_feature]
            features.append(np.concatenate(segment_means))

        # 各特徴量を連結して1次元の特徴ベクトルを作成
        mfcc_feature_vec = np.concatenate(features)
        features_list.append(mfcc_feature_vec)

    return np.array(features_list)


def feature_extraction(
    train_path_list, test_path_list, cumulative_variance_ratio=0.99, noise_scale=-1
):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す.

    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        train_path_list: 学習データのwavファイルリスト
        test_path_list: テストデータのwavファイルリスト
        cumulative_variance_ratio: 累積寄与率の閾値（デフォルトは0.9）
        noise_scale: ホワイトノイズの標準偏差を信号の標準偏差の一定割合にする
    Returns:
        features: 特徴量
    """

    def load_data(path):
        return librosa.load(path)[0]

    train_data = list(map(load_data, train_path_list))
    test_data = list(map(load_data, test_path_list))

    if noise_scale > 0:
        data_whitenoise = []
        for signal in train_data:
            signal_std = np.std(signal)
            noise_std = noise_scale * signal_std

            white_noise = np.random.normal(0, noise_std, len(signal))
            data_whitenoise.append(signal + white_noise)

        train_data.extend(data_whitenoise)

    train_features = extract_mfcc(train_data)
    test_features = extract_mfcc(test_data)

    # PCAを適用して次元削減
    pca = PCA()
    pca.fit(train_features)
    contribution_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratio_ = np.cumsum(contribution_ratios)
    n_components = (
        np.argmax(cumulative_variance_ratio_ >= cumulative_variance_ratio) + 1
    )
    pca_final = PCA(n_components=n_components)
    train_features = pca_final.fit_transform(train_features)
    test_features = pca_final.transform(test_features)

    return train_features, test_features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット.

    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """
    cm = confusion_matrix(predict, ground_truth)
    # 各真のラベルに対する予測の割合を計算 (列方向の合計が1になるように正規化)
    cm_normalized = cm.astype("float") / cm.sum(axis=0)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()

    # x軸とy軸のラベルを設定
    tick_marks = np.arange(10)  # 0から9までのラベル
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")

    # 各セルに確率を表示
    thresh = cm_normalized.max() / 2.0
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(
                j,
                i,
                format(cm_normalized[i, j], ".2f"),
                horizontalalignment="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig("result/cm.png", transparent=True)
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する.

    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """
    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def plot_history(history):
    """
    学習履歴をプロットする.

    Args:
        history: Kerasの学習履歴オブジェクト
    Returns:
        Nothing
    """
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc)
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("result/history_acc.png", transparent=True)
    plt.show()

    plt.figure()
    plt.plot(epochs, loss)
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("result/history_loss.png", transparent=True)
    plt.show()


def objective(trial, X, Y, input_shape, output_dim):
    """
    Optunaの目的関数.

    Args:
        trial: OptunaのTrialオブジェクト
        X, Y: 学習データ
        input_shape: 入力データの形状
        output_dim: 出力層の次元数
    Returns:
        検証データの精度 (最大化を目指す)
    """
    # ハイパーパラメータの提案
    units1 = trial.suggest_int("units1", 64, 512, step=64)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5, step=0.1)
    units2 = trial.suggest_int("units2", 64, 512, step=64)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
    l2_rate = trial.suggest_float("l2_rate", 1e-5, 1e-1, log=True)

    # EarlyStoppingコールバック
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    kf = KFold(n_splits=5, shuffle=True)  # K=5
    val_accuracies = []

    for train_index, val_index in kf.split(X, Y):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]

        # 各フォールドでモデルを構築・コンパイル
        model = my_MLP(
            input_shape=input_shape,
            output_dim=output_dim,
            units1=units1,
            units2=units2,
            dropout1=dropout1,
            dropout2=dropout2,
            l2_rate=l2_rate,
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            X_train_fold,
            Y_train_fold,
            epochs=50,  # チューニング時の最大エポック数
            validation_data=(X_val_fold, Y_val_fold),
            callbacks=[early_stopping_callback],
            verbose=0,  # Optunaの試行中はログを抑制
        )

        # 検証データの精度を取得
        val_accuracy = history.history["val_accuracy"][-1]
        val_accuracies.append(val_accuracy)

    # K個のフォールドの平均精度を返す
    return np.mean(val_accuracies)


def main():
    """
    メイン関数.

    1. 引数のパース
    2. データの読み込み
    3. 特徴量の抽出
    4. Optunaによるハイパーパラメータのチューニング
    5. 最適なハイパーパラメータでモデルを再学習
    6. 学習履歴のプロット
    7. テストデータに対する予測と結果の保存
    8. 正解ファイルが指定されていれば評価を行う
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()
    path_to_e7 = "../"

    # データの読み込み
    training = pd.read_csv("../training.csv")
    test = pd.read_csv("../test.csv")

    # 学習データの特徴抽出
    X, X_test = feature_extraction(
        path_to_e7 + training["path"].values,
        path_to_e7 + test["path"].values,
        noise_scale=0.2,
    )

    # 特徴量の次元数を表示
    print(f"Number of features: {X.shape[1]}")

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y = to_categorical(y=training["label"], num_classes=10)
    Y = np.tile(Y, (2, 1))  # ホワイトノイズを追加したのでラベルも2倍にする

    print("\nStarting hyperparameter tuning with Optuna...")
    # OptunaのStudyオブジェクトを作成
    # direction="maximize" で目的関数の戻り値を最大化する
    study = optuna.create_study(direction="maximize")

    # 最適化の実行
    # lambdaを使って固定引数を渡す
    study.optimize(
        lambda trial: objective(
            trial,
            X,
            Y,
            input_shape=X.shape[1],
            output_dim=10,
        ),
        n_trials=10,  # 試行するハイパーパラメータの組み合わせの最大数
        # n_jobs=-1 # 並列実行する場合 (環境によっては設定)
    )

    # 最適なハイパーパラメータの取得
    best_hps = study.best_params

    print(
        f"""
    Hyperparameter search complete.
    Optimal units for layer 1: {best_hps.get("units1")}
    Optimal dropout for layer 1: {best_hps.get("dropout1")}
    Optimal units for layer 2: {best_hps.get("units2")}
    Optimal dropout for layer 2: {best_hps.get("dropout2")}
    Optimal learning rate: {best_hps.get("learning_rate")}
    Optimal L2 regularization rate: {best_hps.get("l2_rate")}
    """
    )

    # 全学習データでモデルを再学習
    print("\nTraining on full training data...")
    model = my_MLP(
        input_shape=X.shape[1],
        output_dim=10,
        units1=best_hps.get("units1"),
        units2=best_hps.get("units2"),
        dropout1=best_hps.get("dropout1"),
        dropout2=best_hps.get("dropout2"),
        l2_rate=best_hps.get("l2_rate"),
    )
    # モデル構成の表示
    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=best_hps.get("learning_rate")),
        metrics=["accuracy"],
    )
    history = model.fit(X, Y, batch_size=32, epochs=100, verbose=1)

    # モデル構成，学習した重みの保存
    model.save("my_model.h5")

    # 最終モデルの学習履歴をプロット
    plot_history(history)

    # 予測結果
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        test_accuracy = accuracy_score(truth_values, predicted_values)
        plot_confusion_matrix(
            predicted_values,
            truth_values,
            title=f"Acc. {round(test_accuracy * 100, 4)}%",
        )
        print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
