#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""


from __future__ import division
from __future__ import print_function

import argparse

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def my_MLP(input_shape, output_dim):
    """
    MLPモデルの構築
    Args:
        input_shape: 入力の形
        output_dim: 出力次元
    Returns:
        model: 定義済みモデル
    """

    model = Sequential()

    model.add(Dense(256, input_dim=input_shape))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim))
    model.add(Activation("softmax"))

    # モデル構成の表示
    model.summary()

    return model


def feature_extraction(path_list, noise=False):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    load_data = lambda path: librosa.load(path)[0]

    data = list(map(load_data, path_list))

    if noise:
        data_whitenoise = data
        data_whitenoise = list(
            map(lambda y: y + np.random.normal(0, 1, len(y)), data_whitenoise)
        )
        data.extend(data_whitenoise)

        """
        fig, ax = plt.subplots(nrows=2, sharex=True)
        librosa.display.waveplot(data[0], ax=ax[0])
        librosa.display.waveplot(data_whitenoise[0], ax=ax[1])
        plt.savefig("result/whitenoise.png", transparent=True)
        plt.show()
        print(len(data))
        print(len(data_whitenoise))
        print(1 / 0)
        """

    features = np.array(
        [np.mean(librosa.feature.mfcc(y=y, n_mfcc=13), axis=1) for y in data]
    )

    return features


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """

    cm = confusion_matrix(predict, ground_truth)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Ground truth")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig("result/cm_baseline_adam_whitenoise.png", transparent=True)
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("result_baseline_adam_whitenoise.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def plot_history(history):
    # 学習過程をグラフで出力
    # print(history.history.keys())
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    epochs = range(1, len(acc) + 1)
    plt.figure()
    plt.plot(epochs, acc)
    plt.grid()
    plt.title("Model accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("result/history_baseline_adam_whitenoise_acc.png", transparent=True)
    plt.show()

    plt.figure()
    plt.plot(epochs, loss)
    plt.grid()
    plt.title("Model loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("result/history_baseline_adam_whitenoise_loss.png", transparent=True)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス")
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出
    X_train = feature_extraction(training["path"].values, noise=True)
    X_test = feature_extraction(test["path"].values)

    # print("X_train.shape -> ", X_train.shape)
    # print("X_test.shape -> ", X_test.shape)

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)
    Y_train = np.tile(Y_train, (2, 1))

    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20200616,
    )

    # モデルの構築
    model = my_MLP(input_shape=X_train.shape[1], output_dim=10)

    # モデルの学習基準の設定
    model.compile(
        loss="categorical_crossentropy", optimizer=Adam(lr=1e-4), metrics=["accuracy"]
    )

    # モデルの学習
    history = model.fit(X_train, Y_train, batch_size=32, epochs=250, verbose=1)

    # モデル構成，学習した重みの保存
    model.save("keras_model/my_model_baseline_adam_whitenoise.h5")

    plot_history(history)

    # バリデーションセットによるモデルの評価
    # モデルをいろいろ試すときはテストデータを使ってしまうとリークになる可能性があるため、このバリデーションセットによる指標を用いてください
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print("Validation accuracy: ", score[1])

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
            title=f"Acc. {round(test_accuracy*100,2)}%",
        )
        print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
