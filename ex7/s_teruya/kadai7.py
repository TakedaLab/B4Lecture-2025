#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://qiita.com/yutalfa/items/dbd172138db60d461a56
# https://qiita.com/koshian2/items/ca99b4a489d164e9cec6
# https://zenn.dev/kthrlab_blog/articles/4e69b7d87a2538
# https://zenn.dev/totopironote/articles/aa17833ef00e5f
# https://dev.classmethod.jp/articles/python-assignment-expressions-study/

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト
特徴量；MFCCの平均（0次項含まず）
識別器；MLP
"""

from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.decomposition import PCA
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


def feature_extraction(path_list, pca: PCA = None, cache_file="cached_audio.pkl"):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """

    mtime = {p: os.path.getmtime(f"../{p}") for p in path_list}
    print("loading data...")
    try:  # データに変更がなければキャッシュから読み込み
        with open(cache_file, "rb") as f:
            dict_pd = pickle.load(f)
        if list(dict_pd["path"]) == list(path_list) and dict_pd["mtime"] == mtime:
            data = dict_pd["data"]
        else:
            raise ValueError("Cache invalid due to csv file update.")
    except Exception:
        load_data = lambda path: librosa.load(f"../{path}")[0]
        data = list(map(load_data, path_list))
        # キャッシュ保存
        dict_pd = {"path": path_list, "mtime": mtime, "data": data}
        with open(cache_file, "wb") as f:
            pickle.dump(dict_pd, f)

    print("extracting...")
    start = time.time()  # 実行時間計測
    mfccs = np.array(
        [
            np.concatenate(
                [
                    np.mean(
                        mfcc := librosa.feature.mfcc(y=y, n_mfcc=25)[1:], axis=1
                    ),  # 平均
                    np.std(mfcc, axis=1),  # 標準偏差
                    np.min(mfcc, axis=1),  # 最小値
                    np.max(mfcc, axis=1),  # 最大値
                    np.median(mfcc, axis=1),  # 中央値
                ]
            )
            for y in data
        ]
    )  # MFCC
    dmfccs = np.array(
        [
            np.concatenate(
                [
                    np.mean(
                        dmfcc := librosa.feature.delta(
                            librosa.feature.mfcc(y=y, n_mfcc=25), width=5
                        )[1:],
                        axis=1,
                    ),
                    np.std(dmfcc, axis=1),
                    np.min(dmfcc, axis=1),
                    np.max(dmfcc, axis=1),
                    np.median(dmfcc, axis=1),
                ]
            )
            for y in data
        ]
    )  # ΔMFCC
    ddmfccs = np.array(
        [
            np.concatenate(
                [
                    np.mean(
                        ddmfcc := librosa.feature.delta(
                            librosa.feature.mfcc(y=y, n_mfcc=25), order=2, width=5
                        )[1:],
                        axis=1,
                    ),
                    np.std(ddmfcc, axis=1),
                    np.min(ddmfcc, axis=1),
                    np.max(ddmfcc, axis=1),
                    np.median(ddmfcc, axis=1),
                ]
            )
            for y in data
        ]
    )  # ΔΔMFCC
    features = np.hstack(
        [mfccs, dmfccs, ddmfccs]
    )  # , mel_dbs, zcr_info, rms_info, drms_info

    if pca is None:
        pca = PCA(n_components=0.95)
        pca.fit(features)
    features_reduce = pca.transform(features)  # 次元削減
    end = time.time()  # 実行時間計測

    return features_reduce, pca, end - start


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
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    try:
        plt.savefig("picture/result.png")
    except Exception:
        pass
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

    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_to_truth", type=str, help="テストデータの正解ファイルCSVのパス"
    )
    args = parser.parse_args()
    if not os.path.exists(args.path_to_truth):
        print("Warning: --path_to_truth filename may be wrong")

    # データの読み込み
    training = pd.read_csv("../training.csv")
    test = pd.read_csv("../test.csv")
    print("Successfully reading csv")

    # 実行時間計測
    time_start = time.time()

    # 学習データの特徴抽出
    X_train, pca, t_train = feature_extraction(
        training["path"].values, cache_file="train_audio.pkl"
    )
    X_test, _, t_test = feature_extraction(
        test["path"].values, pca=pca, cache_file="test_audio.pkl"
    )
    print(
        f"complete feature_extraction in {t_train + t_test:.5g}, number of features is {X_train.shape[1]}"
    )

    # 正解ラベルをone-hotベクトルに変換 ex. 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)

    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを20%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=20200616,
    )

    # モデルの構築
    print("creating model")
    model = my_MLP(input_shape=X_train.shape[1], output_dim=10)

    # モデルの学習基準の設定
    model.compile(
        loss="categorical_crossentropy", optimizer=SGD(lr=0.002), metrics=["accuracy"]
    )

    # モデルの学習
    print("fitting data")
    model.fit(X_train, Y_train, batch_size=32, epochs=100, verbose=1)

    # モデル構成，学習した重みの保存
    model.save("keras_model/my_model.h5")

    # バリデーションセットによるモデルの評価
    # モデルをいろいろ試すときはテストデータを使ってしまうとリークになる可能性があるため、このバリデーションセットによる指標を用いてください
    score = model.evaluate(X_validation, Y_validation, verbose=0)
    print("Validation accuracy: ", score[1])

    # 予測結果
    print("predicting")
    predict = model.predict(X_test)
    predicted_values = np.argmax(predict, axis=1)

    # 実行時間計測
    time_end = time.time()
    print(f"Pattern Recognition finished in {time_end - time_start:.5g}")

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth["label"].values
        ac_score = accuracy_score(truth_values, predicted_values)
        print("Test accuracy: ", ac_score)
        plot_confusion_matrix(
            predicted_values, truth_values, title=f"(Accuracy:{ac_score})"
        )


if __name__ == "__main__":
    main()


# ---------------------------------(未使用)----------------------------------#
# mfccs = np.array(
#     [np.mean(librosa.feature.mfcc(y=y, n_mfcc=25)[1:], axis=1) for y in data]
# )  # MFCC
# dmfccs = np.array(
#     [
#         np.mean(
#             librosa.feature.delta(librosa.feature.mfcc(y=y, n_mfcc=25), width=5)[
#                 1:
#             ],
#             axis=1,
#         )
#         for y in data
#     ]
# )  # ΔMFCC
# ddmfccs = np.array(
#     [
#         np.mean(
#             librosa.feature.delta(
#                 librosa.feature.mfcc(y=y, n_mfcc=25), order=2, width=5
#             )[1:],
#             axis=1,
#         )
#         for y in data
#     ]
# )  # ΔΔMFCC
# mel_dbs = np.array(
#     [
#         np.mean(
#             librosa.power_to_db(librosa.feature.melspectrogram(y=y, n_mels=24)),
#             axis=1,
#         )
#         for y in data
#     ]
# )  # メル
# zcr_info = np.array(
#     [
#         [
#             np.mean(zcr := librosa.feature.zero_crossing_rate(y=y)),
#             np.std(zcr),
#             np.min(zcr),
#             np.max(zcr),
#             np.median(zcr),
#         ]
#         for y in data
#     ]
# )  # ZCR
# rms_info = np.array(
#     [
#         [
#             np.mean(rms := librosa.feature.rmse(y=y)),
#             np.std(rms),
#             np.min(rms),
#             np.max(rms),
#             np.median(rms),
#         ]
#         for y in data
#     ]
# )  # RMS
# drms_info = np.array(
#     [
#         [
#             np.mean(
#                 drms := librosa.feature.delta(librosa.feature.rmse(y=y), width=5)
#             ),
#             np.std(drms),
#             np.min(drms),
#             np.max(drms),
#             np.median(drms),
#         ]
#         for y in data
#     ]
# )  # ΔRMS
