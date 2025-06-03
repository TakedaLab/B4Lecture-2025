# Pytorch Lightning版 ベースライン

以下 `pwd = ex9/pytorch_lightning/` を前提とする

## 準備
適当な仮想環境のもとで以下のコマンドを実行
```sh
pip install -r requirement.txt
```


## 実行

```sh
python baseline.py
```
テストする場合
```sh
python baseline.py --path_to_truth [正解テストラベル付きcsvファイルパス]
```

## 結果の確認

結果は全て[TensorBoard](https://www.tensorflow.org/tensorboard?hl=ja)に出力される。

以下の方法でGUIを開いたあと、「SCALARS」タブから学習曲線が、「IMAGES」タブからテストデータの混同行列が見られる。

- vscodeユーザー

  「コマンドパレット(ctrl+shift+p)->Launch Tensorboard」とすれば vscode 上でGUIを開ける

- 非vscodeユーザー

  以下のコマンドを打った後、出力されたURLに飛ぶ
  ```sh
  tensorboard --logdir ./lightning_logs
  ```
