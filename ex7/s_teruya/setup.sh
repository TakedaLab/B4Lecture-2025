#!/bin/bash

if  [ "$1" = "admin"  ];then
    # データセットのダウンロード
    git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
fi

# 必要なパッケージのインストール
pip install -r requirements.txt

if  [ ! -e keras_model ];then
    python -c 'import keras'
    mkdir keras_model
fi
