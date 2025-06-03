# 武田研究室B4輪講 演習問題

本リポジトリは，武田研究室のB4輪講の演習問題を取り扱う．

## 演習の始め方

1. 本リポジトリを自分のアカウントにフォークする
  (右上のForkボタンを押す)

2. フォークした自分のリポジトリに自動的に遷移する。

3. フォークした自分のリポジトリを適当な場所へダウンロードする。(リポジトリページ右上の[↓Code]ボタンからURLをコピー)

    ```bash
    $ git clone <自分のgithubのURL>
    ```

4. 本家リポジトリを登録 (upstreamという名前でなくてもいい)

    ```bash
    $ cd B4Lecture-2023
    $ git remote add upstream https://github.com/TakedaLab/B4Lecture-2023.git
    ```


## 演習の進め方

1. masterブランチに戻る
  ```bash
  $ cd B4Lecture-2023
  $ git checkout master
  ```
2. 本家リポジトリから更新されたソースをマージする
  ```bash
  $ git fetch upstream
  $ git merge upstream/master
  ```
3. ブランチを作成する
  ```bash
  $ git checkout -b ex_XX (ブランチ名。何でもいい)
  ```
3. 自分の名前のディレクトリを作成する
  ```bash
  $ mkdir -p exXX/d_yoshioka
  ```
6. ディレクトリ内でスクリプトを作成する

7. 適宜gitを使ってコミットする(ローカルのgitに反映される。こまめにやっていいよ)
  ```bash
  # 例
  git add main.py
  git commit -m "新しい関数を追加"
  ```

8. githubにpushする(フォークした自分のgithubに反映される。こまめにやっていいよ)
  ```bash
  $ git push origin exXX (ブランチ名)
  ```

9. 一通り実装したら、githubにアクセスしてプルリクエストを作成し，レビューをお願いする（下参照）

10. レビューを受けてRequest Changesを修正 -> add -> commit -> pushを繰り返す

11. 必須レビュアー（修士学生１人）がApproveしたら、米山がマージする。マージされたら本家リポジトリに自分のコードが反映される。


## プルリク出す時

課題ができたら(一旦結果を出力できたら)先輩にコードレビューを依頼する
以下の事項に注意しプルリクエストを送信する

- [ ] タイトルは **[名前] EX○ 解答 [matlab or python]** になっている
- [ ] コメントが適切に書けている
- [ ] 変数名，関数名はわかりやすいものになっている
- [ ] **演習を進める上でのコツ** のコードレビューまとめを参照し過去と似たようなことを指摘されないように注意する
- [ ] 出力画像も添付する
- [ ] どうしても解決できない部分がある場合は、その詳細も書くこと

## その他

[演習を進める上でのコツ](docs/TIPS.md)

[VSCodeを使いこなそう](docs/vscode.md)

[GitとGithubの理解](https://docs.google.com/a/g.sp.m.is.nagoya-u.ac.jp/viewer?a=v&pid=sites&srcid=Zy5zcC5tLmlzLm5hZ295YS11LmFjLmpwfHNwbG9jYWwtc2VtaXxneDoxZmI4YWVhZWVlNDBjNDY1)

[スライドの作り方](https://www.slideshare.net/ShinnosukeTakamichi/ss-48987441)

[PowerPointを使いやすくするために](https://drive.google.com/file/d/1DIMUqkFAyphAla_q7Ec5-M-1p3_GlUHx/view?usp=sharing)

[過去資料](https://sites.google.com/a/g.sp.m.is.nagoya-u.ac.jp/splocal-semi/b4-rinkou)

[講義ビデオ](https://drive.google.com/drive/folders/1aOAgjTjUutiw3qwPwpRKhxDrnY0n5XEX?usp=sharing)

---

## 進行係向けマニュアル

[Github Actionsについて](docs/actions_manual.md)
