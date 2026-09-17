<div align="center">

# Epsilon Reviewer

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Electron](https://img.shields.io/badge/Electron-47848F?logo=electron&logoColor=white)](https://www.electronjs.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [モデル設定](#モデルを登録する) · [Web](#webで使う) · [PCアプリ](#pcアプリで使う) · [対応牌譜](#対応する牌譜) · [保存と履歴](#保存と履歴)

</div>

PC・スマホのブラウザーとElectronアプリで利用できます。Epsilon-Pico、Epsilon-Nano、Epsilonの学習済みモデルに対応し、サーバーに登録したモデルを画面で選択できます。

- 打牌・鳴き・リーチ・カン・和了・見送りを含む、AIの候補手の上位3件を表示。
- 局とプレイヤーの切り替え、手牌公開の切り替え、自動再生に対応。前後ボタン・左右矢印キー・卓上の縦スクロールで手順を移動。
- 和了手牌、役、表ドラ・裏ドラ、局ごとの点数変化、最終順位を表示。
- Web版は解析結果ごとに共有リンクが付き、相手も同じレビュー画面を開けます。

AIの判断には各プレイヤーから見える情報を使い、他家の手牌は入力しません。解析には、学習済みチェックポイントの登録が必要です。

## はじめに

Javaのビルドと実行環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。画面とElectronのビルド・テストにはNode.js 24を使います。

リポジトリのルートからJavaサーバーをビルドし、続けて画面をビルドします。

```powershell
.\mvnw.cmd -pl epsilon-reviewer/backend -am -DskipTests package
cd epsilon-reviewer
npm ci
npm run build
```

Linux・macOSではMavenのコマンドを`./mvnw`に置き換えてください。以降のコマンドは、`epsilon-reviewer`ディレクトリから実行します。

## モデルを登録する

初回は[設定例](backend/models.example.json)をコピーします。

```powershell
Copy-Item backend/models.example.json models.local.json
```

`models.local.json`の`checkpoint`を、実際の学習済みモデルのディレクトリへ変更します。例えば、リポジトリ内の`checkpoints/epsilon-nano/decision_pretrain_logs_00001`を使う場合は次のように指定します。

```json
{
  "models": [
    {
      "modelId": "epsilon-nano",
      "revision": "nano-001",
      "displayName": "Epsilon-Nano",
      "version": "001",
      "series": "epsilon-nano",
      "checkpoint": "../checkpoints/epsilon-nano/decision_pretrain_logs_00001",
      "options": { "device": "cpu" }
    }
  ]
}
```

`checkpoint`はパラメーターとモデル構成を含むディレクトリです。相対パスは設定JSONの場所を基準にします。`series`には、そのモデルと同じ`epsilon-pico`・`epsilon-nano`・`epsilon`のいずれかを指定してください。

`models`に複数の項目を登録すると、解析前にモデルを選べます。`modelId`と`revision`で使用したモデルを識別し、`displayName`と`version`を画面に表示します。重みを更新する際は別の`revision`を設定し、サーバーを再起動してください。

## Webで使う

```powershell
npm run server
```

[http://127.0.0.1:8080](http://127.0.0.1:8080)を開き、牌譜とモデルを選んで「解析する」を押します。完了した結果は履歴の「開く」から確認できます。サーバーを終了するには`Ctrl+C`を押します。

このコマンドはビルド済みのJavaサーバーと画面を使います。ビルドや学習済みモデルの取得は行いません。

<details>
<summary>スマホから同じLANで使う</summary>

PCとスマホを同じLANに接続します。以下の`192.168.1.2`は例示用のアドレスです。利用するPCのIPv4アドレスに置き換え、サーバーを停止して次の引数で起動し直します。

```powershell
java -jar backend/target/epsilon-reviewer-1.0-SNAPSHOT-all.jar `
  --host 0.0.0.0 --port 8080 --origin http://192.168.1.2:8080 `
  --models models.local.json --data-dir .data --web-dir web/dist
```

両方の端末で`http://192.168.1.2:8080`を開きます。`--origin`はブラウザーで開くURLと一致させ、PCのファイアウォールで利用するLANからの接続を許可してください。スマホにJavaやElectronをインストールする必要はありません。

Linux・macOSでは行末のバッククォートを`\`に置き換えるか、1行で実行してください。公開サーバーではreverse proxyでHTTPSを提供し、`--origin`に公開URLを指定してください。

</details>

## 解析結果を共有する

Web版では、解析結果を保存した時点で共有リンクが決まります。履歴のメニュー、卓の「…」、最終結果画面にある「共有リンク」から「リンクをコピー」を押してください。共有開始・停止の操作はありません。HTTP接続などで自動コピーが使えない場合は、表示されたリンクを手動でコピーできます。

リンクを知っている人は、Cookieやログインなしで対局者名・全員の手牌・AIの解析結果を閲覧できます。局・プレイヤー・表示言語を自由に切り替えられます。共有するのは該当する解析結果だけで、所有者の履歴や元牌譜を取得・削除する権限は渡しません。リンクを開いても閲覧者の履歴へは登録されません。

リンクは同じ結果を参照し、サーバーを再起動しても変わりません。履歴から解析結果を削除すると、そのリンクも開けなくなります。結果を複製したり、クライアントから解析結果をアップロードしたりする機能はありません。PCアプリのローカル保存結果は共有対象外です。

共有リンクのURLはサーバーの`--origin`を使います。別の端末から開く場合は、その端末から接続できるアドレスを指定してください。`127.0.0.1`は同じPC内での利用向けです。LANで使う場合は[上記の起動例](#webで使う)、インターネットで共有する場合はHTTPSの公開URLを使用します。

## PCアプリで使う

Javaサーバーを起動したまま、別のターミナルで`epsilon-reviewer`ディレクトリへ移動して実行します。

```powershell
npm run desktop
```

初期接続先は`http://127.0.0.1:8080`です。アプリの設定画面で接続先と結果の保存先を変更できます。ElectronはJavaサーバーを起動しないため、新しい解析・再解析にはサーバーとの接続が必要です。PCに保存した結果はオフラインでも開けます。

`npm run desktop:pack`で`desktop/release/`に展開済みアプリを生成できます。Javaランタイム・サーバーJAR・学習済みモデルは含まず、インストーラーも作成しません。

<details>
<summary>初期接続先と設定用プロファイルを分ける</summary>

起動前に環境変数を指定します。

```powershell
$env:EPSILON_REVIEWER_SERVER = 'http://192.168.1.2:8080'
$env:EPSILON_REVIEWER_PROFILE = Join-Path $env:LOCALAPPDATA 'epsilon-reviewer-dev'
npm run desktop
```

`EPSILON_REVIEWER_PROFILE`は絶対パスで指定します。接続先に保存済みの設定がある場合は、環境変数よりその設定を優先します。

</details>

## 対応する牌譜

| 提供元 | 入力形式 |
| --- | --- |
| 天鳳 | 牌譜XML、牌譜URL |
| mjai | JSON Lines、イベント配列のJSON。`.mjai`・`.mjson`・`.jsonl`・`.json` |
| 雀魂 | 復号済みJSON、`GameDetailRecords`・`ResGameRecord`のprotobufデータ |

四人麻雀の、全員の初期手牌とツモ牌を復元できる牌譜が対象です。gzip圧縮にも対応し、入力・展開後とも上限は16 MiBです。三人麻雀や未知牌を含む記録には対応していません。

雀魂は取得済みのJSON・protobufファイルに対応しています。

役・裏ドラ・和了点は牌譜の記録を使います。最終点数は記録値を優先し、対局終了と最終局の精算が揃う場合だけ復元します。途中で終わる牌譜では、確定した最終順位として表示しません。

## 保存と履歴

解析結果、対局者名と点数、使用モデル、再解析用の元牌譜を保存します。閲覧位置は保存せず、WebとPCアプリの履歴は別々に管理します。

| 利用方法 | 結果と元牌譜の保存先 | 履歴の管理 |
| --- | --- | --- |
| Web | サーバーの`--data-dir`配下の`results/`。上の起動例では`.data/results/` | Cookieの匿名識別子でサーバー上の履歴を取得 |
| PCアプリ | OSの「ドキュメント」内の`epsilon-reviewer/results/`。設定画面で変更可能 | PCのアプリ設定フォルダー内の`review/` |

WebのCookieは本人の履歴の管理に使います。共有リンクからの閲覧にはCookieを使いません。WebのCookieには解析結果本体を保存しません。Cookieを削除すると以前の履歴との紐付けを失い、アカウントによる復旧もできません。

PCアプリは結果を`<resultId>.epsilon-reviewer.json.gz`、元牌譜を隣接する`.source.json`へ保存してから、サーバーの一時結果を削除します。未回収の一時結果も24時間後の削除対象です。保存先を変更しても既存ファイルは移動しません。

履歴を削除すると対応する結果と元牌譜も削除されます。外部の解析結果ファイルを履歴へ登録する機能はありません。保存形式は`formatVersion:4`のみを扱います。旧形式の履歴は表示せず、元牌譜から再解析してください。旧ファイルの自動変換・自動削除は行いません。

解析結果は、局面の差分と32局面ごとの復元用局面をまとめたJSONをgzip圧縮して保存します。同じ局面を参照するイベントとAI判断は局面を共有し、候補手と選択率は全件保持します。画面は必要な範囲だけを復元します。

## 開発

画面はReact・TypeScriptとThree.js、PCアプリはElectron、解析サーバーはJavaです。牌譜の読込・再生を学習側と共有し、モデルの読込と推論は各系列が担当します。

日本語・英語の表示文言は[language/ja.json](language/ja.json)と[language/en.json](language/en.json)で管理しています。内部ログとエラー文は英語です。

```powershell
npm run typecheck
npm test
```

Javaのテストはリポジトリのルートで`.\mvnw.cmd -pl epsilon-reviewer/backend -am test`を実行します。CPUの実推論を含める場合は`-Pnative-tests`を付けてください。

画面の開発中は、Javaサーバーを`127.0.0.1:8080`で起動したまま`npm run dev`を実行します。Viteの開発画面からのAPIリクエストはJavaサーバーへ転送されます。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
