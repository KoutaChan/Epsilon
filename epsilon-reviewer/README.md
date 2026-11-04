<div align="center">

# Epsilon Reviewer

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Electron](https://img.shields.io/badge/Electron-47848F?logo=electron&logoColor=white)](https://www.electronjs.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [モデル設定](#モデルを登録する) · [Web](#webで使う) · [共有](#解析結果を共有する) · [もしも](#もしもで局面を変える) · [PCアプリ](#pcアプリで使う) · [対応牌譜](#対応する牌譜) · [保存と履歴](#保存と履歴) · [開発](#開発)

</div>

PC・スマホのブラウザーと、解析エンジンを同梱したWindowsアプリで利用できます。Epsilon-Pico、Epsilon-Nano、Epsilon、Epsilon-Prismの学習済みモデルに対応し、登録したモデルを画面で選択できます。

- 打牌・鳴き・リーチ・カン・和了・見送りを含む、AIの候補手の上位3件を表示。上位外の実際の選択も区切って表示し、価値を推定するモデルでは最善手との損失（最善・ほぼ同等・やや損・悪手）を主に示す。
- 局とプレイヤーの切り替え、手牌公開の切り替え、自動再生に対応。前後ボタン・左右矢印キー・卓上の縦スクロールで手順を移動。
- 和了手牌、役、表ドラ・裏ドラ、局ごとの点数変化、最終順位を表示。
- 「もしも」で河・ドラ表示牌・自分の手牌・点数・本場・供託を変えたり、チー・ポンした後の打牌を見たりして、その判断をAIに解析し直させる。
- Web版は解析結果ごとに共有リンクが付き、相手も同じレビュー画面を開けます。

AIの判断には各プレイヤーから見える情報を使い、他家の手牌は入力しません。解析には、学習済みチェックポイントの登録が必要です。

## 構成

| 場所 | 中身 |
| --- | --- |
| `backend/` | Javaの解析エンジン。1つのJARに、Web用のHTTPサーバー（`com.epsilon.reviewer.Main`）とPCアプリ用の子プロセス（`com.epsilon.reviewer.StdioMain`）が入る |
| `web/` | React・TypeScript・Three.jsの画面。WebとPCアプリで同じ`web/dist`を使う |
| `desktop/` | Electronのアプリ。Javaのエンジンを子プロセスとして起動し、標準入出力でやり取りする（ポートを開かない） |
| `protocol/` | 3つの間の契約（操作の表、解析結果の形式、失敗の符号、標準入出力の枠）。Javaの試験がこの表と突き合わせる |
| `language/` | 日本語・英語の文言（`language/<言語>/{shell,replay,mahjong,desktop,errors}.json`） |
| `tools/` | 文言の検査（`check-language.mjs`）と牌の画像の取り込み（`vendor-tiles.mjs`） |

## はじめに

Javaのビルドと実行環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。画面とElectronのビルド・テストにはNode.js 24を使います。

リポジトリのルートからJavaのエンジンをビルドし、続けて画面をビルドします。

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

`models.local.json`の`checkpoint`を、実際の学習済みモデルのディレクトリへ変更します。

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
  ],
  "defaultModel": { "modelId": "epsilon-nano", "revision": "nano-001" }
}
```

- `checkpoint`はパラメーターとモデル構成を含むディレクトリです。相対パスは設定JSONの場所を基準にします。Epsilon-Prismでは版のディレクトリ（`runs/<run>/snapshots/<t>`）を指定します。
- `series`には`epsilon-pico`・`epsilon-nano`・`epsilon`・`epsilon-prism`のいずれかを指定します（クラスパスに提供元がある系列だけが使えます）。
- `sha256`（任意）はチェックポイントのディレクトリのSHA-256です。指定すると、初めて開くときに中身を確かめ、違えばそのモデルを使えなくします。`defaultModel`（任意）は画面で最初に選ぶモデルです。
- 知らない欄・同じ`modelId`と`revision`の重複・一覧に無い`defaultModel`は起動の失敗になります。チェックポイントを読めないモデルは、起動は止めずに「使えない」と表示します。

`models`に複数の項目を登録すると、解析前にモデルを選べます。`modelId`と`revision`で使用したモデルを識別し、`displayName`と`version`を画面に表示します。重みを更新する際は別の`revision`を設定し、サーバーを再起動してください。開いたモデルは解析ともしもで共有し、10分使わなければ閉じます。

## Webで使う

```powershell
npm run server
```

[http://127.0.0.1:8080](http://127.0.0.1:8080)を開き、牌譜とモデルを選んで「解析する」を押します。完了した結果は履歴の「開く」から確認できます。サーバーを終了するには`Ctrl+C`を押します。

このコマンドはビルド済みのJARと画面を使います。ビルドや学習済みモデルの取得は行いません。

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

サーバーは`--origin`と一致しないHostやOriginからの要求を断ります。画面のファイルは起動時に1回だけ読み、ビルドが作った`.br`・`.gz`の圧縮版を送ります。画面を作り直したらサーバーを再起動してください。

## 解析結果を共有する

Web版では、解析結果を保存した時点で共有リンク（`<--origin>/?share=<結果のID>`）が決まります。履歴のメニュー、卓の「…」、最終結果画面にある「共有リンク」から「リンクをコピー」を押してください。共有開始・停止の操作はありません。HTTP接続などで自動コピーが使えない場合は、表示されたリンクを手動でコピーできます。

リンクを知っている人は、Cookieやログインなしで対局者名・全員の手牌・AIの解析結果を閲覧できます。局・プレイヤー・表示言語を自由に切り替えられ、そのモデルがサーバーで使えればもしもも使えます。共有するのは該当する解析結果だけで、所有者の履歴や元牌譜を取得・削除する権限は渡しません。元牌譜は閲覧者へ送らず、もしもはサーバーが保存した牌譜で解析します。リンクを開いても閲覧者の履歴へは登録されません。

リンクはサーバーを再起動しても変わりません。履歴から解析結果を削除すると、そのリンクもすぐに開けなくなります。PCアプリの解析結果は共有の対象外です。

共有リンクのURLはサーバーの`--origin`を使います。別の端末から開く場合は、その端末から接続できるアドレスを指定してください。`127.0.0.1`は同じPC内での利用向けです。

## もしもで局面を変える

解析結果の卓で「もしも」を押すと、見ている席の判断の局面を変えて、AIに解析し直させられます。

| 変えられるもの | 決まり |
| --- | --- |
| 他家の河の牌と、手出し・ツモ切り | 鳴かれた牌は変えられない。リーチ宣言牌は変えられる |
| 自分の河の牌 | 鳴かれた牌は変えられない。手出し・ツモ切りは記録のまま（その牌が入った配牌かツモも同じ牌に替える） |
| 開いているドラ表示牌 | 槓ドラを含む |
| 自分の手牌とツモ牌 | 枚数は変わらない。リーチの後は変えられない。同じ牌が複数あれば、最後に手に入った1枚を替える |
| 4人の点数・本場・供託 | 点数は100点単位 |
| チー・ポン | 応答の判断で鳴いたとして、鳴いた後の打牌を解析する |

牌を変えるときは、見えている牌が5枚以上（赤は2枚以上）になる牌を選べません。解析は判断の局面だけを作り直し、変えた手牌と河から前の見逃しによるフリテンやリーチの可否も作り直します。結果は元の候補との差（順位・選択率・損失）と一緒に表示します。

もしもは同時に1件を解析し、4件まで待たせます。混んでいるときは少し待ってからもう一度押してください。

## PCアプリで使う

PCアプリは、Javaの実行環境・解析エンジン・学習済みモデルを同梱した全部入りのWindowsアプリです。サーバーは要らず、解析ともしもはこのPCの中で動きます。ネットワークを使うのは、天鳳の牌譜URLから牌譜を取得するときだけです。

### 全部入りのアプリを作る

Windows x64向けの次の5つを揃えてから、`desktop/engine/`を組み、Electronで展開済みのアプリを作ります。

1. デスクトップ用のJAR（ROCmのJNIと他のOSのネイティブを除いたもの）。リポジトリのルートで作ります。

   ```powershell
   .\mvnw.cmd -pl epsilon-reviewer/backend -am -DskipTests package "-P!linux-native,desktop"
   ```

   `backend/target/epsilon-reviewer-1.0-SNAPSHOT-desktop.jar`ができます。

2. Java 21のWindows x64の実行環境。WindowsのJDKの`jmods`から`jlink`で作ります。

   ```powershell
   jlink --module-path "<JDK 21>\jmods" `
     --add-modules java.base,java.xml,jdk.jfr,jdk.unsupported,java.management,java.logging,java.naming,java.net.http,jdk.crypto.ec `
     --strip-debug --no-man-pages --no-header-files --compress=zip-6 --output runtime
   ```

3. libtorch 2.11.0のCPU版のDLL（既定は`%USERPROFILE%\.djl.ai\pytorch\2.11.0-cpu-win-x86_64`）と、`djl_torch.dll`を含むDJLのJNIのJAR（既定は`~/.m2`の`pytorch-jni-cpu-win-x86_64-2.11.0-0.40.4-rocm-SNAPSHOT.jar`）。7つのDLLの大きさとSHA-256は`desktop/scripts/native-files.json`で固定しています。

4. 同梱するモデルの`models.json`（[モデルを登録する](#モデルを登録する)と同じ形）。

5. 画面のビルド（`npm run build`）。

揃ったら`desktop/engine/`を組んで、アプリを作ります。

```powershell
npm run desktop:engine -- --runtime <実行環境> --jar <デスクトップ用のJAR> --models <models.json>
npm run desktop:pack
```

`desktop:engine`は推論に使うファイルだけを`engine/models/<modelId>/<revision>/`へ写し、相対パス・CPU・`sha256`付きの`models.json`を書きます。`--libtorch`と`--jni`で3.の場所を変えられます。`desktop:pack`は`desktop/release/win-unpacked/`に展開済みのアプリを作ります。インストーラーは作りません。

`desktop/engine/`を組んだ後は、`npm run desktop`で梱包せずにアプリを起動できます（画面は先に`npm run build`で作った`web/dist`を使います）。

### 設定

設定画面では、解析結果の保存先と表示言語を変えられます。保存先の既定はOSの「ドキュメント」内の`epsilon-reviewer`です。保存先を変えるとエンジンを起動し直し、そのフォルダーの解析結果を履歴に表示します。解析の途中は保存先を変えられません。

<details>
<summary>設定用プロファイルを分ける</summary>

起動前に環境変数を指定します。

```powershell
$env:EPSILON_REVIEWER_PROFILE = Join-Path $env:LOCALAPPDATA 'epsilon-reviewer-dev'
npm run desktop
```

`EPSILON_REVIEWER_PROFILE`は絶対パスで指定します。設定（`settings.json`）・エンジンの記録（`logs/`）・DJLの一時ファイル（`djl/`）をこのフォルダーに置きます。`EPSILON_REVIEWER_DEV_URL`に`http://127.0.0.1:5173`のような手元のViteのURLを指定すると、その画面を開きます。

</details>

## 対応する牌譜

| 提供元 | 入力形式 |
| --- | --- |
| 天鳳 | 牌譜XML、牌譜URL |
| mjai | JSON Lines、イベント配列のJSON。`.mjai`・`.mjson`・`.jsonl`・`.json` |
| 雀魂 | 復号済みJSON、`GameDetailRecords`・`ResGameRecord`のprotobufデータ |

四人麻雀の、全員の初期手牌とツモ牌を復元できる牌譜が対象です。gzip圧縮にも対応し、入力・展開後とも上限は16 MiBです。三人麻雀や未知牌を含む記録、解析結果のファイルは読み込めません。

雀魂は取得済みのJSON・protobufファイルに対応しています。

役・裏ドラ・和了点は牌譜の記録を使います。記録に役が無い牌譜（Mortal などの mjai）は天鳳の規則で計算し、計算した点の移動が牌譜と一致した和了だけ表示します。最終点数は記録値を優先し、対局終了と最終局の精算が揃う場合だけ復元します。途中で終わる牌譜では、確定した最終順位として表示しません。

## 保存と履歴

WebとPCアプリは同じ形で保存します。Webはサーバーの`--data-dir`（上の起動例では`.data`）、PCアプリは設定の保存先のフォルダーが保存の根です。

```
<保存の根>/
  reviewer.lock            動いている間の鍵（同じフォルダーを2つのエンジンで開かない）
  results/<結果のID>/
    result.json.gz         解析結果
    record.gz              元の牌譜（再解析ともしもに使う）
    meta.json              履歴の1件（持ち主・日時・対局の概要・モデル）
  staging/                 保存の途中（起動時に消す）
  trash/                   削除の途中（起動時と定期の掃除で消す）
```

`staging/`と`trash/`の掃除で消すのは、名前がUUIDのフォルダーの中の`result.json.gz`・`record.gz`・`meta.json`と、空になったそのフォルダーだけです。既にあるフォルダーを保存先に選んでも、他の名前のファイルやフォルダーは消しません。

| 利用方法 | 履歴 |
| --- | --- |
| Web | Cookieの匿名識別子ごとに、サーバー上の履歴を表示 |
| PCアプリ | 保存先のフォルダーの全ての解析結果を表示 |

WebのCookieは本人の履歴の管理に使います。共有リンクからの閲覧にはCookieを使いません。WebのCookieには解析結果本体を保存しません。Cookieを削除すると以前の履歴との紐付けを失い、アカウントによる復旧もできません。

履歴の「解析結果を削除」で、対応する解析結果と元牌譜もすぐに見えなくなり、共有リンクも開けなくなります。「解析結果をファイルに保存」は解析結果を1つの`<結果のID>.epsilon-reviewer.json.gz`ファイルにします（Webはダウンロード、PCアプリは保存の画面）。外部の解析結果ファイルを履歴へ登録する機能はありません。

解析結果は、局ごとの最初の局面と手順ごとの差分、判断ごとの候補（行動IDと値、鳴き・カンだけ牌）をまとめたJSONを、gzipで圧縮して保存します。価値を推定するモデルでは最善手との損失を丸めずに持ち、選択率は損失と温度から画面が作ります。半荘1回でおよそ65 KBです。サーバーは保存したまま送り、ブラウザーは同じ結果を開き直すときに送り直しを受けません。

## 開発

画面はReact・TypeScriptとThree.js、PCアプリはElectron、解析エンジンはJavaです。牌譜の読込・再生を学習側と共有し、モデルの読込と推論は各系列が担当します。

日本語・英語の表示文言は`language/<言語>/`で管理し、`npm run check:language`がどの言語も同じ鍵と`{名前}`を持つことを確かめます。内部ログとエラー文は英語です。

```powershell
npm run typecheck
npm test
```

Javaのテストはリポジトリのルートで`.\mvnw.cmd -pl epsilon-reviewer/backend -am test`を実行します。CPUの実推論を含める場合は`-Pnative-tests`を付けてください。

画面の開発中は、Javaサーバーを`127.0.0.1:8080`で起動したまま`npm run dev`を実行します。Viteの開発画面からのAPIリクエストはJavaサーバーへ転送されます。転送先は環境変数`EPSILON_REVIEWER_API`で変えられます。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
