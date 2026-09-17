<div align="center">

# Epsilon Arena

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [対戦設定](#4つの参加枠を設定する) · [候補モデルの比較](#候補モデルを比較する) · [結果](#結果を確認する) · [速度測定](#実行速度を測定する)

</div>

Epsilon-Pico、Epsilon-Nano、EpsilonのDecisionモデルを、同じプロセスのバッチ推論で対戦させます。同じ系列の異なる学習済みモデルも比較できます。

## はじめに

ビルドと動作環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。以下のコマンドはすべてリポジトリのルートから実行します。

利用できるコマンドと引数を確認します。

```bash
java -jar epsilon-arena/target/epsilon-arena-1.0-SNAPSHOT-all.jar help
```

このコマンドの初回実行時に`config/arena.toml`が作成されます。各設定の説明は[同梱テンプレート](src/main/resources/arena/settings.toml)にあります。既存の設定は上書きしません。

## 4つの参加枠を設定する

`config/arena.toml`の`[seat0]`〜`[seat3]`に、対戦するモデルを指定します。テンプレートのcheckpointは配置例なので、実際に保存したDecisionモデルのディレクトリへ変更してください。

| 項目                           | 指定する内容                                              |
|------------------------------|-----------------------------------------------------|
| `[seat0]`〜`[seat3]`の`series` | `epsilon-pico`、`epsilon-nano`、`epsilon`のいずれか        |
| 各参加枠の`checkpoint`            | 対応する系列の学習済みモデルのディレクトリ                               |
| 各`[seatN.options]`の`device`  | `auto`、`cpu`、`gpu:0`など。`auto`は利用可能な全GPU、GPUがなければCPU |
| `[arena]`の`games`            | 半荘数。4以上の4の倍数。既定は256                                 |
| `[arena]`の`concurrentGames`  | 同時に進める半荘数。既定は128                                    |

同じ牌山のseedで4半荘を実行し、参加枠を東南西北に循環させます。`seat0`〜`seat3`は集計上の参加枠で、固定の席ではありません。同じ系列・checkpoint・設定を複数枠に指定した場合、モデルは共有して読み込みます。

設定した4枠で対戦し、結果を保存します。

```bash
java -jar epsilon-arena/target/epsilon-arena-1.0-SNAPSHOT-all.jar match --output arena-results.json
```

別の設定ファイルを使う場合は`match --config config/my-arena.toml`を指定します。明示したファイルと4つの参加枠はすべて必要です。checkpointと`[seatN.options]`の`settings`の相対パスは、設定ファイルのあるディレクトリを基準にします。例えば`config/arena.toml`の`../checkpoints/epsilon-nano/decision_00001`は、リポジトリ直下の`checkpoints`を指します。

## 候補モデルを比較する

`duel`は候補モデルを1枠、比較相手を3枠に配置します。対戦条件をコマンド引数で指定するため、`config/arena.toml`の参加枠は使いません。

```bash
java -jar epsilon-arena/target/epsilon-arena-1.0-SNAPSHOT-all.jar duel --candidate-series epsilon-nano --candidate-checkpoint checkpoints/epsilon-nano/decision_00001 --opponent-series epsilon --opponent-checkpoint checkpoints/epsilon/decision_00001 --games 256 --device cpu --output arena-duel.json
```

`--candidate-settings`と`--opponent-settings`で系列ごとの設定ファイルも指定できます。これらとcheckpointの相対パスは、コマンドを実行した場所を基準にします。省略した対局数・並列数などには、同梱テンプレートの`[arena]`の既定値を使います。

## 結果を確認する

`--output`で指定したJSONファイルには、使用したモデル・実行条件と次の結果を保存します。保存先の相対パスは、コマンドを実行した場所を基準にします。

| 出力                      | 内容                                 |
|-------------------------|------------------------------------|
| `runs[].players`        | 各参加枠の平均順位・平均得点・1〜4位の回数             |
| `runs[].paired`         | 参加枠0と他3枠の平均の差、牌山の組ごとの標準誤差・95%区間の下限 |
| `runs[].gamesPerSecond` | 1秒あたりの半荘数                          |
| `runs[].inference`      | モデルごとの推論行数・処理速度・バッチ充填率・待機時間        |
| `gpuMemory`             | 各測定の前後のGPUメモリ使用量                   |

`rankAdvantage`と`scoreAdvantage`は、正なら参加枠0が優勢です。`duel`では候補モデルが参加枠0に当たります。標準誤差は4半荘をまとめた牌山の組を単位に計算します。1組だけの場合は分散を推定できず、出力値は0になります。

結果は実行ログにも表示します。JSONを別のツールで読み込む場合は、ログの接頭辞を含まない`--output`の保存ファイルを使ってください。

## 実行速度を測定する

`--warmup-games`で測定前に対局を実行し、`--repetitions`で同じ条件の測定を繰り返します。

```bash
java -jar epsilon-arena/target/epsilon-arena-1.0-SNAPSHOT-all.jar match --warmup-games 32 --repetitions 3 --output arena-benchmark.json
```

この例では32半荘をウォームアップした後、設定した半荘数を3回測定します。各回は同じ牌山の範囲を使い、結果を`runs`に分けて保存します。ウォームアップの対局は成績・速度の集計に含めません。GPUメモリのpeak値にはウォームアップ中の使用量も含まれます。

速度を比較するときは、checkpoint、device、精度、バッチ設定、並列対局数、CPU worker数、seed、ウォームアップ条件を揃えてください。

<details>
<summary>詳細設定とモデルの追加</summary>

`[arena]`の`workers`は、省略するとJVMが認識するCPU論理数の半分（最低1）になります。`firstWallFamily`はウォームアップを始める牌山の組番号で、測定はそこから`warmupGames / 4`組だけ進めます。バッチ待機時間やGPUの実行枠数は[同梱テンプレート](src/main/resources/arena/settings.toml)で確認できます。

各`[seatN.options]`には、複数デバイスを指定する`devices`、系列の設定ファイルを指定する`settings`、個別設定の`epsilon.*`キーを記述できます。`device`と`devices`はどちらか一方を使います。モデル構造はcheckpointから復元します。

新しい系列を追加する場合は、系列側で[ModelProvider](../epsilon-common/src/main/java/com/epsilon/spi/ModelProvider.java)と[BatchedPolicy](../epsilon-common/src/main/java/com/epsilon/spi/BatchedPolicy.java)を実装し、Arenaの[Maven依存](pom.xml)と[起動時のprovider一覧](src/main/java/com/epsilon/arena/Main.java)に登録します。入力・ネットワーク・checkpointの解釈は系列側が所有し、[ArenaRunner](src/main/java/com/epsilon/arena/ArenaRunner.java)は共通の推論契約で対戦を進めます。

</details>

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
