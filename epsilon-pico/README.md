<div align="center">

# Epsilon-Pico

**Riichi Mahjong AI**

牌譜から行動の選び方と局面の評価を学習する麻雀AI。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Parameters: 1.51M](https://img.shields.io/badge/Parameters-1.51M-7c3aed)](../README.md#epsilon-シリーズ)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [設定](#設定と保存先) · [学習](#牌譜から学習する) · [評価](#学習したモデルを評価する) · [オンライン対局](#自己対戦とオンライン対局)

</div>

> [!IMPORTANT]
> 学習済みモデルと牌譜は同梱していません。

既定のDecisionモデルは **1,507,396パラメーター（約1.51M）** です。Policy・Valueを含み、別モデルのGRP・Beliefは含みません。

## はじめに

ビルドと動作環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。以下のコマンドはすべてリポジトリのルートから実行します。

利用できるコマンドと引数を確認します。

```bash
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar help
```

学習済みモデルがなくても、対局エンジンのベンチマークを実行できます。

```bash
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar benchmark 1000
```

## 設定と保存先

初回起動時に`config/epsilon-pico.toml`が作成されます。使用するGPU、学習率、バッチサイズなどをこのファイルで変更できます。各項目の説明は[同梱設定](src/main/resources/pico/settings.toml)にあります。

| 用途 | 既定の場所 |
| --- | --- |
| 設定ファイル | `config/epsilon-pico.toml` |
| 学習用の牌譜 | `data/logs` |
| 学習結果 | `checkpoints/epsilon-pico` |

設定ファイルを明示する場合は、コマンドの前に`--settings`を指定します。省略した項目には同梱設定の値を使います。

```bash
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar --settings config/epsilon-pico.toml help
```

## 牌譜から学習する

天鳳XMLやmjai形式の牌譜を`data/logs`に置き、最終順位を予測するGRPを学習した後、同じ保存先でDecisionを学習します。

```bash
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar pretrain-grp-logs checkpoints/epsilon-pico data/logs 1 0
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar pretrain-decision-logs checkpoints/epsilon-pico data/logs 1 0
```

末尾の`1`は牌譜全体を1周学習する指定（1エポック）、`0`は読み込む牌譜ファイル数を制限しない指定です。

この例ではDecisionのモデルが`checkpoints/epsilon-pico/decision_pretrain_logs_00001`に保存されます。実際の保存先は、完了ログの`outputCheckpoint`でも確認できます。

Decisionの牌譜学習は毎回新しいモデルから始まるため、以前の結果を残す場合は別の保存先を使います。

## 学習したモデルを評価する

同じモデルを4席に配置して、半荘を100回実行します。

```bash
java -jar epsilon-pico/target/epsilon-pico-1.0-SNAPSHOT-all.jar eval-decision checkpoints/epsilon-pico/decision_pretrain_logs_00001 100
```

指定したモデルのディレクトリに`eval-decision.txt`、その配下の`eval-tenhou-logs`に対局ログを保存します。

同じ系列の異なるモデルとの比較には`eval-vs`、系列をまたぐ比較には[Epsilon Arena](../epsilon-arena/README.md)を使います。

## 自己対戦とオンライン対局

自己対戦による追加学習には`train-decision`、RiichiLabでの対局には`play-riichi`を使います。利用できるコマンドと引数は`help`で確認できます。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
