<div align="center">

# Epsilon-Prism

**Riichi Mahjong AI**

人の牌譜を使わず、白紙の網から自己対戦だけで打ち方を育てる麻雀AI。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Parameters: 1.94M](https://img.shields.io/badge/Parameters-1.94M-7c3aed)](../README.md#epsilon-シリーズ)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [設定](#設定と保存先) · [学習](#自己対戦で学習する) · [評価](#学習したモデルを評価する) · [オンライン対局](#オンライン対局と牌譜解析)

</div>

> [!IMPORTANT]
> 学習済みの重みは同梱していません。近日中に公開する予定です。

**1,939,604パラメーター（約1.94M）** です。方策と補助の頭を含み、局の終わりの評価に使うGRPは含みません。

## はじめに

ビルドと動作環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。以下のコマンドはすべてリポジトリのルートから実行します。

利用できるコマンドと引数を確認します。

```bash
java -jar epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar help
```

### HIPの場合

HIPでの推論と自己対戦の収集には、自前のHIPの実行系を使います。LinuxのROCm環境で`-Pnative-hip`を付けて組み立てると、`libprism_hip.so`がJARに入ります。

```bash
./mvnw -Procm -Dpytorch.rocm.flavor=rocm7.2 -Pnative-hip package
```

CPUだけで動かす場合は、設定の`inference`を`"EAGER"`にします（DJLで推論します）。

## 設定と保存先

初回起動時に`config/epsilon-prism.toml`が作成されます。収集と学習に使うGPU、1反復の半荘の数、学習率などをこのファイルで変更できます。各項目の説明は[同梱設定](src/main/resources/prism/settings.toml)にあります。

| 用途            | 既定の場所                                      |
|---------------|--------------------------------------------|
| 設定ファイル        | `config/epsilon-prism.toml`                |
| 学習のrun        | `train`の`--run`で指定（例：`runs/epsilon-prism`） |
| 学習結果（版）       | `<run>/snapshots/<版>`                      |
| RiichiLabで打つ版 | `[epsilon.riichi]`の`checkpointDir`         |

設定ファイルを明示する場合は、コマンドの前に`--settings`を指定します。省略した項目には同梱設定の値を使います。

```bash
java -jar epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar --settings config/epsilon-prism.toml help
```

## 自己対戦で学習する

PRISMは打ち方を人の牌譜から学びません。反復ごとに、今の4席すべてで打つ自己対戦を集めます。選んだ判断では、候補の手をすべて同じ牌山で局の終わりまで打ち分け、局の終わりにGRPが見積もる順位の効用の差を、その手のラベルにします。ラベルの運の揺れは、ツモなどの出来事ごとの補正（運の精算）で減らします。生徒は前の重みを錨にしたTEMPO（それまでの全てのラベルの再帰平均）で学び、次の重みになります。

```bash
java -jar epsilon-nano/target/epsilon-nano-1.0-SNAPSHOT-all.jar pretrain-grp-logs checkpoints/epsilon-nano data/logs 1 0
```

```toml
[epsilon.prism]
grpCheckpoint = "checkpoints/epsilon-nano"
```

白紙の重みから学習を始めます。

```bash
java -jar epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar train --run runs/epsilon-prism
```

反復ごとに次の重みを`runs/epsilon-prism/snapshots/<重み>`に書き、`reports/iterations.jsonl`に記録を1行足します。各重みは規則の打ち手と打たせて測り、`reports/evaluations.jsonl`に書きます。`--stop-after N`を付けるとN反復で止まり、同じ`--run`を指定すると続きから再開します。新しいrunの設定は`config.json`に書かれ、再開ではそれを読みます。

既定値は、GPU2枚で収集（`collectDevices`）と学習（`learnerDevice`）を行う前提です。環境に合わせて変更してください。

## 学習したモデルを評価する

PRISMを1席、相手を3席に置いて打たせ、PRISMの着順を測ります。Epsilon-Nanoの3席と打たせる例です。

```bash
java -jar epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar evaluate --snapshot runs/epsilon-prism/snapshots/00160 --opponent nano:checkpoints/epsilon-nano/decision_pretrain_logs_00001 --games 131072 --per-game x1n3.tsv
```

`--opponent heuristic`にすると、規則の打ち手3席と打たせます（学習中の評価と同じ卓）。`--per-game`は半荘ごとの結果（半荘ID・着順・効用）を書きます。同じ`seed`と半荘の数なら、版を替えても同じ牌山で比べられます。

## オンライン対局と牌譜解析

RiichiLabのランク戦には`play-riichi`を使います。自分の設定ファイルの`[epsilon.riichi]`に、接続のtokenと打つ重みを書きます。`checkpointDir`には重みのディレクトリか、runのディレクトリ（その最新の重み）を指定します。

```toml
[epsilon.riichi]
token = "<RiichiLabのtoken>"
checkpointDir = "runs/epsilon-prism/snapshots/00160"
```

```bash
java -jar epsilon-prism/target/epsilon-prism-1.0-SNAPSHOT-all.jar play-riichi
```

`maxGames`で完了する対局数の上限を決めます（0なら止めるまで続けます）。

牌譜の解析では、[Epsilon AI レビュアー](../epsilon-reviewer/README.md)でEpsilon-Prismを選べます。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
