<div align="center">

# Epsilon-Prism

**Riichi Mahjong AI**

人の牌譜を使わず、自己対戦だけで打ち方を育てる麻雀AI。四人打ちと三人打ちに対応。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![Parameters: 1.94M](https://img.shields.io/badge/Parameters-1.94M-7c3aed)](../README.md#epsilon-シリーズ)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

</div>

## ビルド

[共通の手順](../BUILDING.md)でビルドし、リポジトリのルートで実行します。AMDのGPUを使うときは、`-Pnative-hip`を付けてビルドします。

## 学習

```bash
java -jar epsilon-prism-4p/target/epsilon-prism-4p-1.0-SNAPSHOT-all.jar settings create --run runs/prism-4p --set run.variant=TENHOU_4P --set run.seed=1
java -jar epsilon-prism-4p/target/epsilon-prism-4p-1.0-SNAPSHOT-all.jar train --run runs/prism-4p
```

学習したバージョンは`runs/prism-4p/snapshots`に保存されます。止めても、同じコマンドで再開できます。三人打ちは`epsilon-prism-3p`のJARと`TENHOU_3P`を使います。

## 対局

RiichiLabのランク戦は`play-riichi`、天鳳の個室は`play-tenhou`で打てます。牌譜の解析は[Epsilon AI レビュアー](../epsilon-reviewer/README.md)を使います。

## 詳しく

コマンドの一覧は`help`、設定の一覧は`settings template`で見られます。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
