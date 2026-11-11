<div align="center">

# Epsilon

**Riichi Mahjong AI**

牌譜からの学習、AI同士の対局、RiichiLabでの対戦、牌譜解析に対応。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

[概要](#概要) · [シリーズ](#epsilon-シリーズ) · [ライセンス](#ライセンス)

</div>

> [!IMPORTANT]
> Epsilon-Picoは削除されました。 Epsilon-Nano, Epsilon は最小限のメンテナンスで維持されます。

## 概要

Epsilonは、JavaとDJL / PyTorchを使ったリーチ麻雀AIです。牌譜からの学習と自己対戦による追加学習に対応したEpsilon-Nano・Epsilonと、自己対戦だけで育てる四人打ち・三人打ちのEpsilon-Prismの3系列で、学習・評価・対局を実行できます。

## Epsilon シリーズ

| モデル                                           |                 パラメーター数 |
|-----------------------------------------------|------------------------:|
| Epsilon-Nano (Outdated)                       |   **4,725,700**（約4.73M） |
| Epsilon (Outdated)                            | **10,699,636**（約10.70M） |
| [Epsilon-Prism (v2)](epsilon-prism/README.md) |   **1,939,604**（約1.94M） |

- [Epsilon AI レビュアー](epsilon-reviewer/README.md)：牌譜をAIで解析し、行動の候補を確認できます。

## ライセンス

[GNU General Public License v3.0](LICENSE)で公開しています。Nyanten・ShantenNumberなど、第三者のコードとライセンスについては[LICENSES](LICENSES/README.md)を参照してください。

<div align="center">

[Epsilon-Prism](epsilon-prism/README.md) · [Epsilon AI レビュアー](epsilon-reviewer/README.md)

</div>
