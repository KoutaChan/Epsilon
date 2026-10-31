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
> 近日中に、Epsilon-Prism (PRISM: Posterior-start Roll-in Information-Set Matching) の重みを公開する予定です。 

## 概要

Epsilonは、JavaとDJL / PyTorchを使った4人リーチ麻雀AIです。牌譜からの学習と自己対戦による追加学習に対応したEpsilon-Pico・Epsilon-Nano・Epsilonと、自己対戦だけで育てるEpsilon-Prismの4系列で、学習・評価・対局を実行できます。

## Epsilon シリーズ

| モデル | パラメーター数 |
| --- | ---: |
| [Epsilon-Pico](epsilon-pico/README.md) | **1,507,396**（約1.51M） |
| [Epsilon-Nano](epsilon-nano/README.md) | **4,725,700**（約4.73M） |
| [Epsilon](epsilon/README.md) | **10,699,636**（約10.70M） |
| [Epsilon-Prism](epsilon-prism/README.md) | **1,939,604**（約1.94M） |

- [Epsilon Arena](epsilon-arena/README.md)：異なる系列のモデルを対戦させ、成績を比較できます。
- [Epsilon AI レビュアー](epsilon-reviewer/README.md)：牌譜をAIで解析し、行動の候補を確認できます。

## ライセンス

[GNU General Public License v3.0](LICENSE)で公開しています。Nyanten・ShantenNumberなど、第三者のコードとライセンスについては[LICENSES](LICENSES/README.md)を参照してください。

<div align="center">

[Epsilon-Pico](epsilon-pico/README.md) · [Epsilon-Nano](epsilon-nano/README.md) · [Epsilon](epsilon/README.md) · [Epsilon-Prism](epsilon-prism/README.md)

</div>
