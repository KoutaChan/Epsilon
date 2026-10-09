<div align="center">

# Epsilon-CFR

**Riichi Mahjong AI**

自己対戦から行動の後悔と平均方策を学習する麻雀AI。

[![Java 21](https://img.shields.io/badge/Java-21-ED8B00?logo=openjdk&logoColor=white)](https://openjdk.org/projects/jdk/21/)
[![DJL / PyTorch](https://img.shields.io/badge/DJL-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://djl.ai/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](../LICENSE)

[はじめに](#はじめに) · [設定](#設定と保存先) · [学習](#自己対戦で学習する) · [評価](#学習したモデルを評価する) · [配布](#モデルを書き出す)

</div>

Nanoの入力表現と共通の実行基盤を使い、全4席のOutcome Sampling LCFRで学習します。後悔モデルと平均方策モデルを独立して学習し、対局には平均方策を使います。

## はじめに

ビルドと動作環境は、[プロジェクト共通の手順](../BUILDING.md)を参照してください。以下のコマンドはすべてリポジトリのルートから実行します。

利用できるコマンドと引数を確認します。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar help
```

## 設定と保存先

初回起動時に`config/epsilon-cfr.toml`が作成されます。使用するGPU、学習率、バッチサイズなどをこのファイルで変更できます。各項目の説明と既定値は[同梱設定](src/main/resources/cfr/settings.toml)にあります。

設定ファイルを明示する場合は、コマンドの前に`--settings`を指定します。省略した項目には同梱設定の値を使います。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar --settings config/epsilon-cfr.toml help
```

別の場所へ設定の雛形を作る場合は、`init-config --out config/my-cfr.toml`を使います。既存のファイルは上書きしません。

| 用途 | この例で使う場所 |
| --- | --- |
| 設定ファイル | `config/epsilon-cfr.toml` |
| 学習結果と再開用データ | `checkpoints/epsilon-cfr/run-001` |
| 学習・評価レポート | `checkpoints/epsilon-cfr/run-001/reports` |
| 配布用モデル | `models/epsilon-cfr` |

保存先はコマンドの引数で指定します。学習開始時に、適用済みの設定をrunディレクトリ内の`config.json`へ保存します。これは再開用の保存データです。`resume`はこの設定を使い、起動時のTOMLや上書き値で学習条件を変更しません。

学習目標の順位効用はcommonの定義を使います。`[epsilon.cfr]`の`utilityProfile`を変更すると、半荘終端・GRPの期待効用・分岐比較・昇格評価に同じ効用を適用します。既定は`TENHOU`です。

```toml
[epsilon.cfr]
utilityProfile = "PLACEMENT"
```

| 値 | 1着・2着・3着・4着の効用 |
| --- | --- |
| `TENHOU` | 2/3、1/3、0、−1 |
| `PLACEMENT` | 1.5、0.5、−0.5、−1.5 |
| `TOP` / `MLEAGUE` | 1、0.2、−0.2、−0.6 |
| `LAST_AVOIDANCE` | 0.4、0.2、0、−1 |

Nano・Major・Picoも同じ効用定義を使い、各系列では既存の`[epsilon.decision]`の`utilityProfile`で選択します。CFRのGRPは順位分布を予測するため、得点用の`SCORE`は指定できません。効用はrun・モデル・配布物に保存されます。変更する場合は新しいrunで学習を始めてください。

学習と推論のデバイスはNanoと同じ共通機能で選択します。`[epsilon.cfr.network.devices]`の`learner`と`inference`が空文字列なら利用可能な全GPUを選び、GPUがなければCPUを使います。明示指定は`"cpu"`、`"gpu:0"`、`"gpu:0,gpu:1"`などです。GRP用の`grp`は既定で`"cpu"`です。

設定の選択は、先頭の`--settings`、`-Depsilon.settings.path`、環境変数`EPSILON_SETTINGS`、既定パスの順です。個別の値は`java -Depsilon.cfr.network.devices.learner=gpu:0 -jar ...`のように上書きできます。

共通部分の既定値はNanoに合わせています。hiddenは256、学習率はNanoの教師学習を基準とした`2e-4`、weight decayは`1e-4`、勾配clipは`0.5`です。GPUの計算精度設定はBF16、CPUはFP32で、損失・勾配・学習パラメータ・optimizer状態はFP32を維持します。転送はpinned bufferです。1反復に32,768半荘を収集し、同時収集数と推論batch上限は4,096です。既定の`HAND_GRP`では局ごとに観測を保存先へ渡して解放し、`collection.concurrentGames`で同時進行の半荘数を制限します。初期局面・牌山・行動抽選の乱数はエピソードIDから決まるため、同時収集数を変えても収集予算やサンプリングの定義は変わりません。CPUで小規模な動作確認をする場合は、収集数、同時収集数、バッチサイズ、更新数、評価牌山数も小さくしてください。

`collection.games`は、どちらの方式でも実際に最後まで進める半荘数です。連荘や飛びで局数が変わっても、32,768半荘の収集予算は変わりません。これとは別に、`learning.batchSize`は1回の学習更新でreplayから抽選する判断数で、既定は1,024です。`learning.microBatchSize`は一度に計算する行数の上限で、既定は1,024です。重複の集約と候補形状による分割後は、この上限より少なくなる場合があります。複数GPUでは学習更新の抽選数をデバイス数で分けた上限も適用し、計算が1台へ偏るのを抑えます。

## 自己対戦で学習する

設定ファイルと新しい保存先を指定して学習を始めます。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar --settings config/epsilon-cfr.toml train --run checkpoints/epsilon-cfr/run-001
```

大量の観測を保存する収集では、JavaのGCによる停止も速度に影響します。Linux・Java 21での性能測定には、次のGenerational ZGCの起動条件を使用します。ヒープ上限は利用可能なRAMに合わせて調整してください。`resume`にも同じJVMオプションを指定できます。

```bash
java -XX:+UseZGC -XX:+ZGenerational -Xms1g -Xmx20g -XX:MaxDirectMemorySize=6g \
  -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar \
  --settings config/epsilon-cfr.toml train --run checkpoints/epsilon-cfr/run-001
```

`[epsilon.cfr.collection]`の`mode`で教師値の評価境界を選びます。どちらも東1局から半荘終了まで収集します。以下では設定キーの`epsilon.cfr.`を省略して記載します。

| モード | 学習方法 |
| --- | --- |
| `HAND_GRP`（既定） | 局ごとに学習経路を区切り、局末は固定GRP、半荘終局は実際の最終順位で評価する |
| `MATCH_MC` | 半荘を最後まで収集し、実際の最終順位を使う |

`HAND_GRP`では、GRPを指定していない場合、初回に一様確率で行動する方策の対局からGRPを学習します。`solves`で複数の学習区間を指定した場合は、区間の切り替え時に採用済みの平均方策でGRPを更新します。区間内ではGRPを固定します。各局の開始状態は、その反復の方策が半荘中に到達した状態を使います。以前の固定RootBankから独立した局を抽選する方式とは、局開始分布が異なります。局をまたいで重要度比を掛け合わせず、教師の重みは反復内の実局数で正規化します。

既定では、RONの見逃しと九種九牌に限って二択の分岐比較を行います。RONは見逃しを選んだ判断だけ和了側と比較し、九種九牌は宣言・続行のどちらを選んだ場合も反対側と比較します。追加枝も通常の収集と同じ境界まで進め、`HAND_GRP`なら局末の固定GRP、`MATCH_MC`なら半荘最終順位で評価します。追加枝はその判断の後悔教師に使い、通常の32,768半荘や平均方策の教師には含めません。

設定は`[epsilon.cfr.collection.branchComparison]`です。既定値は`enabled = true`、`winDecline = true`、`kyushu = true`です。行動を抽選する前に比較を予約し、`maxBranchesPerGame = 1`で半荘あたりの予約を1回に制限します。予約後にRONを選んだ場合や、九種九牌の比較対象外であるツモ和了を選んだ場合も、この回数に含みます。`maxInFlightBranches = 8`は同時に保持できる予約枠で、追加枝が不要なら行動確定時、追加枝を進める場合はその終了時に解放します。

`maxExtraInferenceRatio = 0.01`は、通常の推論行数に対する追加推論行数を使った開始予算です。厳密な完了時上限ではありません。開始済みの枝は途中で打ち切らず同じ評価境界まで進めるため、一時的に予算を超えることがあります。`enabled = false`で分岐比較を無効にできます。

学習済みNanoから始める場合は、設定の`initialization.nanoCheckpoint`を指定します。`network.hidden`はNanoの保存モデルと同じ値にしてください。

```toml
[epsilon.cfr.initialization]
nanoCheckpoint = "checkpoints/epsilon-nano"
games = 256
updates = 2048
capacity = 65536
```

`nanoCheckpoint`にはNanoのモデルルート、Decision管理ディレクトリ、または単体checkpointを指定します。空文字列ならNanoからの初期化を行いません。`games`は教師の収集半荘数、`updates`は平均方策の初期蒸留更新数、`capacity`は学習用に保持する判断数です。検証用には半荘単位で5分の1を分離し、最大4,096判断を保持します。双方に`replay.maxStoredBytes`の保存量上限を適用します。

Nanoと共通の表現を引き継ぎ、Nanoの全合法手確率を教師として平均方策を蒸留します。方策の出力構造が異なるため、変換だけでNanoの対局性能を維持するとは限りません。初期化前後の検証結果は`reports/initialization-before.json`と`reports/initialization-heldout.json`に保存し、KL、交差エントロピー、最上位手の一致率を確認できます。

初期化後の最初の収集には蒸留した平均方策を使い、その後はLCFRの後悔学習へ移ります。教師データはLCFRの再生データへ混ぜず、Nanoの模倣は初期化時だけ行います。NanoのGRPが同じモデルルートにある場合はそれを利用でき、`boundary.checkpoint`で別のGRPを指定することもできます。

変換元と境界教師はrun内へ固定保存します。runの`config.json`保存後は、元のNanoやGRPディレクトリがなくても`resume`で初期化から再開できます。教師収集は半荘ごと、蒸留は`learning.checkpointEveryUpdates`ごとに進捗を保存します。

現在の状態を確認するには`status`、保存された状態から学習を続けるには`resume`を使います。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar status --run checkpoints/epsilon-cfr/run-001
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar resume --run checkpoints/epsilon-cfr/run-001
```

再開時はモデル、optimizer、再生データを保存済みの工程から復元します。保存物と保持中の再生データの整合性は、学習を停止した状態で`verify`から確認できます。

保存済みrunの設定JSONと学習データは現行形式だけを読み込みます。半荘単位の収集へ変更したため、以前の局単位収集のrunは再開できません。設定は`collection.games`で指定し、旧設定の自動補完・変換は行いません。局データは順次保存し、半荘が最後まで完了したファイルだけを再開時に再利用します。

モデルの入力はNanoと同じ盤面情報です。CFR専用の行動履歴・局境界系列・GRUは使いません。旧履歴モデルとは構造と観測の保存形式が異なるため、以前のCFRモデルやrunは再開できません。Nanoから初期化するか、新しいrunで学習してください。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar verify --run checkpoints/epsilon-cfr/run-001
```

## 学習したモデルを評価する

学習中は`evaluation.every`で指定した間隔と、各学習区間の最終反復で評価します。既定の間隔はNanoの学習区間に合わせて10反復です。後悔モデルは毎反復学習し、平均方策は評価する反復だけ、比較の直前に学習します。平均方策の教師と平均化の重みは毎反復replayへ蓄積します。例えば最終反復が128なら、10、20、…、120、128で平均方策を学習・評価します。

平均方策の学習には、それまでにreplayへ保持した各反復の教師を使います。Nanoからの初期蒸留は、自己対戦を始める前に別途実行します。

Nanoと同じ共通duelを使い、候補の平均方策1席と採用済み平均方策3席を対戦させます。各牌山で候補席を4通りに入れ替え、4半荘を1つの独立標本として集計します。評価では双方とも最大確率の合法手を選びます。

昇格判定もNanoと同じ2段階の計画を使います。既定では50,000牌山で途中判定し、保留なら100,000牌山まで継続します。実対局数はそれぞれ200,000・400,000半荘です。両側有意水準の合計は`0.01`で、その25%を途中判定、残りを最終判定へ割り当てます。

設定した順位効用の「候補−相手3席平均」の信頼区間下限が0を超えれば`PROMOTED`として`champion`を更新します。上限が0未満なら`HARMFUL`、それ以外は`UNRESOLVED`です。後者2つはchampionを更新せず、LCFRの累積学習を継続します。平均順位・トップ率・ラス率・点差と判定結果はrun配下の`reports`に保存します。

評価予算は`evaluation.championDuel`、並列対局数・推論batch・進捗間隔は`evaluation.evalVs`、CPU並列度は`evaluation.arena`で設定します。既定値はNanoと同じ、同時32,768対局、推論batch上限2,048、CPU worker数32です。

異なる系列のモデルと対戦させる場合は、次の手順でモデルを書き出し、[Epsilon Arena](../epsilon-arena/README.md)を使います。

## モデルを書き出す

採用済みの`champion`を配布用ディレクトリへ書き出します。出力先には新しいディレクトリを指定してください。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar export --run checkpoints/epsilon-cfr/run-001 --out models/epsilon-cfr
```

現在の学習済み平均方策を使う場合は、`--selection average`を指定します。

```bash
java -jar epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar export --run checkpoints/epsilon-cfr/run-001 --out models/epsilon-cfr-average --selection average
```

配布物には推論用のモデルと構成情報を保存し、optimizerは含めません。Arenaでは`series`を`epsilon-cfr`、`checkpoint`をこの出力先に指定します。CFRの実行デバイスには`cpu`または`gpu:0`などを明示してください。

[Epsilon AI レビュアー](../epsilon-reviewer/README.md)も同じ配布形式を読み込みます。解析にはNanoと同じ盤面情報を使い、CFR専用の行動履歴やGRPは必要ありません。`HAND_GRP`と`MATCH_MC`は学習時の評価境界を選ぶ設定で、どちらのモデルも同じ方法で推論します。

実装構成、保存形式、CLIの出力仕様、テスト方法は[開発者向けドキュメント](DEVELOPMENT.md)を参照してください。

<div align="center">

[シリーズ一覧](../README.md) / [ライセンス](../LICENSE) / [第三者ライセンス](../LICENSES/README.md)

</div>
