# Pivot方策の再利用実験

この文書は完了した実験の手法・結果記録です。一時的な実行・診断コードは削除済みです。以下の実行手順は、削除前のコミット `79a7530` のソースと当時の入力データを使う場合の手順です。

同じ対局から、抽選された1判断だけを使う`pivot`と、pivot候補の事後確率で対象席の各判断を使う`mixture`を比較します。Qモデルを追加せず、既存のCollector、履歴モデル、Replay、Learner、共通duelを使います。実験結果は独立したディレクトリへ保存し、元の学習runやchampionを更新しません。

## 比較条件

1反復は32,768局で、Regret用24,576局と平均方策用8,192局に分けます。対象席は均等に割り当てます。初回はNanoから蒸留した初期平均方策を`σ₁`として一度だけ収集し、同じ軌跡から両armの教師を作ります。Regret learnerは、両armとも同じ初期後悔モデルとoptimizerから開始します。

参照方策`ν`は初期平均方策に一様探索を10%混ぜたものとして固定します。pivotは対象席の非強制判断ごとに確率`h=0.0625`で選びます。Regret用の対局では対象席がpivotまで`ν`、pivotで合法手の一様分布、その後は`σ`を使い、他席は`σ`を使います。平均方策用の対局では対象席が`σ`、他席が`ν`を使います。

2反復の試行では、初回のRegret学習後のモデルを各armの`σ₂`として収集します。平均方策は最終反復の2で学習します。試行当時の通常設定`averageEvery=4`を保持したまま、最終反復では更新する扱いに合わせました（現行版は評価時・最終反復だけ更新します）。この平均方策の教師は`σ₁`と`σ₂`です。2回目のRegret学習で得る`σ₃`は、まだ平均方策の教師に含みません。

Regretと平均方策のReplayはarmごとに分離します。大量のRegret行が平均方策の保持容量を消費することを避け、両roleで既存の一様優先度reservoirと重み付き抽選を使います。両roleの重みが0の行は保存せず、生成数と除外数を計測します。

## 設定とビルド（履歴ソース用）

ビルド・ネイティブ実行環境は[共通手順](../../../BUILDING.md)に従います。以下はリポジトリのルートから実行するLinuxの例です。実際のGPU実行では、既存の学習と同じJNI、PyTorch、デバイス環境を設定してください。

```bash
mvn -pl epsilon-cfr -am -DskipTests package
javac --release 21 \
  -cp epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar \
  -d epsilon-cfr/target/pivot-reuse-classes \
  epsilon-cfr/experiments/pivot-reuse/PivotReuseExperiment.java \
  epsilon-cfr/experiments/pivot-reuse/PivotReuseNanoDuel.java
export PIVOT_CP="epsilon-cfr/target/pivot-reuse-classes:epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar"
```

`settings.example.json`をコピーして入力・出力先を指定します。`initialRegret`と`initialAverage`にはoptimizerを含む学習checkpointを指定し、`referenceAverage`には同じ初期平均方策を指定します。`outputDirectory`は入力と重ならない新しい場所にします。

| 設定 | 用途 |
| --- | --- |
| `baselineConfig` | 共通SettingsLoaderから生成された既存runの`config.json`。学習・デバイス・並列度・Replay・評価設定を引き継ぐ |
| `initialRegret` / `initialAverage` | 両arm共通の学習開始点 |
| `referenceAverage` | 試行中に固定する参照方策の元モデル |
| `rootBank` / `boundaryCheckpoint` | 元runの固定局開始分布とGRP |
| `regretEpisodes` / `averageEpisodes` | 反復ごとの局数。既定例は24,576／8,192 |
| `trialIterations` | 試行反復数。既定例は2 |
| `maxStoredBytesPerArm` | armごとのReplay保存量上限。例は8 GiBで、両roleに半分ずつ割り当てる |
| `evaluationWallSeeds` | 初期CFR平均方策との試験duelの牌山数。本番の昇格評価予算とは別 |

学習条件は既存の`CfrConfig`を使い、実験JSONには試行固有の入力と配分を指定します。Nanoに合わせた本試行はbatch 1,024、microbatch 512、各role 256更新です。読み込んだモデルの学習率・weight decay・clip・microbatch・精度・転送方法が保存設定と異なる場合、学習を開始しません。初回に実験設定と適用済みの`CfrConfig`を出力先へ固定保存します。

## 収集・学習・再開（履歴ソース用）

```bash
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json collect 1 shared
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json learn 1 pivot regret
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json learn 1 mixture regret

for arm in pivot mixture; do
  java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json collect 2 "$arm"
  java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json learn 2 "$arm" regret
  java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json learn 2 "$arm" average
done
```

同じコマンドを再実行すると、保存済みの収集行、Replay世代、学習checkpointから再開します。収集行は既存バイナリ形式をZstdで圧縮します。更新数の保存間隔は`learning.checkpointEveryUpdates`を使います。

Linuxの`run_stages.py`には、`directory`と`stages`を持つ計画JSONを渡せます。各stageは`name`と、上記コマンドを文字列の配列にした`command`を持ちます。成功した工程をスキップし、工程別ログ、終了コード、`current-stage.json`、全工程完了時の`complete.json`を保存します。

```bash
python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py plan.json
```

## 評価

初期Nano蒸留CFR平均方策との比較は、runnerの`evaluate`で実行できます。

```bash
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json evaluate 2 pivot
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" PivotReuseExperiment experiment.json evaluate 2 mixture
```

元のNanoそのものとの比較には`PivotReuseNanoDuel`を使います。候補には平均方策checkpointのディレクトリ、またはrunnerが保存した`progress/average-2.json`を指定できます。次の例の牌山数1,024は試行用です。

```bash
java --enable-native-access=ALL-UNNAMED -cp "$PIVOT_CP" \
  com.epsilon.cfr.evaluation.PivotReuseNanoDuel \
  /path/to/experiment/baseline-config.json \
  /path/to/experiment/mixture/progress/average-2.json \
  /path/to/original-nano 1024 /path/to/experiment/mixture-nano.json
```

Nanoと共通のgreedy選択、同じ牌山、4席交代で評価します。両armに同じ設定と牌山数を渡すと、保存された`*.walls.jsonl`の同じ牌山IDを対応付けてarm間の差と信頼区間を計算できます。このhelperの固定予算評価は95%信頼区間を使い、本番の2段階昇格判定とは別です。評価結果による自動昇格は行いません。

## 計測結果の読み方

`collections`には収集数、非強制判断数、正の重みの行数、深さ別の行数とposterior質量を保存します。再開した収集の`countsThisInvocation`は、その呼び出しで追加収集した分です。`pivot`の1局の正規化重みは0または1、`mixture`の合計は1以下です。到達しなかったpivotの質量を残し、局ごとに合計1へ再正規化しません。

各armの`reports`には、保持後のESS・最大抽選確率、loss、gradient norm、適用済みモデル設定、学習速度を保存します。`logicalSamplesPerSecondIncludingCheckpoints`は抽選件数換算、`distinctDrawnSamplesPerSecondIncludingCheckpoints`は各batch内で同じ`CfrSampleId`をまとめた件数換算です。後者も別batch間の重複や同一局の相関を除いた独立標本数ではありません。主要な学習速度にはsamplingとGPU同期を伴うcheckpoint保存を含めます。

元の相対パスを保って保存した実験結果は、`analyze_results.py`で`comparison.json`へ集約できます。対戦評価は完了JSONと牌山別JSONLの件数を確認し、初期モデルとの差と`mixture−pivot`の対応する牌山ごとの差を集計します。信頼区間は探索的な95%正規近似で、多重比較補正は行いません。

ダウンロードした結果の`trial`を指定した場合は、兄弟ディレクトリの`control/stages`も読み、smokeの工程時間は除きます。`reservoirAndStorage`は新規にReplayへ渡した行、過去世代を含む保持行、保持容量への到達、Replayの保存量を分けて示します。容量が埋まると新しい行が古い行を置き換えるため、保持行数の増加を新規行数とは扱いません。保存量の数値はReplayのみで、収集行ファイルやモデルは含みません。`normalizedWeightMassPerEpisode`はRegretではposterior質量、平均方策では対象席の各判断の重み合計です。

```bash
python3 epsilon-cfr/experiments/pivot-reuse/analyze_results.py /path/to/experiment
```

収集速度には圧縮行保存を含め、モデル読込とReplay登録の時間は別です。両armのCollectorは全候補のmetadataを取る同じ推論wrapperを使います。この比較で測れるのは教師再利用の効果であり、不要な参照推論を省いた単一pivot専用実装との速度差ではありません。工程全体の所要時間はcontrollerの開始・終了記録で確認します。

有限ゲームの全列挙テストは、posterior重み、未到達component、期待後悔、局ごとの重み上限などの計算を検証します。これは共有ニューラルネットでの学習改善や麻雀の強さを保証するものではありません。正の行数・ESS・学習lossの改善と、同じ予算での対戦評価を分けて判断します。

## 実測結果（2026-10-08）

実験runは`cfr-pivot-reuse-20261008`です。各armで32,768局×2反復を行いました。初回の収集を共用したため、学習用の実収集は計98,304局です。各学習工程はbatch 1,024で256更新し、初期モデルを含むNano評価まで全12工程が正常終了しました。

**この試行では両案とも初期モデルより弱くなり、複数判断再利用の`mixture`は単一判断の`pivot`より悪化しました。採用に進める結果ではありません。** 元のNanoを相手に同じ2,048牌山・4席交代、各候補8,192半荘で評価しました。全候補で牌山IDとseedの一致を確認しています。

| 候補 | 平均順位（小さいほど良い） | Nanoに対する天鳳効用差 |
| --- | ---: | ---: |
| 初期Nano蒸留平均方策 | 2.508057 | −0.003581 |
| 2反復後の`pivot`平均方策 | 2.600342 | −0.075629 |
| 2反復後の`mixture`平均方策 | 2.736694 | −0.171495 |

同じ牌山で対応付けた`mixture−pivot`の平均順位差は**+0.136353、95% CI [+0.104847, +0.167858]**、天鳳効用差は**−0.095866、95% CI [−0.120648, −0.071084]**でした。いずれもこの試行では`mixture`が不利です。CIは4席交代をまとめた牌山を単位とする探索的正規近似です。

| 学習・計算量 | `pivot` | `mixture` |
| --- | ---: | ---: |
| 最終Regretの保持行数 | 32,483 | 716,156 |
| 最終Regretの重みESS | 29,428.74 | 57,256.65 |
| 最終Regretの最大抽選確率 | 0.00398367% | 0.00394698% |
| 反復2のCollector、圧縮行保存込み | 123.12局/s | 119.68局/s |
| 反復1のRegret learner、論理件数/s（batch内重複除外件数/s） | 2,542（2,457） | 2,511（2,475） |
| 反復2のRegret learner、同上 | 2,266（2,227） | 2,364（2,343） |
| 反復2のAverage learner、同上 | 1,659（1,656） | 1,676（1,672） |
| 反復2の読込・収集・Replay登録・両learnerの工程全体 | 561.42秒 | 592.75秒 |
| 最終Replay保存量、Regret＋Average | 0.583 GB | 1.962 GB |

初回の同一データでは重みESSが14,761から37,214.15へ2.52倍になりました。最終ReplayでもESSは約1.95倍ですが、独立した対局情報が同じ割合で増えたとは言えません。両案とも保持容量に達しておらず、今回の正の重みの行は全件保持されています。重みの集中は抑えられた一方、対戦成績は改善しませんでした。

反復2の工程全体は`mixture`が**5.6%長く**、主な増分はReplay登録の14.46秒→40.83秒でした。learner速度はcheckpoint保存とGPU同期を含みます。Collectorは両案共通の全metadata取得wrapperで、単一pivot専用に推論を削った実装との比較ではありません。保存量は十進GBで、圧縮収集行ファイルとモデルcheckpointを除いたReplayのみです。

これは学習seedが1つ、2反復だけの試行です。評価した平均方策の教師には`σ₁`と`σ₂`が入り、最後のRegret更新で得た`σ₃`は入りません。長期の優劣や悪化の原因は、この結果だけでは確定しません。今回の設定のまま性能改善を主張したり、本番へ採用したりする根拠にはなりません。

再集計に使う保存物は、run内の`trial/comparison.json`、`trial/*/reports/iteration-*-*-learning.json`、`trial/*/reports/iteration-*-*-replay.json`、`trial/results/nano-{initial,pivot,mixture}.json`と対応する`.walls.jsonl`、`control/stages/*.result.json`です。
