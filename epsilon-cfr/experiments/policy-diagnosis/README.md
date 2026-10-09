# 現在方策と平均方策の診断

この文書は完了した実験の手法・結果記録です。一時的な実行・診断コードは削除済みです。以下の実行手順は、削除前のコミット `79a7530` のソースと当時の入力データを使う場合の手順です。

[pivot再利用実験](../pivot-reuse/README.md)と[凍結状態V実験](../value-baseline/README.md)で観測した性能低下を、現在方策・厳密平均・学習済み平均の比較で調べます。保存済みcheckpointと共通Nano評価を使い、診断結果は独立した出力先へ保存します。本番モデルの昇格は行いません。

## 比較する方策

補正なしの`original`と凍結V補正の`baseline`について、それぞれ`pivot`・`mixture`を評価します。

| 方策 | 内容 |
| --- | --- |
| 初期`σ₁` | Nanoから蒸留した初期平均方策 |
| `current`の`σ₂` | 初回のRegret学習256更新後。`progress/regret-1.json`を参照 |
| `exact` | 保存済み`σ₁`と`σ₂`から、その場で計算する厳密なLCFR平均 |
| `learned` | 既存の2反復試験で学習した平均ネットワーク。完了済みの対戦結果を再利用 |

2回目のRegret学習で得た`σ₃`、すなわち`regret-2`は今回の比較対象にしません。既存の平均ネットワークの教師には`σ₁`と`σ₂`までしか含まれないためです。

## 厳密平均

情報集合`I`までに本人が選んだ行動の確率積を、各snapshotについて`π₁(I)`・`π₂(I)`とします。反復重み1:2の平均は次式です。

```text
average(a | I) = [π₁(I) σ₁(a | I) + 2 π₂(I) σ₂(a | I)]
                 / [π₁(I) + 2 π₂(I)]
```

局開始時は`π₁ = π₂ = 1`なので重みは1/3・2/3です。その後は本人が選んだ行動の各snapshotでの確率を掛けます。相手や牌山の到達確率は入れません。単純に出力を常時1:2で平均する方法や、snapshotを一つ選んでgreedyにする方法とは異なります。

両snapshotには同じ本人観測・完全な局内行動履歴を渡します。選択した応答が他家の応答により実行されなかった場合も本人の選択として数え、強制行動の確率は1とします。到達確率は対数で保持し、各局の開始でリセットします。平均分布を求めてからgreedyで行動します。

両方の到達確率が0なら平均が未定義になるため、評価を失敗させます。結果の`diagnostics.exactAverageMetrics`には`allZeroReachFailures`、判断数、snapshot重みの平均・範囲、保持中の状態数などを保存します。正常完了時の`allZeroReachFailures`と`retainedOwnerStates`は0が期待値です。集計は前者が非0、または診断情報が欠落した厳密平均結果を除外します。

## 対照実験と評価条件

`initial-recheck`では初期`σ₁`を新しい評価経路で再評価し、保存済み初期評価との一致を確認します。

`initial-retained-learned`では、元の初回average streamだけを使って平均ネットワークを256更新します。初期平均checkpointと保存された元のAdamW状態から開始し、Replayの重み・抽選seed・batch 1,024・microbatch 512などを維持します。新しい対局収集はなく、`σ₂`の教師は含みません。元々表現できていた`σ₁`だけを再学習しても性能が崩れるかを確認する対照です。1世代の教師だけなので、2世代を含む元の平均学習とデータ量まで同一ではありません。

8件の現在方策・厳密平均評価と2件の対照評価を新たに実行し、既存の初期評価・4件の学習済み平均評価と比較します。各評価は元のNanoを相手に双方greedy、共通2,048牌山×4席交代で8,192半荘です。Nano checkpoint、効用profile、duel sequenceと牌山ID・seedを照合します。

主な比較は`current−initial`、`exact−initial`、`learned−exact`、`current−exact`です。平均順位は小さいほど良く、天鳳効用差は大きいほど良い指標です。信頼区間は4席をまとめた牌山単位の探索的な95%正規近似で、半荘を独立標本として扱いません。多重比較補正は行いません。

`learned−exact`は、有限の収集データ・ネットワークの表現能力・最適化を含む平均方策学習全体の差です。差があってもニューラルネット実装のバグとは断定できません。また、現在方策が弱いことだけで厳密平均の弱さは決まりません。後悔マッチング・平均化・近似・greedy選択は相互に影響するため、順位差を加算可能な原因の寄与率として分解しません。

この診断は単一学習seed・2反復時点の評価であり、確率的方策のexploitabilityや長期収束を測るものではありません。

## 実行と集計（履歴ソース用）

ビルドとネイティブ実行環境は[共通手順](../../../BUILDING.md)に従います。`prepare_diagnosis.py`には、元の2実験の`control`と`trial`を含む親ディレクトリ、および独立した出力先を渡します。

```bash
python3 epsilon-cfr/experiments/policy-diagnosis/prepare_diagnosis.py \
  /path/to/cfr-pivot-reuse /path/to/cfr-value-baseline /path/to/diagnosis
```

元の共通`CfrConfig`と実行classpathを引き継ぎ、`manifest.json`、`control/cases/*.json`、`control/plan-0.json`・`plan-1.json`を生成します。診断用Javaクラスは生成された`control/classpath.txt`を使って`classes`へコンパイルします。共通helperの変更も反映するため、対象は次の4ファイルです。

```bash
javac --release 21 -cp "$(cat /path/to/diagnosis/control/classpath.txt)" \
  -d /path/to/diagnosis/classes \
  epsilon-cfr/experiments/pivot-reuse/PivotReuseExperiment.java \
  epsilon-cfr/experiments/pivot-reuse/PivotReuseNanoDuel.java \
  epsilon-cfr/experiments/policy-diagnosis/ReachAverageDuelPolicy.java \
  epsilon-cfr/experiments/policy-diagnosis/PolicyDiagnosisDuel.java

python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py \
  /path/to/diagnosis/control/smoke-plan.json
python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py \
  /path/to/diagnosis/control/plan-0.json
python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py \
  /path/to/diagnosis/control/plan-1.json
```

smoke完了後、2つの本計画は別GPUを各processの論理GPU 0として割り当てて並行実行できます。工程の終了コードと時間は`control/stages-0`・`stages-1`に保存されます。

集計には相対配置を保持した結果一式を使います。`--original`と`--baseline`には各実験の`trial`、`--diagnosis`には今回の出力先を指定します。

```bash
python3 epsilon-cfr/experiments/policy-diagnosis/analysis.py \
  --original /path/to/cfr-pivot-reuse/trial \
  --baseline /path/to/cfr-value-baseline/trial \
  --diagnosis /path/to/diagnosis \
  --output /path/to/diagnosis/comparison.json
```

完了JSONと`*.json.walls.jsonl`の件数・評価条件・牌山集合が揃った結果だけを比較します。途中結果から平均順位や信頼区間は生成しません。`ReachAverageDuelPolicyTest`は到達確率による平均、ゼロ確率、局境界、履歴と推論予約の扱いを検証しますが、強さを保証するテストではありません。

## 実測結果（2026-10-08）

実験runは`cfr-policy-diagnosis-20261008`です。全12工程が正常終了し、10件の新規評価で計81,920半荘を実行しました。保存済みの5件を含む15評価・29比較を牌山別JSONLから再計算し、Nano checkpoint・効用profile・duel sequence・2,048牌山のIDとseedの一致を確認しました。Javaの8テスト、Pythonの7テストが通り、全4件の厳密平均で`allZeroReachFailures=0`・`retainedOwnerStates=0`でした。

**後悔学習後に後悔マッチングから作った現在方策`σ₂`の時点で、すでに大きく性能が低下していました。平均ネットワークの学習が悪化の主因だという証拠はなく、今回の実対戦では学習済み平均が厳密平均より良い結果でした。**

平均順位は小さいほど良い指標です。初期`σ₁`の平均順位は2.508057でした。

| 条件 | 現在方策`σ₂` | 厳密平均 | 学習済み平均 |
| --- | ---: | ---: | ---: |
| 補正なし・pivot | 3.930664 | 3.126587 | 2.600342 |
| 補正なし・mixture | 3.859863 | 3.151611 | 2.736694 |
| 凍結Nano V・pivot | 3.917969 | 3.008911 | 2.592041 |
| 凍結Nano V・mixture | 3.843018 | 3.105347 | 2.767090 |

学習済み平均と厳密平均の差は、対応する牌山ごとに計算しました。負の値は学習済み平均が良いことを表します。

| 条件 | 学習済み平均 − 厳密平均 | 95%信頼区間 |
| --- | ---: | --- |
| 補正なし・pivot | −0.526245 | [−0.556866, −0.495624] |
| 補正なし・mixture | −0.414917 | [−0.445804, −0.384030] |
| 凍結Nano V・pivot | −0.416870 | [−0.449428, −0.384313] |
| 凍結Nano V・mixture | −0.338257 | [−0.368877, −0.307636] |

初期方策の再評価と、`σ₁`だけを教師にした保持対照は次の結果でした。保持対照は元のaverage streamの139,260行を使い、Replay重みが元と一致し、256更新すべてのloss・勾配が有限であることを確認しました。

| 対照 | 平均順位 | 平均順位の95%信頼区間 | 初期評価との差の95%信頼区間 |
| --- | ---: | --- | --- |
| 保存済み初期評価 | 2.508057 | [2.485766, 2.530347] | — |
| 初期方策の再評価 | 2.501831 | [2.479419, 2.524243] | [−0.015573, +0.003122] |
| `σ₁`のみ256更新後 | 2.499878 | [2.477689, 2.522067] | [−0.030588, +0.014231] |

保持対照と初期再評価の対応牌山差は−0.001953、95%信頼区間は[−0.024377, +0.020471]です。初期方策を保持するだけの平均学習で性能が崩れる現象は確認できませんでした。初期再評価は個別牌山まで完全一致したわけではなく、統計的な差を検出しなかったという結果です。

この結果から、平均ネットワークの近似品質が良好だとは断定できません。弱い`σ₂`の影響を近似が弱めた可能性などは残り、この比較だけでは後悔推定・学習の問題と後悔マッチング変換そのものの問題は分離できません。後続の[後悔学習の原因診断](../regret-diagnosis/README.md)では、独立データでの回帰誤差と、凍結重みでのraw-score対照を追加しています。CFR向けに新たに学習する状態Vは未実施で、ここで比較したVは元のNanoに固定したものです。結論はgreedy評価、単一学習seed、2反復時点に限ります。

元データは`manifest.json`、`results/*.json`・`results/*.json.walls.jsonl`、`retention/pivot/reports`、`control/stages-0`・`stages-1`にあります。`comparison.json`はサーバー上の集計、`comparison.local.json`は取得後の再集計です。
