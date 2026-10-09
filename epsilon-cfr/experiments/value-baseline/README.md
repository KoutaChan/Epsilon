# 凍結状態V baselineの比較実験

この文書は完了した実験の手法・結果記録です。一時的な実行・診断コードは削除済みです。以下の実行手順は、削除前のコミット `79a7530` のソースと当時の入力データを使う場合の手順です。

Nanoの学習済み状態Vを固定し、[pivot再利用実験](../pivot-reuse/README.md)の後悔教師から引く単一要因比較です。通常の学習workflowの既定値は変更しません。

## 教師と独立性

局開始時のGRP効用を `root`、収集終了時の効用を `leaf` とすると、既存の中心化済み効用は `Y = leaf − root` です。今回のbaselineは `b(I) = V_Nano(I) − root`、新しい効用は `Y − b(I) = leaf − V_Nano(I)` です。Nano Vは101-binの分布からTENHOU効用にdecodeします。

合法行動数を `n`、一様pivotで選んだ行動を `A`、目標方策を `σ` として、教師は `d_a = (Y − b(I)) (1[a=A] − σ(A))` です。行動前に固定されたscalar baselineについて、一様抽選した `A` に関するbaseline項の期待値は0です。mixtureのposterior重みは変更しません。

- VとGRPは今回の収集以前のcheckpointに固定し、追加学習しません。
- Vの入力は保存済みの行動前の本人観測と公開された局境界です。選択行動・終局結果・非公開情報は渡しません。
- 保存済みNano入力のGRP priorを公開prefixから復元し、状態V専用の順伝播を使います。候補別Qや環境の追加分岐はありません。
- Replay行のID・観測・目標方策・重みを維持し、regret行のutilityだけを残差へ変換します。平均方策行は変換しません。
- 元の圧縮行を別ディレクトリに保存し、再開時に二重にbaselineを引かないようにします。

## 比較条件

各armは32,768局/iteration（regret 24,576局、average 8,192局）、2 iterationです。pivot、mixtureの両方に同じ補正を適用し、各々の補正なしの結果と比較します。

1 iteration目は前回の同一の保存済み対局を使用します。初期checkpoint、Replay容量・抽選seed、batch 1,024、microbatch 512、256更新、学習率 `2e-4`、BF16を維持します。2 iteration目は各armの学習後の方策で新たに収集します。最終averageはσ1とσ2の教師から学習します。2回目のregret更新によるσ3はこのaverageには含まれません。

評価は元のNanoを相手に双方greedy、共通2,048牌山×4席交代＝8,192半荘/候補です。前回と同じseedを使い、候補間の差は牌山単位でpaired集計します。単一の学習seedによる探索的比較で、本番昇格は行いません。

測定項目は次のとおりです。

- 同じ行・重みに対する教師二乗ノルム、utilityと残差の二乗平均、baselineとの相関
- 固定観測上のσ1→σ2のtotal variation、二乗Hellinger距離、greedy選択の変化率
- learnerのloss・勾配ノルム・samples/s、collectorの局/s、別工程のV計算時間
- Nano対戦の平均順位と、補正なしとの差の95%信頼区間

異なる状態を混ぜた教師の二次モーメントは、同一情報集合における条件付きノイズ分散そのものではありません。lossや教師ノルムが下がっただけでは強くなったと判定しません。凍結Nano Vの方策・評価期間は現在のCFRと一致する保証がなく、baselineが分散を増やす可能性もあります。

## 実行（履歴ソース用）

`prepare_trial.py <前回trialの親ディレクトリ> <独立した出力先>` で前回の確定設定を読み、smokeと本試験の実行計画を生成します。共通の `CfrConfig` とNanoの `DecisionInferenceSettings` を使用します。

`ValueBaselineExperiment` は次のコマンドを持ちます。

- `settings.json import`: 1 iteration目の保存済み行を独立した出力先へコピー
- `settings.json center <iteration> <arm>`: Vを計算し、教師診断を保存してReplayを確定
- `settings.json probe <iteration> <arm>`: 固定観測上の方策変化を測定

収集・学習・対戦評価は既存の `PivotReuseExperiment` と `PivotReuseNanoDuel` を再利用します。`collect-rows` は圧縮行の保存まで、`commit` は補正済み行のReplay保存を担います。実行計画は既存の `run_stages.py` で終了コードを記録します。

結果集計:

```text
python analysis.py --baseline <前回trial> --candidate <今回trial> --output comparison.json
```

`FrozenBaselineMathTest` は一様pivotとmixtureの期待教師・MSE勾配、ゼロ確率、強制行動、途中終局、未到達pivot、情報制約、中心化単位を検証します。

## 2026-10-08の結果

**凍結Nano Vをそのまま引く今回の方式では、強さの改善を確認できませんでした。既定の学習方式には採用しません。**

サーバー上の実行先は `/home/koutachan/epsilon-ai/runs/cfr-value-baseline-20261008` です。17工程すべて終了コード0、実行時間は約29分でした。物理GPU 1を使用し、補正なしの保存モデルの方策変化測定のみ空いていた物理GPU 0で実行しました。新規の学習用収集は合計65,536局、別途smokeが48局です。初回の共有データを再利用したため、各armが学習に用いた局数はそれぞれ65,536局です。

各候補8,192半荘で、平均順位は小さいほど良い値です。

| 方式 | Vなし | 凍結Vあり | 差（Vあり−なし） | 差の95%信頼区間 |
| --- | ---: | ---: | ---: | --- |
| pivot | 2.60034 | 2.59204 | −0.00830 | [−0.04028, +0.02368] |
| mixture | 2.73669 | 2.76709 | +0.03040 | [−0.00117, +0.06197] |

両方とも補正なしとの差の信頼区間は0を含み、改善を確認できません。初期Nano蒸留averageの平均順位は2.50806で、Vありとの差はpivotが+0.08398 [ +0.05288, +0.11509 ]、mixtureが+0.25903 [ +0.22736, +0.29071 ]でした。両候補とも初期モデルより悪化しています。すべて2,048個の共通牌山を対応させた探索的95%区間です。

同一の各収集データに対する教師二乗ノルムの変化は次のとおりです。2 iteration目の行はarmごとに新規収集したもので、前回試験と同じ対局ではありません。

| 方式 | iteration 1 | iteration 2 |
| --- | ---: | ---: |
| pivot | −1.85% | +9.74% |
| mixture | −2.47% | +8.68% |

固定Nano Vは初回の教師二次モーメントを少し減らしましたが、学習後の方策で集めたデータでは増やしました。現在のCFR方策の効用に合わせて新たに学習したVは試していないため、状態baseline一般が無効という結論ではありません。また、最後のaverage評価はσ1とσ2を含むため、2回目のregret更新における教師の増加がこの評価悪化の直接原因とは言えません。

共通の15,577観測で測定した初期σ1からσ2への方策変化も、小幅な減少にとどまりました。

| 方式 | total variation（なし→あり） | greedy行動変更率（なし→あり） |
| --- | --- | --- |
| pivot | 0.73504 → 0.71626 | 83.67% → 82.72% |
| mixture | 0.66944 → 0.64950 | 75.61% → 72.79% |

初回Replayの重み指標、実効learner設定、batch、更新数、logical/batch内unique抽選件数は前回と一致しました。最終Regret Replayはpivotが32,485行・ESS 29,430.68、mixtureが719,504行・ESS 58,534.14です。ESSは重みの集中度の指標で、学習品質や独立した情報量の保証には使いません。

実測速度と追加費用は次のとおりです。

| 指標 | pivot | mixture |
| --- | ---: | ---: |
| iteration 2 collector（局/s） | 143.25 | 141.43 |
| iteration 1 Regret learner（samples/s） | 3,164.27 | 3,063.10 |
| iteration 2 Regret learner（samples/s） | 2,836.58 | 2,850.72 |
| iteration 2 Average learner（samples/s） | 2,043.14 | 2,081.96 |
| V変換の2 iteration合計（秒） | 106.76 | 196.53 |

collectorは両streamの収集と圧縮行保存込みです。learnerは抽選・checkpoint保存・GPU同期込みのlogical samples/sで、V計算を含みません。V変換はモデルロード・行展開・推論・圧縮保存込み、Replayへのcommitを除きます。前回とは実行時刻と他プロセスの負荷が異なるため、速度の差をV補正の高速化効果とは解釈しません。

Java 21コンパイル、数学テスト5件、解析テスト8件、GPU smokeを通過し、全1,536更新のlossと勾配ノルムは有限でした。JavaとPythonのフォーマッタを適用済みです。完全な集計と生の対戦結果は `epsilon-cfr/target/remote-training/value-baseline-evidence/` に保存しています。

次に切り分けるべき点は、初期Nano方策からRegret Matching方策へ切り替える際の大きな変化と、averageネットワークの近似誤差です。この試験では両者の寄与を確定していません。
