# CFR 後悔学習の修正比較

この文書は完了した実験の手法・結果記録です。一時的な実行・診断コードは削除済みです。以下の実行手順は、削除前のコミット `79a7530` のソースと当時の入力データを使う場合の手順です。

**比較は完了しました。** 初期表現の変更と scalar baseline を組み合わせた D は、従来条件 A より平均方策の悪化を抑えましたが、初期モデルより弱い結果でした。初回 baseline にも汎化の問題が残っています。[全13評価・独立テストの結果](RESULTS.md)を参照してください。基準ソースは `edcc748` です。

Q・Oracle・未選択行動の環境分岐は追加しません。既存の Nano 表現、CFR collector、posterior-mixture 教師、replay、learner を使い、以前の実験 runner には条件分岐を継ぎ足さず、`RegretRepairExperiment` で比較を管理します。

本番の Nano 初期化は従来どおりです。蒸留済み表現への切替と baseline の適用はこの runner の比較条件として明示しています。今回の結果では、既定動作への採用を見送ります。

## 4条件

| 条件 | 初回 REGRET の表現 | scalar baseline |
| --- | --- | --- |
| A | 保存済みの従来初期 REGRET | なし |
| B | 蒸留済み初期 AVERAGE からコピー | なし |
| C | A と同じ | 物理局で2分割した cross-fit |
| D | B と同じ | C と同じ方式 |

A/C の従来初期モデルは Nano から共通部分を移植したものです。B/D は `CfrModel.initializeRepresentationFrom` で初期 AVERAGE の出力以外の表現を引き継ぎ、学習済みの履歴表現も利用します。どちらも REGRET の出力はゼロ、optimizer は新規です。AVERAGE の出力や optimizer を REGRET に流用しません。

初期 AVERAGE、参照方策、root bank、GRP、元の収集ファイルは固定します。初回は同一の収集データを4条件で共有し、C/D の baseline も共有します。2反復目は各条件が選択した方策から収集し、C/D の baseline は各条件のデータだけで別々に学習します。

## 学習とデータ数

| 項目 | 設定 |
| --- | --- |
| 反復数 | 2 |
| 学習用局数 / 反復 / 条件 | 32,768：REGRET 24,576 + AVERAGE 8,192 |
| REGRET 更新 / 反復 / 条件 | 256 |
| AVERAGE 更新 | 2反復目の最後に256 |
| batch / microbatch | 1,024 / 512 |
| learning rate / weight decay / gradient clip | 0.0002 / 0.0001 / 0.5 |
| 計算精度 / 転送 | BFLOAT16 / PINNED_BUFFER |
| 教師推定 | MIXTURE、pivot hazard 1/16、参照方策の uniform 混合率0.1 |
| モード | HAND_GRP、root utility で中心化 |
| 学習収集 seed | 20261008 |

初回の32,768局は既存ファイルを読み込むため、この比較のために再生成する局数はゼロです。「条件あたり同じ学習局数」と「新しく生成した物理局数」を区別します。2反復目は各条件で32,768局を新規収集します。同じ乱数系列を使いますが、方策が変わるため、その後の状態や行動は一致しません。

各条件の replay は過去反復の行も保持し、2反復目に直近の行だけへ置き換えません。容量は REGRET / AVERAGE 各1,048,576行です。実際の保持数、抽選重み、storage は replay report に記録します。

## baseline の目的と独立性

保存済み行の utility を `Y`、選択行動を `A`、その実際の target probability を `p = σ(A)`、合法手数を `n` とします。後悔教師は次です。

```text
d = Y (e_A − p 1)
h = ||e_A − p 1||² = 1 − 2p + n p²
```

`CfrBaselineModel` は状態・本人の判断前履歴から scalar `b(I)` を出します。候補行動、選択行動、終局結果を推論入力に与えません。Nano の既存 value head の値をそのまま使う方式ではありません。初期 AVERAGE の状態・履歴表現から新しい scalar head を学習し、保存済みの中心化された `Y` を直接予測します。

baseline の目的は `Σ w h (Y − b(I))²` です。`BaselineSample.from` により replay の抽選重みを `w*h` とし、抽選後の learner は通常の均一 MSE を使います。損失に同じ重みをもう一度掛けません。baseline の学習は各 fold 256更新で固定し、検証データによる早期停止や checkpoint 選択は行いません。

fold は `hash(seed, iteration, episodeId / 4) & 1` です。1物理局の全行を同じ fold に置き、4席の owner が1回ずつ現れる4局の組も同じ fold にするため、各 fold 内の owner 数が均等になります。fold 0 のデータには fold 1 だけで学習したモデルを適用し、逆も同様です。行をランダム分割して同じ物理局を学習側と予測側へ分散させません。

REGRET に渡す行では `utility = Y − b(I)` だけを変更します。行ID、観測、合法手、選択行動、σ、元の regret weight は維持します。AVERAGE の教師は変更しません。baseline 用 replay の `w*h` を REGRET replay に持ち込まないことが重要です。

`baseline-complete.json` の `teacherMoments` は、元の `w` を一度だけ使った `E_w[||d||²]`、補正後の二次モーメント、比、utility / residual / baseline の二次モーメント、重みの実効サンプル数を記録します。中断からの再実行でも全局を再計測するため、新たに処理した分だけの値にはなりません。これは教師の二次モーメントであり、条件付き雑音分散を直接測った値ではありません。

baseline の学習・補正推論・検証は追加コストです。共通の REGRET / AVERAGE 更新数を揃えても、4条件の総 GPU 計算量が等しくなるわけではありません。

事後診断用の `BaselineGeneralizationAudit SETTINGS OUTPUT.json` は、固定した初回 baseline を全24,576局に対して推論し、自分の学習 fold と反対 fold の誤差を分けます。`w*h` による MSE、`b²`、`Y*b` を記録するため、学習への適合と汎化を比較できます。モデルや教師行を更新せず、checkpoint 選択にも戻しません。この診断の所要時間は学習速度に含めません。

## 検証による checkpoint 選択

全条件で256更新まで実行し、32 / 64 / 128 / 256更新の training snapshot を保存します。更新を途中で打ち切る実験ではありません。

各反復で、学習とは別 seed の4,096局を使い、元の未補正教師に対する重み付き `τ_RM · d` の平均が最大の候補を選択します。`τ_RM` は stochastic Regret Matching 方策です。完全同値の場合だけ少ない更新数を選びます。greedy の局所値と MSE は診断用で、選択基準には追加しません。

初回の検証は既存の独立収集から owner が均等な先頭4,096局を使用します。seed は `2026100801` です。これは checkpoint 選択に使う validation なので、最終テストの証拠とは区別します。2反復目の validation は、各条件の実際の収集方策で同じ seed 系列から新しく収集します。validation は baseline と regret の学習に入りません。

保存先を分けています。

```text
trial/A/manifests/regret-1.json   # 4個の候補をすべて保持
trial/A/selected/regret-1.json    # 検証が選択した候補
trial/A/progress/regret-1.json    # 実行は256更新まで完了
```

2反復目の REGRET は `selected.optimizerCheckpoint` の重みと AdamW 状態を一緒に継続します。最終256更新の optimizer と、早い更新時点の重みを混ぜません。検証値が負でも初期方策へ戻す gate はありません。負の値は警告として保存します。この比較は、gate や Nano との固定混合で悪化を隠す実験ではありません。

## 平均方策に何が反映されるか

初回の収集方策 `σ1` は初期 AVERAGE、2反復目の収集方策 `σ2` は初回 REGRET の選択 checkpoint です。平均用の独立 stream と既存の iteration weight により、この2世代を重み1:2で学習します。本人の reach は収集方式に含まれており、2つのネットワーク出力を状態ごとに単純に1:2で混ぜる処理ではありません。

最終 AVERAGE は初期 AVERAGE の重み・optimizer から256更新します。2反復目の REGRET 更新で得られる次の方策は、まだ収集に使用していないため、この平均には入りません。2反復目の REGRET 学習も既存比較と学習予算を揃えるため実行しますが、その改善を最終 AVERAGE の変化の原因と説明しません。

`current` 対戦も、実際に `σ2` として使用した初回の選択 REGRET を評価します。2反復目の最後の REGRET を別モデルとして紛れ込ませません。

2反復目の REGRET から選択した次の方策を `σ3` と呼びます。これを調べる追加の `next-current` 対戦は、下記の副次評価として扱います。`σ3` を評価しても、既存の最終 AVERAGE の教師は `σ1` と `σ2` のままです。

## 最終評価

主評価は最終 AVERAGE と固定 Nano の greedy 対戦です。初期 AVERAGE、A/B/C/D の `current`、A/B/C/D の最終 `average` を同じ未使用の壁系列で評価します。2,048 wall seeds、4席交替の8,192半荘 / モデル、seed base `50000`、sequence `773` です。差と信頼区間は4席分が揃った wall を単位として対応付けます。

raw score の独立テスト用には seed `2026100803` で32,768局（REGRET 24,576 + AVERAGE 8,192）を収集済みです。データは実験出力の `test/collections/shared/iteration-1/mixture-regret` にあり、元の `σ1` に対する、初回で選択した4つの REGRET を比較します。このテストは validation の候補選択へ戻しません。`prepare_final_test.py` で選択済みモデルの評価計画を確定し、gzip 行出力から物理局単位の区間推定を行います。

```bash
python3 prepare_final_test.py OUTPUT
CUDA_VISIBLE_DEVICES=0 python3 ../pivot-reuse/run_stages.py OUTPUT/control/final-test-plan.json
```

この helper は `control/final-test-audit.json`、`control/final-test-plan.json`、`analysis-manifest.json` を生成します。元の manifest や selected を書き換えません。評価結果は `audits/final-test.json` と条件別の gzip 行ファイルへ保存され、集計には `analysis.py OUTPUT --manifest OUTPUT/analysis-manifest.json` を使用します。

局所の `τ·d` は後続を収集方策に固定した1判断の尺度です。半荘全体の強さと同じものではありません。baseline による教師二次モーメントの低下だけでも、学習改善とは判定しません。

学習 seed は1本です。条件間差と `D − C − B + A` の交互作用はこの固定実験の探索的な結果として扱います。多重比較の補正や、4人一般和麻雀での均衡収束の保証はありません。主評価・追加評価・独立テストはすべて完了しており、自動昇格は行っていません。

## 追加の次方策評価

途中の教師診断を受け、2反復目の後悔学習で得られた `σ3` も調べる副次評価を追加します。A/B/C/D 各条件の `trial/ARM/selected/regret-2.json` から、既定の validation 基準で選択済みの checkpoint を固定します。全条件とも32 / 64 / 128 / 256更新のいずれかであり、対戦結果を見て選び直しません。

既存の9評価（初期 AVERAGE、4条件の `current`、4条件の最終 `average`）はそのまま残し、`A-next-current` から `D-next-current` の4評価を追加します。対戦は同じ Nano、同じ2,048 wall seeds・4席交替で、追加費用は4モデル × 8,192半荘です。モデルの追加学習や3反復目の収集は行いません。最終 AVERAGE を主評価とする位置づけも変えません。

`σ3` と `σ2` の差には、新しい収集データ、累積 replay、追加の後悔学習、checkpoint 選択の影響が含まれます。C/D の教師二次モーメントが下がった場合でも、この差を baseline の効果だけと解釈しません。条件間の `next-current` の比較は2反復にわたる各介入全体の結果であり、2反復目の baseline だけを独立に入れ替えた比較ではありません。

```bash
python3 prepare_next_policy_test.py OUTPUT
CUDA_VISIBLE_DEVICES=0 python3 ../pivot-reuse/run_stages.py OUTPUT/control/next-policy-0-plan.json
CUDA_VISIBLE_DEVICES=1 python3 ../pivot-reuse/run_stages.py OUTPUT/control/next-policy-1-plan.json
python3 analysis.py OUTPUT --manifest OUTPUT/next-policy-analysis-manifest.json
```

`next-policy-0` は A/C、`next-policy-1` は B/D を評価します。plan は選択済みの実 checkpoint パスを固定し、結果を `results/ARM-next-current.json` へ保存します。拡張 manifest は `next-policy-analysis-manifest.json` で、元の manifest、`analysis-manifest.json`、9評価の定義や selected は書き換えません。この追加評価は主評価の処理と baseline の事後診断が終了してから開始し、同じ GPU 上で推論を競合させません。

## 実行手順（履歴ソース用）

`prepare_trial.py` は既存実験・診断結果を入力にし、別ディレクトリへ設定と実行計画を保存します。元の checkpoint、root bank、行ファイルを変更しません。

```bash
python3 prepare_trial.py ORIGINAL_PIVOT_EXPERIMENT REGRET_DIAGNOSIS OUTPUT \
  --baseline-commit edcc748
```

必要な Java ソースを固定した class directory へコンパイルしてから、既存の `../pivot-reuse/run_stages.py` で次の順に実行します。各 stage は終了コードとログを保存し、成功済み stage の再実行を省略します。

| 順序 | 計画 | 内容 |
| --- | --- | --- |
| 1 | `bootstrap-plan.json` | 表現初期化、共通の train / validation 行をコピー |
| 2 | `first-0-plan.json` | A/B の REGRET 学習、検証、選択 |
| 2 | `first-1-plan.json` | 共通 baseline の2 fold 学習、train / validation 補正、C/D の学習・検証・選択 |
| 3 | `second-0-plan.json` | A/C の2反復目収集、学習、平均学習 |
| 3 | `second-1-plan.json` | B/D の同じ処理 |
| 4 | `duels-0-plan.json` / `duels-1-plan.json` | 固定 Nano との対戦 |
| 4 | `prepare_final_test.py` の計画 | 独立 raw-score テストと行出力 |
| 5 | `next-policy-0-plan.json` / `next-policy-1-plan.json` | 主評価と baseline 診断の完了後、追加の `σ3` 対戦 |

同じ順序番号の2計画は別 GPU で並列実行できます。次の番号は必要な前段が正常終了してから開始します。設定内は `gpu:0` のまま、プロセスごとの `CUDA_VISIBLE_DEVICES` で物理 GPU を分けます。

```bash
CUDA_VISIBLE_DEVICES=0 python3 ../pivot-reuse/run_stages.py OUTPUT/control/bootstrap-plan.json
# bootstrap 完了後、別のプロセスとして開始する。
CUDA_VISIBLE_DEVICES=0 python3 ../pivot-reuse/run_stages.py OUTPUT/control/first-0-plan.json
CUDA_VISIBLE_DEVICES=1 python3 ../pivot-reuse/run_stages.py OUTPUT/control/first-1-plan.json
```

runner 単体の command は次です。Java の classpath / heap 設定は生成された `control/java-command.json` と揃えます。

```text
RegretRepairExperiment settings.json init
RegretRepairExperiment settings.json import
RegretRepairExperiment settings.json baseline 1 shared
RegretRepairExperiment settings.json center 1 shared train|validation
RegretRepairExperiment settings.json collect 2 A|B|C|D train|validation
RegretRepairExperiment settings.json baseline 2 C|D
RegretRepairExperiment settings.json center 2 C|D train|validation
RegretRepairExperiment settings.json commit 1|2 A|B|C|D regret|average
RegretRepairExperiment settings.json learn 1|2 A|B|C|D regret
RegretRepairExperiment settings.json learn 2 A|B|C|D average
```

`audit_candidates.py` が manifest の全候補を同一行で評価し、`selection.py` が選択結果を別ファイルへ書きます。`analysis.py OUTPUT` は対戦、選択結果、教師診断、学習・収集・stage 時間をまとめます。collector の games/s、learner の samples/s、baseline の追加費用、並列実行の経過時間は区別して報告します。

## 検証済みの範囲

Java formatter を適用し、server native test は28/28通過しています。fold 分離と owner の均等性、重みの一度だけの適用、補正による期待教師の維持、state-only 入力、optimizer の保存復元、選択 snapshot の整合、全件の重み付き二次モーメントを検査しました。これらは実装の検査であり、対戦成績の改善を示す結果ではありません。
