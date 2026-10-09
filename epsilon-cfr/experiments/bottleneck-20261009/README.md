# CFR と Nano の速度差の原因調査

2026-10-09、RX 7900 XTX 24 GiB の物理 GPU 0 で測定しました。対象は `ebd2e4b427903df8ac3772e3e20de7bbaa328dea` です。既存の学習 run・checkpoint・本番ソースは変更していません。

今回の測定条件では、主因は learner の長い履歴系列の順逆伝播と、collector の保存待ちが収集の主スレッドを止める構造でした。optimizer の更新・同期は train 時間の 1% 未満で、主要因ではありません。

## Learner：全履歴の系列処理

同じ実観測・候補 bucket `16x1`・固定重みを使い、GPU 完了を待って比較しました。各条件は warmup 8 回、測定 15 回の中央値です。

| 入力行数 | 平均履歴 token | CFR 全体の順逆伝播 | 履歴を事前計算した CFR | 履歴だけの順逆伝播 | Nano Actor/Value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 113 | 454 | 52.01 ms | 10.86 ms | 41.37 ms | 14.37 ms |
| 512 | 450 | 88.14 ms | 28.60 ms | 59.91 ms | 34.12 ms |
| 1,024 | 450 | 146.32 ms | 53.91 ms | 93.62 ms | 61.64 ms |
| 2,596 | 449 | 350.68 ms | 138.26 ms | 212.63 ms | 154.43 ms |
| 113 | 847 | 83.72 ms | 11.48 ms | 72.47 ms | 15.07 ms |
| 512 | 841 | 138.25 ms | 31.70 ms | 107.04 ms | 37.31 ms |

512 行・約 450 token では、履歴計算を除くと全体の 67.6% に当たる時間が減りました。通常の重み付き抽出に近い 113 行・約 847 token では 86.3% です。大きな batch でも差が残るため、batch 拡大だけでは解消しません。

盤面入力の生成・転送は共通化して測定区間から除外しています。CFR 全体と履歴単独は、本番の `fullHistory` による履歴の CPU packing・転送を含みます。optimizer・教師・checkpoint 保存は含めず、出力の二乗平均を使う診断です。Nano とは本番の損失関数が異なります。

「履歴を事前計算」は同じ hidden を GPU 上で切り離して渡す原因切り分けです。履歴への勾配を除くため、そのまま本番に採用できる高速化ではありません。重み更新をまたぐ古い hidden の再利用も検証していません。収集時はすでに増分 GRU を使っており、この全履歴 forward の結果を collector の推論速度とは扱いません。

### 系列長が増える構造

[`VisibleHistoryRecorder`](../../src/main/java/com/epsilon/cfr/collection/VisibleHistoryRecorder.java) はイベント記号と各フィールドの可変長整数表現を token に展開します。今回、1 イベントが平均 4.42～4.65 token になっていました。[`CfrStateRepresentation.fullHistory`](../../src/main/java/com/epsilon/cfr/model/CfrStateRepresentation.java) は学習の各 microbatch で全系列を再計算し、batch 内の最大長まで padding します。

| 本番 learner 経路の測定 | 重複なし診断 | 通常の重み付き抽出 |
| --- | ---: | ---: |
| 実処理行数 / 更新数 | 65,536 / 64 | 14,438 / 128 |
| 履歴 microbatch 当たりの平均行数 | 512 | 112.8 |
| 1 行の平均実 token | 457.0 | 853.8 |
| padding 後の平均 token | 683.5 | 934.7 |
| 実 token 数に対する padding 増分 | 49.6% | 9.5% |

重み付き抽出では同一更新内の重複を集約するため行数が減り、長い履歴を少数行ずつ処理しています。今回の replay では長い履歴の行へ重みが偏っていました。抽選数/s と実処理行/s は分けて判断する必要があります。

### Optimizer 仮説の確認

本番 learner の各区間を現在の worker stream の GPU event で同期させた診断です。

| 区間 | 重複なし 64 更新 | 重み付き 128 更新 |
| --- | ---: | ---: |
| train 呼び出し全体 | 16.866 s | 12.578 s |
| 履歴 GRU 順伝播 | 3.277 s | 3.844 s |
| 全 graph の逆伝播 | 10.345 s | 7.111 s |
| optimizer と更新完了待機 | 0.046 s（0.28%） | 0.083 s（0.66%） |

逆伝播の数字には盤面・候補 network も含まれるため、全てを GRU の時間とは扱いません。履歴の寄与は上の同入力比較で確認しています。細かな optimizer 更新や勾配 norm の読み戻しが主因という仮説は、この測定では支持されません。

## Collector：保存待ちが収集全体を止める

同じ checkpoint・開始局面・seed で `HAND_GRP` 4,096 局、warmup 256 局を実行しました。保存なしでも観測生成、payload encode、decision 保持、全席推論、ロン見逃し／九種九牌の分岐、CPU の固定 GRP は維持しています。非同期の batch 構成による対局内容の小さな揺らぎがあります。

| 条件 | 経過時間 | decision/s | 推論 pipeline に処理中 batch がない時間 |
| --- | ---: | ---: | ---: |
| 保存あり・区間計測と JFR あり | 14.841 s | 24,434 | 6.009 s |
| 保存なし・区間計測あり・1 回目 | 9.947 s | 36,454 | 1.323 s |
| 保存なし・区間計測あり・2 回目 | 9.898 s | 36,614 | 1.537 s |

保存を省いた診断では 49.2～49.8% 速くなりました。これは保存まで含む本番速度の改善結果ではありません。保存ありの無計測対照も 24,666 decision/s で、区間計測と JFR を含む速度低下は約 1% でした。

[`CfrCollector.Scheduler.publishCompleted`](../../src/main/java/com/epsilon/cfr/collection/CfrCollector.java) が完了通知を同期的に呼び、その中の [`CfrEpisodeWriter.accept`](../../src/main/java/com/epsilon/cfr/data/CfrEpisodeWriter.java) が待機・書き込み中の合計 16 件という上限で `wait()` します。保存自体は 4 ワーカーでも、主スレッドがそこで停止すると対局進行の回収・補充と推論の供給も止まります。

- writer への投入区間は主スレッドの wall 時間で 4.670 s。JFR でも `CfrEpisodeWriter.accept → Object.wait` を 4.130 s、1,225 件確認しました。JFR は 1 ms 以上の待機を記録する設定です。差には短い待機やキュー処理などが含まれ、4.670 s 全てを純粋な待機時間とは扱いません。
- writer の native sample 529 件中 370 件が `FileChannel.force(true) → force0`、107 件が圧縮、43 件が rename でした。局ごとの同期保存が詰まりと整合します。sample 数は GPU kernel 時間やディスク待ち秒数ではなく、fsync 単独の寄与は未分離です。
- `pipelineIdle` は入力の enqueue から結果の compose・release までを追う pipeline に処理中 batch がない時間です。GPU のハードウェア idle 率ではありません。保存なし 1 回目では、この時間の減少が wall 短縮の約 96% に相当しました。

保存なしでも Nano の以前の本番収集結果（約 81,000 推論要求/s）には達していません。局と半荘、保持・保存対象の違いがあるため、残る差をそのまま一つの処理の寄与とは断定しません。今回特定したのは、CFR 内で大きな時間を失う保存経路と主スレッドの停止です。

### 次に大きい CPU コスト

測定区間の JFR による推定累積 allocation は 53.09 GB、そのうち観測 encode・`Prediction` 構築・`CfrDecision` 構築・保存時の accessor による `byte[]` が 32.29 GB（60.8%）でした。これは確保量の累計推定であり、同時に使用した RAM 容量ではありません。GC pause は計 1.246 s でした。

入力連結は 1.616 s、出力構築は 1.840 s。これらは他スレッドや保存待ちと重なるため、wall 時間へ足し合わせられません。hidden の転送量は片方向 185.6 MB、cache 公開は計 65 ms で、転送量だけから hidden の GPU 常駐化を最優先とする根拠は得られませんでした。

## 実測精度の訂正

設定は `BFLOAT16`、master parameter の保持は FP32 ですが、実際の GRU 出力 tensor は **FLOAT16** でした。履歴 GRU・境界 GRU とも、本番 learner の区間計測と同入力比較の両方で確認しています。DJL の `DataType.toString()` も確認し、BFLOAT16 の別名表示ではありませんでした。

以前の「GRU は FP32 を維持する」という説明は、パラメーターの保持型と実計算を混同していました。訂正します。出力型から内部の全 kernel の演算・累積精度までは断定していません。今回、精度設定は変更していません。

## 修正の優先順位

1. **保存容量の制限と収集 scheduler の進行を分離する。** 保存が詰まっても進行中の対局・推論を回収できる構造にし、新しい局の投入などで流量を制御します。容量上限、保存完了待ち、atomic 公開、エラー伝播は維持します。局ごとの同期保存をまとめる設計は、復旧単位を定義して別途評価します。
2. **全情報を保ったイベント単位の履歴表現へ設計し直す。** 一つのイベントの型・席・牌などの各フィールドをまとめて符号化し、GRU が追う系列を短くします。入力表現・network が変わるため、再学習と強さの比較が必要です。4.4 倍の token 削減がそのまま 4.4 倍の速度改善になるとは見積もりません。padding 削減も同じ測定で評価します。
3. **観測 payload の所有権を整理して重複コピーを減らす。** engine が更新する盤面の寿命と、保存まで必要な不変データの寿命を明示し、不要な clone をなくします。

## 再現条件と証拠

[測定結果 JSON](results.json) に実行条件、通常経路の結果、区間計測、同入力比較の全反復、JFR 集計を記録しています。設定は [前回の CFR 設定](../performance-20261009/cfr.toml) と同じです。本番の 1 iteration 32,768 局は変更せず、今回の収集診断だけを 4,096 局にしています。

- GPU 0 を単独利用。Java 21.0.12.1、DJL JNI 0.40.4、PyTorch 2.11.0 / ROCm 7.2。`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。GPU 1 と既存学習 run は操作していません。
- 無計測の再確認：収集 24,666 decision/s、重複なし learner 3,653 実処理行/s、通常抽出 9,337 抽選/s・1,029 実処理行/s。learner は抽選・展開・順逆伝播・更新・定期 checkpoint 保存と同期を含みます。warmup・初期読込・replay 構築は除きます。
- learner の計測追加による train 時間増加は、GPU 強制同期なしで 0.09% / 0.32%、ありで 4.24% / 3.26%（重複なし / 重み付き）。`inclusive.*` は子区間を含み、重複加算しません。
- 同入力比較の最初の smoke は計測 helper の tensor 寿命管理に不備があったため除外しました。修正後の `model-short-1` / `model-long-1` は各条件の全反復で step 後の GPU allocation が一定、`complete=true`、exit 0 を確認しています。本番のメモリリークを示した結果ではありません。
- 本番へ計測クラスを追加していません。計測 overlay・helper・実行 script はローカルの `epsilon-cfr/target/bottleneck-20261009/`、サーバーの `/home/koutachan/epsilon-ai/benchmarks/cfr-bottleneck-20261009/` に保持しています。Java helper は本番 fat JAR に対して compile し、上記の実行で検証しました。

今回の成果は原因の特定です。学習品質や本番の高速化を達成したという報告ではありません。
