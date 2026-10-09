# Learner最適化の比較（2026-10-09）

盤面入力版、global batch 1,024 / microbatch 1,024のまま、重み・抽選分布・1更新1clip/AdamWを保つ実装変更を検証する。収集の半荘化は別の作業として扱う。

## 測定条件

GPU0のみ。基準jarは[results.json](results.json)の`provenance`参照。旧固定局collector時点の保存済みreplayをそのまま読み、各ケースを同一初期checkpoint、抽選seed20261008から開始する。warmup8更新は除外し、weighted128更新、uniform-distinct64更新を測る。32更新ごとのcheckpoint保存と最終GPU完了を含める。保存頻度とglobal batchは両者同じ。2回目は候補と基準の実行順を反転する。

`ExistingReplayBenchmark`はreplay再構築と形状集計を計測外に置く診断用helperである。旧benchmark値との差ではなく、このhelperで再測定した基準と候補を比べる。VRAMはGPU0全体を20ms間隔で、準備・warmupを含むプロセス全期間に取得した値。PyTorch allocatorの計測値とは区別する。

## バッチ内の行読み込み集約：不採用

同じ抽選indexの行をバッチ内で一度だけloadする候補。tree走査、乱数消費、抽選順、教師の値は同じ。CPUで内容と乱数の完全一致、defensive getterの所有権、次バッチでの再loadを確認した。GPU既存4testsおよび同初期状態からの局所6更新×2roleも合格した。しかし、既存payloadキャッシュが有効な分布では明確な速度改善を確認できず、コードと専用testを撤回した。

| ケース | 基準の保存込み秒（2回） | 候補の保存込み秒（2回） | 平均時間から計算した速度差 |
|---|---:|---:|---:|
| REGRET weighted | 4.3959 / 4.5174 | 4.4028 / 4.3718 | +1.58% |
| AVERAGE weighted | 8.8278 / 8.9350 | 8.9544 / 9.1439 | -1.85% |
| REGRET uniform-distinct | 6.4287 / 6.5974 | 6.4386 / 6.4908 | +0.75% |

uniform-distinctは候補の一括sampleを通らず、揺らぎの対照にもなる。REGRETの基準2回目はsamplingに通常約0.075秒のところ0.149秒かかっており、差をすべて候補の改善へ帰属させない。

## tensor除算とノルム統計の一括回収：採用根拠なし

maxと平方和を一度にhostへ返す候補。GPU4testsは合格したが、追加したbit完全一致testはFP32極大値付近で相対4.54e-8の差により失敗した。これは通常のFP32丸め差であり、失敗だけを数値不健全の証拠とはしない。その後、旧scalar除算との同初期重み・optimizer状態からの局所6更新×2roleを既存BF16基準で比較し、全更新が合格した。最大AdamW更新relativeL2はREGRET0.000850、AVERAGE0.000761。

| ケース | 基準の保存込み秒（2回） | tensor候補の保存込み秒（2回） | 速度差 |
|---|---:|---:|---:|
| REGRET weighted | 4.2980 / 4.2606 | 4.2965 / 4.3893 | -1.46% |
| AVERAGE weighted | 8.8241 / 8.9709 | 8.8418 / 8.8883 | +0.37% |

明確な速度改善も確認できなかった。元の失敗結果と、その後の数値比較をどちらも保存している。

## Nanoと同じ盤面数での計算量

同じresident DecisionDeviceBatch、同じ精度、同じ形状の実盤面を両ネットワークに渡し、順伝播と逆伝播をGPU完了eventまで測った。各8warmup+15反復、2回目は実行順を反転。教師、optimizer、入力の復号・packing・転送、保存は含まない。CFRはcandidate出力、Nanoはactorとvalue出力の二乗平均を使う診断であり、実際のlearner/sや学習品質の比較ではない。

| 行数 | CFR forward+backward中央値ms（2回） | Nano forward+backward中央値ms（2回） |
|---:|---:|---:|
| 113 | 10.534 / 10.560 | 14.451 / 14.828 |
| 512 | 27.719 / 27.716 | 34.125 / 34.122 |
| 1,024 | 52.082 / 52.156 | 61.941 / 61.809 |

同じ盤面数ではCFRのネットワーク計算自体がNanoより遅いという結果にはなっていない。REGRET weightedは128更新で14,438実行行（約112.8行/更新）に対し、logical drawsは131,072件である。rows/sではこの違いを明示する。

## 共通検証値の再利用：不採用

DataParallelGroupが有限性検証ですでに回収している、スケール適用前の最大絶対値をCFRへ渡す。CFRのscale=1では同じ値で従来のscalar除算を使い、重複したabs/maxとhost完了待ちだけを削除する。scaleが1以外なら実際のスケール済み勾配から最大値を求め直す。共通の検証・scale処理は変えない。

CPUではノルム/clip完全一致1test、既存learner一致5tests、共通のreduce/scale/dual-optimizer契約4testsが合格。GPUも新しいexact norm testと既存4testsが合格した。従来の最大値再計算を残した参照learnerとの同初期重み・optimizer状態からの局所6更新×2roleも全合格。最大AdamW更新relativeL2はREGRET0.0008645、AVERAGE0.0007678で、既存BF16基準を変更していない。

初回のhook性能測定には別プロセスのJFR CPU解析とSFTP回収が重なったことが確認されたため、初回巡回を見出しの性能比較から除外する。全rawは残す。`quiet-*`のケースとして、他のserver CPU解析・大量I/Oを止めた後に同条件で2pairを再測定した。

| ケース | 基準の保存込み秒（2回） | hook候補の保存込み秒（2回） | 保存込み速度差 | train呼出しのみの速度差 |
|---|---:|---:|---:|---:|
| REGRET weighted | 4.5382 / 4.3520 | 4.3220 / 4.3844 | +2.11% | +0.64% |
| AVERAGE weighted | 8.8430 / 8.8673 | 9.0105 / 8.9947 | -1.64% | -1.82% |
| REGRET uniform-distinct | 6.5513 / 6.5698 | 6.4806 / 6.5970 | +0.33% | -0.14% |

数値と契約は保たれるが、learner全体に有効な速度改善が出たとはいえない。REGRETの保存込み差は変更対象外の保存時間の差も大きく、AVERAGEは両runで遅い。約2%の改善とだけ報告することは避ける。性能を採否基準として、この候補も見送った。common hook、CFRの変更、専用testは採用前へ戻し、[results.json](results.json)の`evidenceFiles`内の`rejected-hook/`項目に検証済みソースを保存した。今回の3候補はいずれも製品ソースに残していない。

## 保存処理と診断の限界

現設定の32更新ごとの保存は、REGRETの測定で128更新あたり約1.68秒、保存込み全体の約39%を占める。`CfrModel.saveTraining`までの値であり、campaignのprovenanceやsealは含まない。

128更新済みの固定checkpointから、1回warmup後に5回ずつ変更せず保存した。

| role | weights保存平均ms | optimizer保存平均ms | weights bytes | optimizer bytes |
|---|---:|---:|---:|---:|
| REGRET | 17.88 | 403.60 | 11,427,721 | 19,975,096 |
| AVERAGE | 17.78 | 414.55 | 11,427,721 | 19,911,566 |

bundled Optimizer.saveStateはstate配列をByteArrayOutputStreamへNPZでエンコードし、既定のZIP DEFLATE圧縮を使う。圧縮部分だけを、同一host NPY entries（2配列、22,825,416 bytes）で別測定した。以下はGPU転送やファイルI/Oを含まない、メモリ内圧縮の1warmup+3反復平均である。

| 圧縮level | REGRET ms | AVERAGE ms | REGRET ZIP bytes | AVERAGE ZIP bytes |
|---|---:|---:|---:|---:|
| 既定DEFLATE | 380.02 | 389.57 | 19,975,013 | 19,911,483 |
| 1 | 338.80 | 349.05 | 20,031,130 | 19,976,736 |
| 0 | 8.11 | 8.10 | 22,829,204 | 22,829,204 |

すべて再展開したNPY内容がbyte単位で一致した。level0はoptimizerの容量を約14〜15%、checkpoint全体を約9%増やす代わりに、圧縮計算だけなら約0.37〜0.38秒/saveを削れる。これは保存経路を変更した本番の実測ではなく、寄与を切り分けた診断結果である。

現在のOptimizer APIは圧縮level・stream・encodingを指定できない。AdamWは継承でき、moment stateへのprotectedアクセスもあるが、更新カウンタはprivateでaccessorがなく、loadStateはclass名の完全一致を要求する。薄いsubclassでsaveStateだけを置き換えると、既存状態の継続をきれいに保証できない。private状態の複製管理やreflectionを導入せず、正式なSDK側の保存圧縮APIが必要という結論に留める。今回SDKは改変していない。

更新固定費probeの初版はsynthetic gradientのfixtureによりNano側でzero-gradientになったため無効とし、その時間から結論は出していない。

## 再現用ファイル

[results.json](results.json)に全測定値、設定、数値比較、失敗試行、診断ソースを集約した。

- `summary`、`measurements`、`resources`、`numericalComparisons`、`diagnosticProbes`: 判定に用いた構造化データ。
- `provenance`: 基準jarと差替えclassの識別情報。
- `evidenceFiles`: 元の293ファイルを相対パス名で索引した全文。`raw/`の失敗ログ・終了コード、helper、実行script、不採用ソース、manifest、整理前のREADMEを含む。各項目の`content`をUTF-8で保存すると、元の改行を含むbyte列を復元できる。元サイズとSHA-256も付けた。

元ディレクトリ一式はGit管理対象外の`epsilon-cfr/target/learner-refinement-20261009/experiment-source-snapshot`にも保存し、全293ファイルについてバックアップ・JSON内の全文・元ファイルのbyte一致を確認してから整理した。製品repo上の実験記録はこのREADMEとresults.jsonの2ファイルだけである。

本番ソースに診断フック、時刻計測、出力コードは追加していない。
