# Epsilon CFR 後悔学習の原因診断

この文書は完了した実験の手法・結果記録です。一時的な実行・診断コードは削除済みです。以下の実行手順は、削除前のコミット `79a7530` のソースと当時の入力データを使う場合の手順です。

2026-10-08の`cfr-regret-diagnosis-20261008`の結果です。[方策と平均ネットワークの診断](../policy-diagnosis/README.md)で確認した現在方策`σ₂`の性能低下について、後悔予測の汎化、後悔マッチング（RM）の全非正時の挙動、行動選択を分けて調べました。

**今回確認できた直接の問題は、学習データには適合した後悔予測が独立データに汎化せず、その予測から作った方策でNano由来の初期方策を置き換えていたことです。** pivot・mixtureとも、独立データでの二乗誤差はゼロ予測より悪く、予測から選んだ行動の局所的な改善量も負でした。全非正時の一様方策を取り除いても実対戦の大幅な悪化は残りました。

## 対象と方法

対象は[元のpivot / mixture比較](../pivot-reuse/README.md)の補正なし2モデルです。どちらも第1反復の後悔学習256更新後の凍結checkpointで、これから第2反復の現在方策`σ₂`を作ります。凍結Nano Vを使ったモデルは今回の回帰診断の対象外です。

- 学習データは各モデルの実際の学習元streamを使いました。pivotモデルには`pivot-regret`、mixtureモデルには`mixture-regret`を対応させています。各ミニバッチに実際に抽選された回数の再現ではなく、学習元に保存された全行の評価です。
- 独立データは学習seed `20261008`から別seed `2026100801`に変更して収集しました。凍結`σ₁`、参照方策、局開始状態bank、GRP、効用は同じです。新規収集は合計32,768局（regret 24,576局＋average 8,192局）で、regretの同じ実行軌跡からpivotとmixtureの両方の行を作りました。
- 今回の診断では学習を行っていません。新規average行も学習には使用していません。Q、Oracle、未選択行動の分岐実行は追加していません。
- `logRegretWeight`を一度だけ適用し、合法行動について二乗誤差を合計しました。これはlearnerの回帰目的に対応します。mixtureの重みなし集計は補助値であり、学習目的や局所的な改善の判定には使いません。

保存行数はtrainのpivot / mixtureが14,761 / 376,392、heldoutが15,110 / 378,741です。行のない局は順に9,815 / 3 / 9,466 / 0でした。以下の95%信頼区間は同一局の全行をまとめ、空局も含む24,576局で計算した正規近似delta methodです。比の分子・分母の共分散を含み、多重比較補正はありません。学習元の当てはまりと独立データでの汎化は分けて扱います。ESSも保存していますが、独立な結果数や有益な信号量の保証には使いません。

## 1. 学習データでは改善し、独立データではゼロ予測より悪化

保存教師を`d`、予測を`f`とすると、ゼロ予測との差は`E_w[||f||² − 2 f·d]`です。表は`100 × (MSE(f) / MSE(0) − 1)`で、正が悪化、負が改善を表します。

| モデル | 学習データ：ゼロ予測比 | 独立データ：ゼロ予測比 | 独立データの95%信頼区間 |
| --- | ---: | ---: | --- |
| pivot | −41.3059% | **+17.2513%** | [+15.4701%, +19.0324%] |
| mixture | −1.6521% | **+0.6599%** | [+0.4743%, +0.8455%] |

ただし「予測に信号が全くない」という意味ではありません。独立mixtureの`E_w[f·d]`は`0.000133964`、95%信頼区間は`[0.000096479, 0.000171450]`で、弱い正の相関が残っています。MSEには出力スケールの誤差も含まれます。一方、RMとraw-scoreのargmaxは予測全体の正の一様倍率に不変なので、スケールだけを縮めても次節の行動選択は改善しません。

第1反復の教師には各行で`σ₁·d=0`という性質があります。そこで参考として共通ずれ`c=σ₁·f`を引いた`f−c`も集計しました。

その独立MSEのゼロ予測比もpivotは+13.5379%（CI [+12.0360%, +15.0397%]）、mixtureは+0.3698%（CI [+0.2368%, +0.5028%]）で、共通offsetだけでは問題を説明できません。これは第1反復だけの参考診断で、二乗誤差の直交射影や累積後悔の中心化アルゴリズムではありません。

## 2. 独立データでは選んだ行動も悪化する方向

教師は実際に選択した行動`A`と効用`Y`から`d_a=Y(1[a=A]−σ₁(A))`として作ります。診断方策`τ`の局所評価は`τ·d=Y(τ(A)−σ₁(A))`です。posterior重みを適用した値は、固定prefixの収集目的で、判断箇所だけ`τ`に変更し、その後を`σ₁`で続ける局所的な変更を評価します。**半荘全体を`τ`で打った場合の順位や勝率の推定ではありません。**

| 独立データでの方策 | pivot：平均 [95%信頼区間] | mixture：平均 [95%信頼区間] |
| --- | --- | --- |
| RMのgreedy | −0.002459 [−0.003741, −0.001178] | −0.002208 [−0.002872, −0.001545] |
| RMから確率的に選択 | −0.002461 [−0.003421, −0.001500] | −0.002318 [−0.002770, −0.001867] |
| raw-scoreのgreedy | −0.002140 [−0.003345, −0.000936] | −0.001713 [−0.002323, −0.001102] |

いずれも信頼区間が負で、RMから確率的に選ぶ局所評価でも改善を確認できません。学習データでは同じ指標が正なので、学習元だけでの改善を強さの根拠にはできません。

## 3. 全非正時の一様方策を変えても、大幅な悪化は残る

通常のRMは全予測が非正なら一様方策になり、greedy対戦では候補列の先頭を選びます。対照では凍結`σ₂`の`weights.bin`をそのまま複製し、signed raw-scoreの最大値を選びました。正の予測がある局面のgreedy選択は同じで、全非正時の選択だけが変わります。

診断用コピーだけを`role=AVERAGE`・`trainingState=false`にしてsoftmax後のargmaxを使います。これは**学習済み平均モデルではなく`RAW_SCORE_GREEDY_ABLATION`**です。重みのbyte一致とmetadataの変更範囲を確認しました。監査した全4集合でsoftmaxとraw-scoreのargmax不一致は0でした。

元のNanoを対戦相手とし、各条件2,048牌山×4席の8,192半荘を評価しました。平均順位は小さいほど良く、差の信頼区間は共通牌山4席を1組とした対応差です。

| モデル | 通常RMの平均順位 | raw-scoreの平均順位 | raw − RM | 対応差の95%信頼区間 |
| --- | ---: | ---: | ---: | --- |
| pivot | 3.930664 | 3.926392 | −0.004272 | [−0.012204, +0.003659] |
| mixture | 3.859863 | 3.830566 | −0.029297 | [−0.040976, −0.017617] |

初期方策の平均順位2.508057に対する悪化の大部分は残っています。独立データの全非正率はpivot 9.54%、mixture 8.50%。RMとraw-scoreの選択不一致は7.47%、6.60%ですが、RMと初期`σ₁`のgreedy選択不一致は85.18%、76.92%でした。

## 4. 行動種別と実装の確認

独立データの行動種別集計も保存しています。候補に鳴きがあるときの選択率（初期`σ₁` / RM / raw）はpivotで20.16% / 71.22% / 52.20%、mixtureで20.16% / 61.20% / 45.78%。和了可能時の見送り率はpivotで0.00% / 30.58% / 33.88%、mixtureで0.25% / 12.68% / 31.04%でした。いずれもposterior重み付きの記述統計で、各行動の良否や対戦中の発生率を直接証明するものではありません。

コード監査では、教師の正負、効用の席対応・root値の差し引き、候補IDとtensorの対応、重みの適用、合法行動のMSE、バッチの行順を確認しました。この範囲では符号反転、候補の入れ違い、重みの二重適用を特定していません。未検出の不具合がないとの断定はしていません。

初期化については[ネットワーク](../../src/main/java/com/epsilon/cfr/model/CfrNetwork.java)の後悔出力weight・biasがゼロで、[Nanoからの移植](../../src/main/java/com/epsilon/cfr/training/CfrWarmStartRunner.java)は表現部分です。初期`σ₁`は別途蒸留したAVERAGEから取得し、ゼロ出力のREGRETを256更新した結果だけで`σ₂`を作ります。Nanoの学習済み行動出力を保持しながら微調整している構造ではありません。この初期化・切替方式と、上記の汎化失敗を合わせると、初期方策の強さが失われた経路を説明できます。

教師の分散・標本数・表現・最適化それぞれの寄与は未分離です。結論は単一学習seed、2反復の実験中の最初の256後悔更新に限り、CFR一般の失敗を意味しません。CFR向けに新規学習した状態Vは未実施で、改善も未検証です。本番の既定値は変更していません。

## 検証と保存物

- `RegretScoreAuditTest`のJava 4テスト、`test_analyze_scores.py`のPython 8テストが成功しました。教師・argmax・重みの検査、cluster内の行分割に対するCI不変性、極小log重み、空局、比の共分散などを確認しています。
- `RegretScoreParity`は各モデル64行・複数bucketで、同一バッチのraw-scoreから復元したRMと通常APIを比較しました。最大差は両モデルとも`1.11e-16`、greedy不一致は`0/64`です。更新数は前後とも256でした。
- 単行推論とバッチ推論は完全一致しません。BF16のバッチ形状による数値差を含み、両モデルともgreedy不一致は`2/64`でした。raw-score最大差はpivot `0.000976563`、mixture `0.000244141`です。同一バッチのAPI一致と、異なるバッチ形状の一致は分けて扱っています。
- 新規収集1、raw-score対戦2、行監査4、Python集計4、parity 2の全13工程が終了コード0でした。独立の集計実装で全4 gzipを読み直し、主要なMSE比・差・局所評価のepisode-cluster CIが`1e-9`以内で一致しました。

取得先は[`target/remote-training/regret-diagnosis-evidence`](../../target/remote-training/regret-diagnosis-evidence/)です。`manifest.json`に条件、`comparison.json`に対戦比較・全4集計・工程結果、`audits/*.json`にJavaの回帰・行動種別集計、`audits/*.rows.jsonl.gz`に行データ、`analyses/*.json`にcluster CIを保存しています。対戦の牌山別記録は`results/`、照合は`parity/`、実行引数と終了コードは`control/`、テストと実行ソースは`logs/`・`source/`です。`collect_evidence.py`はcompactな結果と行データを別archiveにし、checkpointと符号化済み観測は含めません。

## 再現（履歴ソース用）

保存gzipからの再集計はGPU不要で、Python標準ライブラリだけで実行できます。`train`を`heldout`、`pivot`を`mixture`に替えて全4集計を再現できます。出力先は元の結果と分けます。

```bash
python3 epsilon-cfr/experiments/regret-diagnosis/analyze_scores.py \
  /path/to/evidence/audits/heldout-pivot.json.pivot.rows.jsonl.gz \
  --episode-count 24576 --episode-start 0 --partition heldout \
  --output /path/to/recomputed/heldout-pivot.json

python3 -m unittest discover \
  -s epsilon-cfr/experiments/regret-diagnosis -p 'test_*.py'
```

推論・収集から再実行する場合は元trialのcheckpoint・保存観測、Nano/GRP、実験用Javaクラスが必要です。Linux / Java 21 / 当該DJL・ROCm環境の実行例です。

```bash
python3 epsilon-cfr/experiments/regret-diagnosis/prepare_diagnosis.py \
  /path/to/cfr-pivot-reuse /path/to/cfr-policy-diagnosis \
  /path/to/new-regret-diagnosis

# 当該runのコンパイル済みクラスと依存jarを含むclasspathを使用する。
java -cp "$(cat /path/to/run/control/audit-classpath.txt)" \
  RegretScoreAudit /path/to/run/control/audit-heldout-pivot.json

python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py \
  /path/to/new-regret-diagnosis/control/holdout-plan.json
python3 epsilon-cfr/experiments/pivot-reuse/run_stages.py \
  /path/to/new-regret-diagnosis/control/raw-plan.json
```

`prepare_diagnosis.py`が生成するのは独立収集・raw-score対戦の計画です。4件のaudit設定は保存済み`control/audit-*.json`を使い、`inputDirectory`とcheckpointを各arm・各partitionに合わせます。監査クラスのソースは`RegretScoreAudit.java`・`RegretScoreParity.java`、対戦は`RawScoreDuel.java`で、監査にはバッチ版`CfrModel.rawScores(List<byte[]>)`を含むビルドを使います。全工程の正確なclasspath・引数は保存済み`control/*-plan.json`が基準です。別の出力先で再実行するときは、設定内の絶対パスも変更します。工程runnerは既に正常終了した同一引数の工程をスキップします。
