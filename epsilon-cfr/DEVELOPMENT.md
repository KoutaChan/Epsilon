# Epsilon-CFR 開発ガイド

実行手順と設定例は[README](README.md)を参照してください。

## 共通基盤と責務

| 責務 | 実装元 |
| --- | --- |
| 合法手・局進行・精算・順位 | `epsilon-common` の `GameEngine` |
| 順位効用・得点効用・GRPの期待効用への変換 | 共通 `EpsilonUtilityProfile` / `EpsilonUtilityTargets` |
| 現在局面の特徴量・候補スラブ・bucket・デバイス転送 | Nanoの `decision/input` |
| 手牌・公開情報・プレイヤー読み・候補ごとの表現 | Nanoの状態エンコーダ・候補エンコーダ |
| 入力行の保存形式 | Nanoの `DecisionInputRowCodec` |
| 並列学習・勾配集約・AdamW・複製同期 | `epsilon-common` の `DataParallelGroup` |
| 推論要求の予約・収集batch・CPU/GPU実行・終了待機 | 共通 `runtime` |
| CLI・コマンド登録・引数解析・実行結果と終了コード | 共通 `workflow` |
| TOML・設定上書き・型付き設定の生成と検証 | 共通 `config.settings` |
| 限定した分岐比較の有効化・収集予算 | 共通 `DecisionBranchComparisonSettings` |
| 学習・推論・GRPのデバイス指定 | 共通 `DeviceSettings` |
| デバイス指定の解釈・自動選択・可視GPU番号の検証 | 共通 `NetworkDeviceResolver` |
| 解決済みデバイスの順序と主デバイスの保持 | 共通 `NetworkDevices` |
| ネイティブエンジン起動前のCPUスレッド既定値 | 共通 `NetworkRuntime` |
| 4席rotation・対局集計・2段階の昇格判定 | 共通 `EpsilonDecisionDuelArena` / `EpsilonDecisionWallDuelEvaluator` |
| 局境界の順位分布とその学習・保存 | Nanoの `EpsilonGrpTrainingSession` |
| 後悔教師・補正済み平均方策・重み付き再生・solveの固定境界 | CFR固有 |

本番の経路は、`CfrCampaign`の学習区間制御、`CfrIterationRunner`の反復工程、`CfrRunStore`の保存・公開に分けています。モデルは`CfrModel`が重みと実行資源を所有し、`CfrInputEncoder`がNanoのバッチ構築器によるCPU入力生成、`CfrLearner`が目的関数と更新を担当します。設定は`CfrConfig`へ共通`SettingsLoader`から直接bindし、別の設定DTOへ変換しません。

収集器は半荘ID・方策・境界評価・`CfrEpisodeSink`を受け取り、完了した局経路を順次渡します。実験用の方策session、全経路をメモリに集める入口、旧対戦用の席rotation設定は持ちません。対局時の席rotationは共通duelに委ねます。Pivot、scalar baseline、snapshot混合方策の実験実装は本番ソースから削除し、過去の結果だけを`experiments`に残しています。

CFRは各意思決定の全合法手順序、凍結方策、実際の探索確率、Nanoと同じ盤面入力を保存します。状態と候補の表現にはNanoの実装を使い、後悔モデルと平均方策モデルの重みは独立させています。CFR専用の行動履歴や局境界系列はモデルへ入力しません。

Nanoからの初期化でも、教師の推論に使った入力バッチから保存用の観測を作ります。同じ局面の特徴量を作り直さず、候補順序と入力内容をそのまま蒸留に引き継ぎます。

`NetworkDeviceResolver`は設定文字列から`NetworkDevices`を生成します。`NetworkDevices`はデバイスを選び直さず、解決済みの集合を不変に保持します。CFRのCampaign・推論fleet・モデル・並列learnerもこの共通型を受け渡し、学習・推論・GRPをそれぞれの指定で解決します。

収集・学習・評価・保存世代の整理はそれぞれ独立したクラスが担当します。QやOracleは追加しません。NanoのPPO clip・Actor KL学習率制御・Actor/Value損失は、LCFRの符号付き後悔回帰と補正済み平均方策の学習とは目的が異なるためコピーしません。学習率・batchはNanoの教師学習を基準にし、実行基盤と適用可能な設定を共有します。

learnerは1回の抽出結果を`CfrSampleId`で集約し、同じ標本の出現回数を行損失へ掛けます。分母とログのsamplesは集約前のbatchサイズを使い、clip・AdamW更新は1回です。replayの抽出確率を損失に再乗算しません。候補bucketごとにmicrobatchを作り、候補形状を含む推定仕事量でGPUへ配分します。

collectorの特徴生成は共通`PolicyExecutionContext.encodeRows`、エンジン初期化と行動適用は共通`ArenaAdvanceExecutor`を使います。GRP境界評価も一括化します。借用観測を使う全batchが終了してからエンジンを進め、同時応答を選ぶ情報の境界を維持します。

収集時の受付は単遷移・複数遷移の2区分とし、同じ行数上限を使う行動容量の異なる入力をまとめます。入力エンコーダー内では候補bucketごとにNanoの`DecisionBatchBuilder`へ直接書き込みます。入力の計画・確保・特徴生成を共通CPUプールで実行し、全ワーカーの書き込みが終わってからhostを確定します。GPUへはNanoの推論用転送を使い、実在する遷移だけを転送します。1行ずつhostを作成してから連結し直す処理は行いません。合法手が一つの行も観測は保存し、GPUの方策計算だけを省略します。

保存payloadは確定済みhostの行スライスから作り、GPUの演算と並行して生成します。Nanoの`DecisionInputRowCodec.EncodedRow`が実データと未使用遷移のゼロ領域の位置を独立して所有し、入力配列は保持しません。保存・復号時にはゼロを順次補い、従来の標準形式と同じバイト列を返します。固定長入力を符号化する場合は、未使用セルも含めて元のビット列を保存します。GPUの出力は独立した方策配列へ移してから実行枠を解放し、保存payloadと方策の両方が揃った時点で判断を返します。

推論では、保存用の候補bucketとデバイス上の実行単位を分けます。Nanoの`DecisionGroupedInferenceLayout`が複数bucketの盤面と実在遷移を一つの連続バッファへ配置し、盤面表現・牌文脈・遷移表現をまとめて計算します。遷移の集約は元のbucketで行い、一行動一遷移の直接配置と複数遷移の重み付き集約を維持します。その後、合法候補だけを集めて候補射影・正規化・CFRの出力層を計算し、元の入力順へ戻します。盤面射影は盤面ごとに一度計算して共有します。異なるbucketを大きな固定形状へ広げる処理は行いません。

`CfrEpisodeWriter`は共通の待機キューから書き込み可能な局を選び、空いたワーカーへ渡します。同じ半荘の局は受け付けた順に一つずつ保存し、別の半荘は並列に進めます。4ワーカー・待機中と処理中を合わせた16件の上限、半荘完了時のfsyncと原子的な公開を維持します。

Nanoの`DecisionInferenceStoragePool`は、入力の状態・行動・遷移配列を受付予約の単位で貸し出します。GPU転送と保存payloadの読み取りが両方完了するまで返却せず、失敗・呼び出し側のcancel・終了処理でも全ワーカーの完了を待ちます。CFRが保持する待機中の配列は256MiBを上限とし、実行中の入力数は共通の受付予約とデバイス枠で制限します。通常の単独入力APIは引き続き独立した配列を所有します。

凍結モデルの候補表現では、Nanoの`PackedTransitionProjection`へ結合済みの射影重みを保持します。同じ盤面の状態を各遷移へ複製してから射影せず、盤面ごとに一度射影してから必要な行を取り出します。CFRのFP16・BF16ともバイアス付きLinear、状態加算、SiLUの順と丸めの境界を維持します。学習時の計算と保存パラメーター名は変更しません。

凍結GPUの遷移特徴はNanoの`embeddingFeaturePack`で埋め込みと数値特徴を連結し、状態加算とSiLUも共通の所有済みバッファ用演算へまとめます。融合演算の形状と型の条件を満たす場合だけ使用し、CPU・学習・型が一致しない入力は通常の計算を使います。性能測定と未検証の条件は[収集最適化の記録](experiments/collection-100k-20261009/README.md)に分けて記載します。

学習・収集・Reviewer・共通duelは同じ盤面入力を使います。GRU、履歴専用ネットワーク、履歴の転送や隠れ状態のcacheはありません。局開始分布やGRP評価に必要な公開局境界は、モデル入力とは別に収集器が保持します。

盤面特徴では区別できない過去の行動経路は同じ入力へ集約されます。そのため、完全な本人履歴を情報集合とする完全記憶の表形式CFRとは異なる近似です。経路上の到達確率・接尾辞補正・平均方策の重みは引き続き実際の収集経路から計算します。

## 学習方式

全4席のOutcome Sampling LCFRで学習します。反復 `t` の収集方策は凍結し、合法手上一様の探索方策と混合します。元の方策の確率0は保存します。

`epsilon.cfr.utilityProfile`を収集・GRP境界・分岐・duelの集計と信頼区間へ渡します。commonの計算を使い、CFRでは倍精度の順位効用を使用します。GRP校正の教師はCollectorが返す実順位から生成し、効用値から順位を逆算しません。現在のGRPとduelは順位効用を前提とするため、`SCORE`はCFR設定で拒否します。

後悔教師は符号を保った `Y × (1[a = A] - sigma(A))`、サンプル重みは `t / N_t × kappa` です。重みをlog空間の抽出分布へ反映し、損失へ二重に掛けません。合法手の二乗誤差は手数で割らず合計し、batchのサンプル数で平均します。

通常のOutcome Samplingに加えて、`collection.branchComparison`でRON見逃しと九種九牌の二択だけを比較します。RONは見逃しを選んだ場合に和了側の追加枝を作り、九種九牌は宣言から続行、続行から宣言の両方向を対象にします。追加枝は通常の方策と同じ評価境界まで進め、`HAND_GRP`では局末の固定GRP、`MATCH_MC`では半荘の最終順位を使います。主枝と追加枝で同じ開始時効用を差し引きます。追加枝の結果は対象判断の後悔教師へ反映し、独立した通常エピソードとして登録せず、平均方策の教師にも追加しません。1反復32,768半荘の通常収集予算は維持します。

分岐設定には共通`DecisionBranchComparisonSettings`を使い、CFRの同梱TOMLで`enabled`、`winDecline`、`kyushu`を有効にします。比較の採用は主枝の行動を抽選する前に決めます。`maxBranchesPerGame`（既定1）は採用した比較の数で、RON和了や九種九牌の対象外のツモ和了を選び、追加枝を作らない場合も1回を消費します。`maxInFlightBranches`（既定8）は比較の予約枠を制限し、追加枝が不要なら行動確定時、追加枝がある場合はその完了時に解放します。1つの同時応答境界では比較を1件だけ採用します。

`maxExtraInferenceRatio`（既定0.01）は新しい枝を開始するための予算判定です。開始済みの枝は途中で打ち切らないため、完了時には比率を超える場合があります。枝の結果や所要推論数を見て教師を採用・破棄すると推定が偏るため、予算は開始を抑えるために使います。Campaignの通常収集だけに設定を渡し、Nano初期蒸留・GRP校正などの収集では有効にしません。

`CfrBranchSelection`は、主枝で行動を選ぶ確率`mu(a)`とは別に、その行動を主枝または追加枝で評価する包含確率`q(a)`を求めます。宣言手を`c`、続行手の集合を`D`、両者の集合を`G`とし、`mu(D)`と`mu(G)`は各集合の確率和です。

| 比較 | 宣言手の包含確率 | 続行手`a`の包含確率 |
| --- | --- | --- |
| RON | `q(c) = 1` | `q(a) = mu(a)` |
| 九種九牌 | `q(c) = mu(G)` | `q(a) = mu(a) × mu(G) / mu(D)` |

RONでは主枝が和了を選べばその結果を使い、見逃しを選べば追加枝で和了を評価するため、採用した比較の`q(c)`は常に1です。主枝が和了した場合も、この補正を保存します。九種九牌では主枝が宣言した場合の続行手を`mu(a) / mu(D)`で1手だけ抽選します。ツモ和了は`G`に含めず、選択された場合は通常のOutcome Sampling行を使います。

`LcfrTargets`は、実際に評価した各行動`a`について次の寄与を計算します。`sigma`は凍結した探索前方策、`h`は分岐前の履歴、`u_a`はその枝の効用です。接頭辞の`pi_-i_sigma(h)`は本人以外の方策確率積、`pi_mu(h)`は全席の収集確率積です。環境の偶然確率は比から相殺されます。

```text
rho_a = product_after_root(sigma(selected) / mu(selected))
w_a = (t / N_t) × pi_-i_sigma(h) / pi_mu(h) × rho_a / q(a)
T_a[b] = u_a × (1[b = a] - sigma(a))
W = sum_a(w_a)
T[b] = sum_a((w_a / W) × T_a[b])
```

主枝と追加枝の`rho_a`はそれぞれの経路から計算します。追加枝の強制した根の行動は積に含めません。同時応答で再利用した他席の選択のうち、根より大きい判断IDの確率比は追加枝の接尾辞に含め、根より小さいIDは主枝と共通の接頭辞で扱います。

`CfrTrainingRow`には元の判断IDのまま`log(W)`と明示的な後悔ベクトル`T`を1行保存します。`W`と混合比はlog-sum-expで計算し、全寄与が0なら後悔の抽出重みを0にします。この統合は、通常のOutcome Samplingが持つ重みの合計と重み付き教師の両方を期待値で保ちます。元の後悔行を重ねて登録しません。平均方策の教師と抽出重みは主枝から計算した値をそのまま保持します。

平均方策は独立した補正済みreservoirから全合法手の分布を学習します。GRP校正と配布モデルの対局では平均方策から確率的に選択します。昇格評価はNanoと同じgreedy選択です。

- `HAND_GRP`: 東1局から半荘終局まで進め、局ごとに経路を区切ります。局末のGRPはsolve内で固定し、真の半荘終局だけ実順位効用を使います。初期局面のGRPは収集開始時に一度だけ計算し、局末値は次局の中心化基準へ再利用します。GRP未指定時は一様確率方策の半荘から初期校正し、以降のsolveでは採用済み平均方策の通常対局から次のGRPを校正します。
- `MATCH_MC`: 半荘最終順位を直接使います。半荘途中で教師の計算に使う経路の確率積をリセットしません。

両方式の収集予算は`collection.games`で指定する実半荘数です。`HAND_GRP`の局開始状態は現在方策の到達分布に従い、旧方式のsolve固定RootBankから一様に抽選する分布とは異なります。局内の重要度比は各局で初期化し、反復の重み`t/N`の分母には収集済みの実軌跡総数を使います。`HAND_GRP`では実局数、`MATCH_MC`では半荘数です。固定root分布の方式と同じ更新であるとは扱いません。
- 再生: 後悔・平均方策ごとに一様reservoirを持ち、ディスクpayloadと上限付きcacheを共有します。保存量が設定上限を超えると停止します。重みの大きい行だけを残す処理はしません。

`CfrDuelSession` は `DuelEvaluationSource` を実装し、2つの平均方策を一度だけ読み込みます。`CfrDuelPolicy` は既存の推論fleetを借用して盤面と合法手を入力へ変換し、最大確率の合法手位置を返します。対局実行・4席rotation・集計はNanoと同じ共通duelへ委譲します。

判定は共通 `EpsilonDecisionWallDuelEvaluator` の `TWO_LOOK_ALPHA_SPENDING` を使い、`PROMOTED` の場合だけchampionを更新します。`DecisionChampionDuelSettings`、`DecisionEvalVsSettings`、`DecisionDuelArenaSettings`、`InferenceBatchingSettings` をそのまま設定型として使います。NanoのPPO campaignと異なり、`HARMFUL`でもLCFRの累積solve自体は止めません。

平均方策の学習と評価は`evaluation.every`（既定10）と各solveの最終反復に連動します。教師と平均化の重みは毎反復replayへ登録し、評価予定の反復だけ平均方策を学習します。学習・評価の実行判定は`CfrConfig.evaluatesAt`を共有します。完了した判定はchampion参照を書き換える前にjournalへ保存し、再開時に同じ判定を適用します。評価中断時は凍結モデルと同じseedから評価を再実行します。

## CLI

[`Main`](src/main/java/com/epsilon/cfr/Main.java)はNanoと同じ共通 `StartupSettings` で設定ファイルと上書き値を選び、`EpsilonSettings` を通じて `SettingsLoader` の不変スナップショットを作ります。共通 `CommandLine` に設定とコマンドを渡し、[`Workflows`](src/main/java/com/epsilon/cfr/Workflows.java)が6つの `CommandWorkflow` を登録します。

設定ファイルは [`cfr/settings.toml`](src/main/resources/cfr/settings.toml) を既定値とするTOMLです。キーは `epsilon.cfr.*` で、Nanoと同じ `--settings`、`-Depsilon.*`、`EPSILON_SETTINGS` を使います。`train --run DIR` が型付き設定を生成して学習を開始します。`--config` や `--nano-checkpoint` による別系統の設定読み込みはありません。Nanoからの初期化は `epsilon.cfr.initialization.nanoCheckpoint` に指定します。

`init-config --out FILE` は同梱TOMLを任意の新規ファイルへコピーします。通常は初回起動時の `config/epsilon-cfr.toml` 自動生成を使えます。`status`、`verify`、`resume`、`export` は保存runを対象とし、現在の起動設定を学習設定としてバインドしません。

終了コードは成功・ヘルプが `0`、引数／実行エラーが `1`、未知のコマンドが `2` です。実行の開始・完了・失敗は各系列と同じ共通ログに記録します。

既定のコンソール出力にはJSON結果とログの両方が含まれます。JSONのみを取得する場合は、Logbackのコンソール出力先をstderrへ変更するか、`Main.run(args, output)` に結果用のストリームを渡してください。

## 保存と復旧

再開点は収集・再生登録・後悔学習・平均方策学習・評価の工程境界と、指定した更新間隔です。未完了の収集は固定IDとseedで補完し、optimizerと再生世代を同じ状態から戻します。途中まで書き込まれたファイルは公開しません。

runの `config.json` は適用済み学習設定の永続化形式として維持します。現行形式のrunを `CfrConfig.read` で復元し、再開時のTOML・プロパティ上書きを混ぜません。保存済み `config.json` は直接編集しないでください。

`CfrConfig.read`は現行の設定JSONを検証して読み込みます。廃止した設定キーや旧形式は受け付けず、自動補完・変換は行いません。すべての設定セクションを保存します。境界モデルとNano初期化を使わない場合はcheckpointを空文字にし、初期化の有無は`Initialization.enabled()`で判定します。

実行状態はversion 2、`CfrMatchFile`の半荘保存形式はversion 1です。局の圧縮経路は`CfrMatchFile`、再生行と再生snapshotはpackage-privateの`CfrReplayCodec`が扱います。単独の局ファイルや行配列ファイルを読み書きする旧APIはありません。エピソードには分岐の包含確率・追加枝の効用・接尾辞重みを、再生行には任意の明示的後悔ベクトルを保存します。`model.json`はversion 3、配布manifestはversion 2で、どちらにも効用設定を保存します。評価・配布時にはモデルと設定の効用一致を確認します。旧バイナリ・モデル・配布形式、および`utilityProfile`や`collection.branchComparison`を持たない旧run設定は読み込まず、自動補完・変換は行いません。

モデル・optimizerの内容hashは再開時と明示的な `verify` で検証し、通常の工程遷移では参照先の存在だけを確認します。配布モデルは読込時に一度検証します。設定・公開ポインタの小さなhashと、重複登録を防ぐepisode識別は常時確認します。

再生の通常サンプリングでは行ごとのSHA-256照合を行わず、形式・サイズ・行ID・重みの対応を検査します。保持中の全行の内容検証は `verify` で実行できます。

局経路はZstdで個別に圧縮し、同じ半荘の一時ファイルへ局順に追記します。writerは4本・待機中と処理中を合わせて16経路まで受け付け、同じ半荘を同じworkerへ割り当てます。ファイルは局の追記ごとに閉じ、半荘終局で一度だけfsyncとatomic公開を行います。局ごとの観測は保存後に解放します。未完成の`.part`は再開点にせず、その半荘を先頭から再収集します。再生登録を公開した後は、現在solveの登録済み半荘データを削除し、再生データは現在と直前の保存世代を残します。参照されなくなったpayload segmentも削除します。モデルと他solveは削除しません。

復旧は現在の公開点から行います。古いstate JSONだけによる手動復元は保証しません。

## 検証

通常テストは全木の反事実価値照合、平均方策補正、確率0、reservoir、破損検出、盤面入力、局境界、実行状態の公開を検証します。`native` グループでは実CPUでモデルの学習・保存・optimizer再開・配布と、局／半荘の学習全体を検証します。

共有化前のNanoから保存したcheckpointと32局面の出力も照合し、既存パラメータ名と推論結果の互換性を検査します。CFRの候補集約への勾配も実際の重み更新で確認します。

`CfrMultiplicityLearningTest`は展開したbatchとの損失・勾配・AdamW更新を比較します。入力の符号化・復号と、通常経路・staged経路の出力一致を検証します。GPUのBF16・pinned転送も`rocm`グループで検証します。実行基盤の速度比較では同じ保存モデル・入力標本・開始局面・seedを使い、収集は圧縮保存まで、learnerは保存時間の有無を明示して別々に測定します。

リポジトリのルートから実行します。

```powershell
.\mvnw.cmd -pl epsilon-cfr -am test
.\mvnw.cmd -pl epsilon-cfr -am '-Dtest=NanoFeatureSpineCompatibilityTest,CfrNativeModelTest,CfrCandidateLearningTest,CfrDeploymentTest,CfrCampaignTest,CfrWarmStartLifecycleTest,CfrDuelSessionTest,DataParallelGroupTest' '-Dsurefire.failIfNoSpecifiedTests=false' '-Dsurefire.excludedGroups=rocm,legacy' test
```

Linux・macOSではMavenのコマンドを `./mvnw` に置き換えてください。

学習が実行できることと対戦強度の向上は別に測定します。GPU性能と複数seedの長期強度は使用する学習機で確認してください。

GPU推論はNanoと同じデバイスパイプラインでH2D・compute・D2Hを実行します。実行枠ごとに転送バッファを所有し、scoresだけを回収します。再帰状態のcacheやhiddenの転送はありません。合法手が一つの判断はネットワークを省略し、保存する観測と判断数は変えません。

収集は共通の`InferenceAdmission`と`ArenaAdvanceExecutor`を使い、入力生成・推論・対局進行を完了イベントで接続します。同時応答では、その盤面を参照する全入力の推論が完了するまでエンジンを進めません。終了時と失敗時は実行中の処理を回収してからモデルを閉じます。

完了したepisodeは4本の保存ワーカーへ渡し、待機中と保存中の合計を16件に制限します。ファイルごとのZstd圧縮・fsync・atomic公開を維持し、すべての保存が成功するまで収集工程を完了扱いにしません。保存失敗は収集側へ伝播し、未保存の固定IDは再開時に収集し直します。収集速度には保存ワーカーの終了待ちも含めます。

保存先への受け渡しは`CfrEpisodeSink`の非同期な空き通知へ接続します。満杯のときは完了したepisodeをcollectorの既存の同時進行枠に残し、新しい対局の補充を抑えます。主スレッドで保存枠を待たず、進行中の対局・推論・追加枝の完了を処理し続けます。collectorの保持数とwriterの待機・実行件数をそれぞれ制限し、無制限の保存キューは作りません。

観測のバイナリ表現は`ObservationPayload`が所有します。外部の変更可能な配列を受け取る境界ではコピーし、確定後は予測・判断・教師・保存の間で同じ不変payloadを共有します。復号とストリーム書き込みは生配列を公開せずに行います。

learnerは抽出済みの重複を出現回数へまとめ、候補bucketごとのmicrobatchを処理します。抽選分布、教師、更新ごとの全標本数による正規化、optimizerの更新回数は変更しません。

方策モデルの構造IDは盤面入力だけを使う構造を表します。旧履歴モデルの重みや観測payloadは読み込み時に拒否します。Nanoの共有feature spineと候補表現の初期化は維持しています。構造変更の比較では同じNano表現から初期化し、速度改善と学習品質は別に検証します。

## 性能測定

盤面入力への統一と非同期保存の変更前後は、[2026-10-09の測定結果](experiments/board-optimization-20261009/README.md)を参照してください。

半荘単位への変更とNanoのバッチ構築器を直接使う最適化は、[半荘収集の比較](experiments/hanchan-optimization-20261009/README.md)にまとめています。学習側は[microbatch上限の比較](experiments/microbatch-20261009/README.md)と[追加候補・保存処理の診断](experiments/learner-refinement-20261009/README.md)を参照してください。

測定用のエントリポイントはテスト側に置き、配布CLIには含めません。入力モデルを更新せず、新しい出力ディレクトリへ適用設定・計測範囲・結果を保存します。`SETTINGS`は通常のTOMLで、`-`は同梱設定です。GPU測定では対応するROCmまたはCUDAのDJLランタイムを使ってください。

```bash
./mvnw -pl epsilon-cfr -am -DskipTests package
java -cp 'epsilon-cfr/target/test-classes:epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar' \
  com.epsilon.cfr.benchmark.CfrPerformanceBenchmark \
  collect SETTINGS CHECKPOINT GRP_CHECKPOINT OUTPUT 256 true
java -cp 'epsilon-cfr/target/test-classes:epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar' \
  com.epsilon.cfr.benchmark.CfrPerformanceBenchmark \
  learn SETTINGS CHECKPOINT EPISODES_DIR OUTPUT 8 128 weighted
java -cp 'epsilon-cfr/target/test-classes:epsilon-cfr/target/epsilon-cfr-1.0-SNAPSHOT-all.jar' \
  com.epsilon.cfr.benchmark.CfrPerformanceBenchmark \
  inspect MATCHES_DIR OUTPUT 32
```

Windowsではclasspathの区切りを`;`に置き換えます。収集半荘数は`collection.games`を使います。収集はwarmupを除き、通常の分岐比較と圧縮保存まで含めます。learnerは同じ保存済みエピソードから専用replayを準備し、準備時間を分離して、抽選・展開・転送・順逆伝播・optimizer・定期checkpoint保存とGPU完了待ちを測定します。最後の二つの数値はwarmup更新数と測定更新数です。

`inspect`は保存済み半荘をファイル名順に指定件数だけ読み、合法手数・実遷移数・容量区分・未使用領域の内訳を集計します。モデルやGPUは使わず、元データも変更しません。この集計は入力分布の診断であり、収集速度の測定ではありません。

learnerは抽選された論理サンプル数と、同じ更新内の重複をまとめた実処理行数を別々に報告します。最後の引数を`uniform-distinct`にすると、保持行全体から更新内で重複なく抽選する計算量の診断になります。これは通常のLCFR抽選分布を変更する測定専用モードで、学習設定には反映しません。候補形状・抽選重みの実効サンプル数も結果へ記録します。

Nanoの比較対象は`com.epsilon.nano.ai.decision.benchmark.NanoPerformanceBenchmark SETTINGS CHECKPOINT_ROOT OUTPUT GAMES WARMUP_PASSES MEASURED_PASSES SEED`です。`epsilon-nano/target/test-classes`とNanoのall.jarをclasspathへ指定します。本番の収集・保存・Actor/Value learnerを使い、学習では入力展開・転送・optimizer・KL検査・GPU完了待ちを含めます。モデル書き出しは含めず、拒否された更新を処理済み行数へ数えません。

Nanoの収集ログの`decisionsPerSecond`は全評価席への推論要求数です。CFRの通常判断数、Nanoの1席分の保存行数、局数と半荘数を同一の指標として比較しないでください。性能結果は学習品質や対戦強度の改善を示すものではありません。

[使い方へ戻る](README.md)
