package com.epsilon.pico.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.pico.ai.decision.EpsilonDecisionConstants;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.pico.ai.decision.input.DecisionInferenceDeviceBatch;
import com.epsilon.pico.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.pico.ai.decision.input.DecisionInputLayout;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.pico.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.pico.ai.decision.input.DecisionStateInputs;
import com.epsilon.pico.ai.decision.policy.DecisionAlternative;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyExecution;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.pico.ai.decision.policy.EpsilonDecisionPolicyHead;
import com.epsilon.pico.ai.model.fusion.EpsilonMahjongStateReadoutGroupExecution;
import com.epsilon.pico.ai.model.fusion.EpsilonPlayerMemoryFusionExecution;
import com.epsilon.pico.ai.model.fusion.EpsilonStrategicContextFusionExecution;
import com.epsilon.pico.ai.model.fusion.EpsilonTileRelationFusionExecution;
import com.epsilon.pico.config.settings.DecisionSettings;
import java.util.Objects;

/**
 * 動的な合法手の方策と、HL-Gauss による期待効用を予測する Decision ネットワーク。
 *
 * <p>方策と価値関数は状態エンコーダーを共有し、特徴量を局面全体へ集約する部分は分離する。共有エンコーダーは方策の更新対象とし、価値関数へ渡す表現では勾配を切り離す。価値損失は共有表現や方策を更新しない。
 *
 * <p>方策だけ、価値だけの専用経路では不要な計算を省く。両方を求める場合は、共有する状態表現を一度だけ計算する。
 */
public final class EpsilonDecisionNetwork extends AbstractBlock {

  /** 設定から解決した標準隠れ層埋め込み幅。 */
  public static final int DEFAULT_HIDDEN = DecisionSettings.defaults().hidden();

  private static final int POLICY_READOUT_MAXIMUM_FEED_FORWARD_WIDTH = 384;
  private static final int VALUE_READOUT_MAXIMUM_FEED_FORWARD_WIDTH = 128;

  private final int hiddenSize;
  private final EpsilonUtilityProfile utilityProfile;
  private final float[] utilityWeights;
  private final EpsilonMahjongStateEncoder stateEncoder;
  private final EpsilonMahjongStateReadout policyStateReadout;
  private final EpsilonMahjongStateReadout valueStateReadout;
  private final EpsilonDecisionPolicyHead policyHead;
  private final Linear valueHiddenHead;
  private final Linear valueHead;

  /** 設定で指定された隠れ層の幅を使ってネットワークを構築する。 */
  public EpsilonDecisionNetwork() {
    this(DEFAULT_HIDDEN);
  }

  /**
   * 指定隠れ層の幅でネットワークを構築する。
   *
   * @param hiddenSize 状態・行動埋め込み幅。Attentionヘッド数で割り切れる正数
   */
  public EpsilonDecisionNetwork(int hiddenSize) {
    this(hiddenSize, EpsilonUtilityTargets.configuredProfile());
  }

  /** 複製モデルが入力元モデルと同じ効用定義を明示的に継承するコンストラクター。 */
  public EpsilonDecisionNetwork(int hiddenSize, EpsilonUtilityProfile utilityProfile) {
    if (hiddenSize <= 0 || hiddenSize % EpsilonMahjongStateEncoder.ATTENTION_HEADS != 0) {
      throw new IllegalArgumentException(
          "hiddenSize must be positive and divisible by "
              + EpsilonMahjongStateEncoder.ATTENTION_HEADS);
    }
    this.hiddenSize = hiddenSize;
    this.utilityProfile = Objects.requireNonNull(utilityProfile, "utilityProfile");
    EpsilonDecisionHlGauss.support(utilityProfile);
    this.utilityWeights = utilityProfile.rankUtility();
    stateEncoder = addChildBlock("stateEncoder", new EpsilonMahjongStateEncoder(hiddenSize));
    policyStateReadout =
        addChildBlock(
            "policyStateReadout",
            new EpsilonMahjongStateReadout(hiddenSize, POLICY_READOUT_MAXIMUM_FEED_FORWARD_WIDTH));
    valueStateReadout =
        addChildBlock(
            "valueStateReadout",
            new EpsilonMahjongStateReadout(hiddenSize, VALUE_READOUT_MAXIMUM_FEED_FORWARD_WIDTH));
    policyHead = addChildBlock("policyHead", new EpsilonDecisionPolicyHead(hiddenSize));
    valueHiddenHead =
        addChildBlock("valueHiddenHead", Linear.builder().setUnits(hiddenSize).build());
    Linear residualHead = Linear.builder().setUnits(EpsilonDecisionHlGauss.BIN_COUNT).build();
    residualHead.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0.0f));
    residualHead.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0.0f));
    valueHead = addChildBlock("valueHead", residualHead);
  }

  /**
   * ネットワークの隠れ層埋め込み幅を返す。
   *
   * @return 状態、候補、特徴量の集約が共有する埋め込み幅
   */
  public int hiddenSize() {
    return hiddenSize;
  }

  /** このモデルの生成時に固定した効用の定義。 */
  public EpsilonUtilityProfile utilityProfile() {
    return utilityProfile;
  }

  /**
   * 既定隠れ層の幅を含むチェックポイント用構造記述を返す。
   *
   * @return スキーマとネットワーク構成を固定する構造文字列
   */
  public static String architectureSummary() {
    return architectureSummary(DEFAULT_HIDDEN);
  }

  /**
   * 指定隠れ層の幅を含むチェックポイント・ログ用構造記述を返す。
   *
   * @param hiddenSize 記述へ埋め込む隠れ層埋め込み幅
   * @return スキーマとネットワーク構成を固定する構造文字列
   */
  public static String architectureSummary(int hiddenSize) {
    return EpsilonDecisionConstants.ARCHITECTURE_ID
        + " schema="
        + DecisionInputSchema.fingerprint()
        + " hidden="
        + hiddenSize
        + " state=round+player4+tile34+river96+meld16/canonical-tile"
        + "+tile-rel2h4-c64+player-local1h4-c64"
        + "+"
        + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
        + " towers=shared-policy-owned-memory+policy-readout-c64-f"
        + Math.min(hiddenSize * 2, POLICY_READOUT_MAXIMUM_FEED_FORWARD_WIDTH)
        + "+value-readout-c64-f"
        + Math.min(hiddenSize * 2, VALUE_READOUT_MAXIMUM_FEED_FORWARD_WIDTH)
        + "+value-detached"
        + "+hl-gauss-value101+frozen-grp-scalar-prior+decision-utility-bin-residual"
        + " policy=coherent-transition-attention+fused-key-value+fused-wait-yaku-score"
        + "+player-tile-safety+player-tile-context-residual-c64"
        + "+shared-discard+broadcast-aware-linear-fusion+binary-branch-gates-c64"
        + "+direct-alternative-offsets"
        + "+policy-consistent-candidate-context"
        + " graph=response{RON,DECLINE_RON{PASS,MELD{TYPE,CANDIDATE}}}"
        + "+turn{TSUMO,DECLINE_TSUMO{KYUSHU,CONTINUE{KAN{TYPE,CANDIDATE},"
        + "DISCARD{IDENTITY,RIICHI_VS_DAMA}}}} output=dynamic-bucket/node-full-support";
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape stateCategories =
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_CATEGORIES);
    Shape stateNumerics =
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS);
    stateEncoder.initialize(manager, dataType, stateCategories, stateNumerics);
    Shape entityMemory = new Shape(-1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, hiddenSize);
    policyStateReadout.initialize(manager, dataType, entityMemory);
    valueStateReadout.initialize(manager, dataType, entityMemory);
    policyHead.initialize(
        manager,
        dataType,
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.ACTION_CATEGORIES),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.ACTION_NUMERICS),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.TRANSITION_CATEGORIES),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.TRANSITION_NUMERICS),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.TRANSITION_TILES),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.WAIT_YAKUS),
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.WAIT_SCORES));
    valueHiddenHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
    valueHead.initialize(manager, dataType, new Shape(-1, hiddenSize));
  }

  /**
   * 方策と価値を一回の呼び出しで評価する。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param batch 状態、合法行動、遷移を持つ型付きデバイス側バッチ
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 方策グラフ用スコアと効用-区間価値ロジット
   */
  public EpsilonDecisionOutput forwardDecision(
      ParameterStore parameterStore,
      DecisionDeviceBatch batch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    DiagnosticOutput diagnostics =
        forwardWithDiagnostics(
            parameterStore,
            batch.inputs(),
            batch.playerMemoryPresentIndices(),
            batch.transitionPresentIndices(),
            training,
            runtimeParameters);
    return new EpsilonDecisionOutput(diagnostics.policyScores(), diagnostics.valueLogits());
  }

  /** ホストで有効要素の集約した逐次実行入力から方策と価値を推論する。 */
  public EpsilonDecisionOutput forwardInferenceDecision(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    return forwardInferenceDecision(parameterStore, batch, outputManager, null, runtimeParameters);
  }

  /**
   * ホストで有効要素の集約した入力を、位置-awareな方策推論実行境界とともに推論する。
   *
   * <p>入力テンソルの管理元は長寿命でもよい。順伝播中だけ入力を{@code outputManager}配下の一時有効範囲へ借用し、 返却テンソルだけを{@code
   * outputManager}へ残す。これにより固定再生でも反復回数一時テンソルが入力管理元へ蓄積しない。
   */
  public EpsilonDecisionOutput forwardInferenceDecision(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      DecisionInferenceExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    DiagnosticOutput diagnostics =
        forwardInferenceWithDiagnostics(
            parameterStore, batch, outputManager, inferenceForward, runtimeParameters);
    return new EpsilonDecisionOutput(diagnostics.policyScores(), diagnostics.valueLogits());
  }

  private EpsilonDecisionOutput forwardDecision(
      ParameterStore parameterStore,
      DecisionNetworkInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    DiagnosticOutput diagnostics =
        forwardWithDiagnostics(parameterStore, inputs, training, runtimeParameters);
    return new EpsilonDecisionOutput(diagnostics.policyScores(), diagnostics.valueLogits());
  }

  /**
   * 価値特徴量の集約を起動せず、共有状態エンコーダーと方策スコア計算処理だけを実行する。
   *
   * @param parameterStore パラメーター取得先
   * @param batch 状態、合法行動、鳴き後打牌を保持する型付きデバイス側バッチ
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 方策グラフが正規化する3系統の未正規化スコア
   */
  public DecisionPolicyScores forwardPolicy(
      ParameterStore parameterStore,
      DecisionDeviceBatch batch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardPolicy(
        parameterStore,
        batch.inputs(),
        batch.playerMemoryPresentIndices(),
        batch.transitionPresentIndices(),
        training,
        runtimeParameters);
  }

  /** 価値特徴量の集約を起動せず、ホストで有効要素の集約した逐次実行方策入力だけを実行する。 */
  public DecisionPolicyScores forwardInferencePolicy(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    return forwardInferencePolicy(parameterStore, batch, outputManager, null, runtimeParameters);
  }

  /**
   * 価値を起動せず、位置-awareな方策推論実行境界とともに方策を推論する。
   *
   * <p>入力テンソルは複製せず一時有効範囲へ借用し、返却する三系統のスコアだけを{@code outputManager}へ移す。
   */
  public DecisionPolicyScores forwardInferencePolicy(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      DecisionInferenceExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    try (NDManager scope = outputManager.newSubManager()) {
      temporarilyAttachInferenceBatch(scope, batch);
      DecisionInferenceInputs inputs = batch.inputs();
      EpsilonMahjongStateEncoder.EncodedState encodedState =
          encodePolicyInferenceState(
              parameterStore,
              inputs,
              batch.playerMemoryPresentIndices(),
              inferenceForward,
              runtimeParameters);
      DecisionPolicyScores scores =
          policyHead.scoreInference(
              parameterStore,
              encodedState,
              inputs,
              batch.transitionPresentIndices(),
              batch.policyExecutionIndices(),
              inferenceForward == null ? null : inferenceForward.policy(),
              runtimeParameters);
      outputManager.attachAll(scores.toNDList());
      return scores;
    }
  }

  private DecisionPolicyScores forwardPolicy(
      ParameterStore parameterStore,
      DecisionNetworkInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardPolicy(parameterStore, inputs, null, null, training, runtimeParameters);
  }

  private DecisionPolicyScores forwardPolicy(
      ParameterStore parameterStore,
      DecisionNetworkInputs inputs,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (!training) {
      EpsilonMahjongStateEncoder.EncodedState encodedState =
          encodePolicyInferenceState(
              parameterStore, inputs, playerMemoryPresentIndices, runtimeParameters);
      return policyHead.score(
          parameterStore, encodedState, inputs, transitionPresentIndices, false, runtimeParameters);
    }
    EpsilonMahjongStateEncoder.EncodedMemory memory =
        encodeMemory(
            parameterStore,
            inputs.stateCategories(),
            inputs.stateNumerics(),
            playerMemoryPresentIndices,
            training,
            runtimeParameters);
    EpsilonMahjongStateEncoder.EncodedState encodedState =
        memory.withStateEmbedding(
            policyStateReadout.read(parameterStore, memory, training, runtimeParameters));
    return policyHead.score(
        parameterStore,
        encodedState,
        inputs,
        transitionPresentIndices,
        training,
        runtimeParameters);
  }

  /**
   * 方策特徴量の集約を起動せず、共有状態エンコーダーと価値効用-区間出力層だけを実行する。
   *
   * <p>共有メモリは価値特徴量の集約の直前で勾配を切り離す。価値出力層と価値特徴量の集約だけが勾配を受け取る。
   *
   * @param parameterStore パラメーター取得先
   * @param batch 状態を保持する型付きデバイス側バッチ
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return {@code [batch,101]} の効用-区間ロジット
   */
  public NDArray forwardValue(
      ParameterStore parameterStore,
      DecisionDeviceBatch batch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardValue(
        parameterStore,
        new DecisionStateInputs(
            batch.inputs().stateCategories(),
            batch.inputs().stateNumerics(),
            batch.inputs().boundaryContext()),
        training,
        runtimeParameters);
  }

  /**
   * 合法行動をデバイスへ転送せず、価値専用状態入力から効用-区間ロジットを計算する。
   *
   * <p>価値出力用ネットワークは行動/遷移を参照しないため、このoverloadは通常の{@link DecisionDeviceBatch}経路と同じ状態
   * エンコーダーおよび出力層を実行しながら、価値のみ事前学習のホストからデバイスへの転送を最小化する。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param inputs 状態テンソルだけを持つ価値専用入力
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return {@code [batch,101]} の効用-区間ロジット
   */
  public NDArray forwardValue(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (!training) {
      return forwardInferenceValue(
          parameterStore, inputs, inputs.stateCategories().getManager(), runtimeParameters);
    }
    NDArray boundaryMarginals = detachedGrpBoundaryMarginals(inputs.boundaryContext());
    EpsilonMahjongStateEncoder.EncodedMemory memory =
        encodeMemory(
                parameterStore,
                inputs.stateCategories(),
                inputs.stateNumerics(),
                training,
                runtimeParameters)
            .stopGradient();
    NDArray stateEmbedding =
        valueStateReadout.read(parameterStore, memory, training, runtimeParameters);
    return forwardValueHeads(
        parameterStore,
        stateEmbedding,
        inputs.stateCategories(),
        boundaryMarginals,
        training,
        runtimeParameters);
  }

  /**
   * 価値専用入力を推論し、返却ロジットを明示した管理元へ所有させる。
   *
   * <p>固定再生の長寿命入力と、反復回数ごとに破棄する出力・一時テンソルの寿命を分離するための推論境界である。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param inputs 状態テンソルだけを持つ価値専用入力
   * @param outputManager 返却ロジットと順伝播一時テンソルを所有する管理元
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return {@code [batch,101]}の効用-区間ロジット
   */
  public NDArray forwardInferenceValue(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    return forwardInferenceValue(parameterStore, inputs, null, outputManager, runtimeParameters);
  }

  /** ホストで確定したプレイヤー-メモリインデックスを使い、GPUで存在行を再集計せず価値を推論する。 */
  public NDArray forwardInferenceValue(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      NDArray playerMemoryPresentIndices,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          inputs.stateCategories(), inputs.stateNumerics(), inputs.boundaryContext());
      if (playerMemoryPresentIndices != null) {
        scope.tempAttachAll(playerMemoryPresentIndices);
      }
      NDArray stateEmbedding =
          encodeValueInferenceState(
              parameterStore, inputs, playerMemoryPresentIndices, runtimeParameters);
      NDArray valueLogits =
          forwardValueHeads(
              parameterStore,
              stateEmbedding,
              inputs.stateCategories(),
              detachedGrpBoundaryMarginals(inputs.boundaryContext()),
              false,
              runtimeParameters);
      outputManager.attachAll(valueLogits);
      return valueLogits;
    }
  }

  /**
   * 価値専用状態入力から効用-区間ロジットと価値状態埋め込みを返す診断経路。
   *
   * <p>方策 trunkと候補スコア計算処理は実行しない。保存データによる検証で1候補行の方策統計を作らず、価値表現だけを監査するために使う。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param inputs 状態テンソルだけを持つ価値専用入力
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 効用-区間ロジットと価値特徴量の集約埋め込み
   */
  public ValueDiagnosticOutput forwardValueWithDiagnostics(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardValueWithDiagnostics(parameterStore, inputs, null, training, runtimeParameters);
  }

  /** 詰めたプレイヤー-メモリインデックスを再利用する価値専用診断経路。 */
  public ValueDiagnosticOutput forwardValueWithDiagnostics(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    EpsilonMahjongStateEncoder.EncodedMemory memory =
        encodeMemory(
                parameterStore,
                inputs.stateCategories(),
                inputs.stateNumerics(),
                playerMemoryPresentIndices,
                training,
                runtimeParameters)
            .stopGradient();
    NDArray stateEmbedding =
        valueStateReadout.read(parameterStore, memory, training, runtimeParameters);
    NDArray valueLogits =
        forwardValueHeads(
            parameterStore,
            stateEmbedding,
            inputs.stateCategories(),
            detachedGrpBoundaryMarginals(inputs.boundaryContext()),
            training,
            runtimeParameters);
    return new ValueDiagnosticOutput(valueLogits, stateEmbedding);
  }

  /**
   * 価値推論のロジットと状態埋め込みを明示した管理元へ残す診断経路。
   *
   * @param parameterStore 順伝播に使うパラメーターの取得先
   * @param inputs 状態テンソルだけを持つ価値専用入力
   * @param outputManager 返却テンソルと順伝播一時テンソルを所有する管理元
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 効用-区間ロジットと価値状態埋め込み
   */
  public ValueDiagnosticOutput forwardInferenceValueWithDiagnostics(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(
          inputs.stateCategories(), inputs.stateNumerics(), inputs.boundaryContext());
      ValueDiagnosticOutput output =
          forwardValueWithDiagnostics(parameterStore, inputs, null, false, runtimeParameters);
      outputManager.attachAll(output.valueLogits(), output.valueStateEmbedding());
      return output;
    }
  }

  /**
   * 両出力用ネットワークの出力に加え、それぞれの状態埋め込みを監査用に返す。
   *
   * <p>通常推論では埋め込みのデバイスからホストへの転送が増えるため、診断時だけ使用する。
   *
   * @param parameterStore パラメーター取得先
   * @param batch 状態と合法行動を保持する型付きデバイス側バッチ
   * @param training 学習固有のブロック挙動を有効にするか
   * @param runtimeParameters DJL ブロックへ伝播する実行時パラメーター
   * @return 方策・価値出力と両trunkの集約表現
   */
  public DiagnosticOutput forwardWithDiagnostics(
      ParameterStore parameterStore,
      DecisionDeviceBatch batch,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardWithDiagnostics(
        parameterStore,
        batch.inputs(),
        batch.playerMemoryPresentIndices(),
        batch.transitionPresentIndices(),
        training,
        runtimeParameters);
  }

  /** ホストで有効要素の集約した逐次実行入力から方策・価値と診断埋め込みを推論する。 */
  public DiagnosticOutput forwardInferenceWithDiagnostics(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      PairList<String, Object> runtimeParameters) {
    return forwardInferenceWithDiagnostics(
        parameterStore, batch, outputManager, null, runtimeParameters);
  }

  /**
   * 診断特徴量の集約を含め、位置-awareな方策推論実行境界とともに推論する。
   *
   * <p>入力は複製せず一時有効範囲へ借用し、方策・価値出力と診断埋め込みだけを{@code outputManager}へ残す。
   */
  public DiagnosticOutput forwardInferenceWithDiagnostics(
      ParameterStore parameterStore,
      DecisionInferenceDeviceBatch batch,
      NDManager outputManager,
      DecisionInferenceExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    try (NDManager scope = outputManager.newSubManager()) {
      temporarilyAttachInferenceBatch(scope, batch);
      DecisionInferenceInputs inputs = batch.inputs();
      InferenceReadouts readouts =
          encodeInferenceReadouts(
              parameterStore,
              inputs,
              batch.playerMemoryPresentIndices(),
              inferenceForward,
              runtimeParameters);
      DecisionPolicyScores policyScores =
          policyHead.scoreInference(
              parameterStore,
              readouts.policyState(),
              inputs,
              batch.transitionPresentIndices(),
              batch.policyExecutionIndices(),
              inferenceForward == null ? null : inferenceForward.policy(),
              runtimeParameters);
      NDArray valueLogits =
          forwardValueHeads(
              parameterStore,
              readouts.valueStateEmbedding(),
              inputs.stateCategories(),
              detachedGrpBoundaryMarginals(inputs.boundaryContext()),
              false,
              runtimeParameters);
      DiagnosticOutput output =
          new DiagnosticOutput(
              policyScores,
              valueLogits,
              readouts.policyState().stateEmbedding(),
              readouts.valueStateEmbedding());
      outputManager.attachAll(
          policyScores.alternativeScores(),
          policyScores.actionCandidateScores(),
          policyScores.riichiGateScores(),
          valueLogits,
          output.policyStateEmbedding(),
          output.valueStateEmbedding());
      return output;
    }
  }

  /** 詰めた推論バッチの論理ビューと疎インデックスを順伝播有効範囲へ一時的に借用する。 */
  private static void temporarilyAttachInferenceBatch(
      NDManager scope, DecisionInferenceDeviceBatch batch) {
    scope.tempAttachAll(batch.inputs().arrays());
    scope.tempAttachAll(batch.playerMemoryPresentIndices(), batch.transitionPresentIndices());
    if (batch.policyExecutionIndices() != null) {
      scope.tempAttachAll(batch.policyExecutionIndices().packedIndices());
    }
  }

  /**
   * このネットワークの既存方策パラメーターを使い、複数の未回収順伝播を保持できる実行境界を作る。
   *
   * <p>新しいブロックやパラメーターは登録しない。各順伝播のメソッド呼び出しと依存解放は、デバイスパイプラインの投入スレッドが直列化する。
   *
   * @param manager パイプラインと同じデバイスを持つ長寿命管理元
   * @param parameterStore 凍結パラメーターの取得先
   * @param fusionSettings 構成要素ごとのFusion実行方式
   * @param expectedDataType 凍結パラメーターに要求する推論データ型
   * @param inputDataType デバイスへ転送する連続入力のデータ型
   * @param maxBatch 通常容量区分の最大行数
   * @param multiTransitionMaxBatch 複数遷移容量区分の最大行数
   * @param executionSlots 同時に保持できる未回収順伝播数
   * @param frozenParameters パラメーターが推論コンテキストの寿命中に更新されないなら{@code true}
   * @param valueReadoutEnabled 同じ実行計画へ価値特徴量の集約も含めるなら{@code true}
   * @return パイプライン-局所的な方策推論実行境界
   */
  public DecisionInferenceExecution newInferenceExecution(
      NDManager manager,
      ParameterStore parameterStore,
      DecisionInferenceFusionSettings fusionSettings,
      DataType expectedDataType,
      DataType inputDataType,
      int maxBatch,
      int multiTransitionMaxBatch,
      int executionSlots,
      boolean frozenParameters,
      boolean valueReadoutEnabled) {
    Objects.requireNonNull(fusionSettings, "fusionSettings");
    if (fusionSettings.requiresFrozenParameters() && !frozenParameters) {
      throw new IllegalArgumentException("FUSION execution requires frozen parameters");
    }
    if (fusionSettings.hasFusion() && !manager.getDevice().isGpu()) {
      throw new UnsupportedOperationException("FUSION execution requires a GPU inference device");
    }
    EpsilonPlayerMemoryFusionExecution playerMemory = null;
    EpsilonTileRelationFusionExecution tileRelation = null;
    EpsilonStrategicContextFusionExecution strategic = null;
    EpsilonMahjongStateReadoutGroupExecution readout = null;
    try {
      playerMemory =
          fusionSettings.playerMemory() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newPlayerMemoryInferenceExecution(
                  manager,
                  parameterStore,
                  expectedDataType,
                  inputDataType,
                  maxBatch,
                  executionSlots)
              : null;
      tileRelation =
          fusionSettings.tileRelation() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newTileRelationInferenceExecution(
                  manager, parameterStore, expectedDataType, maxBatch, executionSlots)
              : null;
      strategic =
          fusionSettings.strategicContext() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newStrategicInferenceExecution(
                  manager, parameterStore, expectedDataType, maxBatch, executionSlots)
              : null;
      readout =
          fusionSettings.stateReadout() == DecisionInferenceFusionMode.FUSION
              ? EpsilonMahjongStateReadoutGroupExecution.create(
                  manager,
                  maxBatch,
                  executionSlots,
                  policyStateReadout.frozenParameterView(parameterStore, manager),
                  valueReadoutEnabled
                      ? valueStateReadout.frozenParameterView(parameterStore, manager)
                      : null)
              : null;
      DecisionPolicyExecution policy =
          policyHead.newExecution(
              manager,
              parameterStore,
              fusionSettings,
              expectedDataType,
              maxBatch,
              multiTransitionMaxBatch,
              executionSlots,
              frozenParameters);
      return new DecisionInferenceExecution(playerMemory, tileRelation, strategic, readout, policy);
    } catch (RuntimeException | Error failure) {
      if (readout != null) {
        try {
          readout.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (strategic != null) {
        try {
          strategic.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (tileRelation != null) {
        try {
          tileRelation.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (playerMemory != null) {
        try {
          playerMemory.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  private DiagnosticOutput forwardWithDiagnostics(
      ParameterStore parameterStore,
      DecisionNetworkInputs inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return forwardWithDiagnostics(parameterStore, inputs, null, null, training, runtimeParameters);
  }

  private DiagnosticOutput forwardWithDiagnostics(
      ParameterStore parameterStore,
      DecisionNetworkInputs inputs,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (!training) {
      InferenceReadouts readouts =
          encodeInferenceReadouts(
              parameterStore, inputs, playerMemoryPresentIndices, runtimeParameters);
      DecisionPolicyScores policyScores =
          policyHead.score(
              parameterStore,
              readouts.policyState(),
              inputs,
              transitionPresentIndices,
              false,
              runtimeParameters);
      NDArray valueLogits =
          forwardValueHeads(
              parameterStore,
              readouts.valueStateEmbedding(),
              inputs.stateCategories(),
              detachedGrpBoundaryMarginals(inputs.boundaryContext()),
              false,
              runtimeParameters);
      return new DiagnosticOutput(
          policyScores,
          valueLogits,
          readouts.policyState().stateEmbedding(),
          readouts.valueStateEmbedding());
    }
    EpsilonMahjongStateEncoder.EncodedMemory memory =
        encodeMemory(
            parameterStore,
            inputs.stateCategories(),
            inputs.stateNumerics(),
            playerMemoryPresentIndices,
            training,
            runtimeParameters);
    EpsilonMahjongStateReadout.ReadoutContext readoutContext =
        EpsilonMahjongStateReadout.summarize(memory);
    NDArray policyStateEmbedding =
        policyStateReadout.read(
            parameterStore, memory, readoutContext, training, runtimeParameters);
    EpsilonMahjongStateEncoder.EncodedState policyEncodedState =
        memory.withStateEmbedding(policyStateEmbedding);
    DecisionPolicyScores policyScores =
        policyHead.score(
            parameterStore,
            policyEncodedState,
            inputs,
            transitionPresentIndices,
            training,
            runtimeParameters);
    NDArray valueStateEmbedding =
        valueStateReadout.read(
            parameterStore,
            memory.stopGradient(),
            readoutContext.stopGradient(),
            training,
            runtimeParameters);
    NDArray valueLogits =
        forwardValueHeads(
            parameterStore,
            valueStateEmbedding,
            inputs.stateCategories(),
            detachedGrpBoundaryMarginals(inputs.boundaryContext()),
            training,
            runtimeParameters);
    return new DiagnosticOutput(
        policyScores, valueLogits, policyStateEmbedding, valueStateEmbedding);
  }

  /** Entity全体を特徴量の集約後までに解放し、方策が参照するメモリだけを残す。 */
  private EpsilonMahjongStateEncoder.EncodedState encodePolicyInferenceState(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      PairList<String, Object> runtimeParameters) {
    return encodePolicyInferenceState(parameterStore, inputs, null, runtimeParameters);
  }

  private EpsilonMahjongStateEncoder.EncodedState encodePolicyInferenceState(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      PairList<String, Object> runtimeParameters) {
    return encodePolicyInferenceState(
        parameterStore, inputs, playerMemoryPresentIndices, null, runtimeParameters);
  }

  private EpsilonMahjongStateEncoder.EncodedState encodePolicyInferenceState(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      DecisionInferenceExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    NDManager outputManager = inputs.stateCategories().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(inputs.stateCategories(), inputs.stateNumerics());
      EpsilonMahjongStateEncoder.EncodedMemory memory =
          encodeMemory(
              parameterStore,
              inputs.stateCategories(),
              inputs.stateNumerics(),
              playerMemoryPresentIndices,
              inferenceForward,
              false,
              runtimeParameters);
      NDArray policyStateEmbedding =
          inferenceForward != null && inferenceForward.readout() != null
              ? inferenceForward.readout().read(memory).policyStateEmbedding()
              : policyStateReadout.read(parameterStore, memory, false, runtimeParameters);
      EpsilonMahjongStateEncoder.EncodedState encodedState =
          memory.withStateEmbedding(policyStateEmbedding);
      outputManager.attachAll(
          encodedState.stateEmbedding(),
          encodedState.tileEmbeddings(),
          encodedState.playerMemory(),
          encodedState.playerMemoryMask());
      if (encodedState.tileProjectionEmbeddings() != encodedState.tileEmbeddings()) {
        outputManager.attachAll(encodedState.tileProjectionEmbeddings());
      }
      return encodedState;
    }
  }

  /** 価値特徴量の集約完了後に共有局・プレイヤー・牌などの特徴表現を解放する。 */
  private NDArray encodeValueInferenceState(
      ParameterStore parameterStore,
      DecisionStateInputs inputs,
      NDArray playerMemoryPresentIndices,
      PairList<String, Object> runtimeParameters) {
    NDManager outputManager = inputs.stateCategories().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(inputs.stateCategories(), inputs.stateNumerics());
      EpsilonMahjongStateEncoder.EncodedMemory memory =
          encodeMemory(
              parameterStore,
              inputs.stateCategories(),
              inputs.stateNumerics(),
              playerMemoryPresentIndices,
              false,
              runtimeParameters);
      NDArray stateEmbedding =
          valueStateReadout.read(parameterStore, memory, false, runtimeParameters);
      outputManager.attachAll(stateEmbedding);
      return stateEmbedding;
    }
  }

  /** 方策・価値特徴量の集約を共有構成要素の寿命内で完了し、後段に必要な配列だけを残す。 */
  private InferenceReadouts encodeInferenceReadouts(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      PairList<String, Object> runtimeParameters) {
    return encodeInferenceReadouts(parameterStore, inputs, null, runtimeParameters);
  }

  private InferenceReadouts encodeInferenceReadouts(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      PairList<String, Object> runtimeParameters) {
    return encodeInferenceReadouts(
        parameterStore, inputs, playerMemoryPresentIndices, null, runtimeParameters);
  }

  private InferenceReadouts encodeInferenceReadouts(
      ParameterStore parameterStore,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      DecisionInferenceExecution.Forward inferenceForward,
      PairList<String, Object> runtimeParameters) {
    NDManager outputManager = inputs.stateCategories().getManager();
    try (NDManager scope = outputManager.newSubManager()) {
      scope.tempAttachAll(inputs.stateCategories(), inputs.stateNumerics());
      EpsilonMahjongStateEncoder.EncodedMemory memory =
          encodeMemory(
              parameterStore,
              inputs.stateCategories(),
              inputs.stateNumerics(),
              playerMemoryPresentIndices,
              inferenceForward,
              false,
              runtimeParameters);
      NDArray policyStateEmbedding;
      NDArray valueStateEmbedding;
      if (inferenceForward != null && inferenceForward.readout() != null) {
        EpsilonMahjongStateReadoutGroupExecution.Readouts readouts =
            inferenceForward.readout().read(memory);
        policyStateEmbedding = readouts.policyStateEmbedding();
        valueStateEmbedding = readouts.valueStateEmbedding();
      } else {
        EpsilonMahjongStateReadout.ReadoutContext readoutContext =
            EpsilonMahjongStateReadout.summarize(memory);
        policyStateEmbedding =
            policyStateReadout.read(
                parameterStore, memory, readoutContext, false, runtimeParameters);
        valueStateEmbedding =
            valueStateReadout.read(
                parameterStore, memory, readoutContext, false, runtimeParameters);
      }
      EpsilonMahjongStateEncoder.EncodedState policyState =
          memory.withStateEmbedding(policyStateEmbedding);
      outputManager.attachAll(
          policyState.stateEmbedding(),
          policyState.tileEmbeddings(),
          policyState.playerMemory(),
          policyState.playerMemoryMask(),
          valueStateEmbedding);
      if (policyState.tileProjectionEmbeddings() != policyState.tileEmbeddings()) {
        outputManager.attachAll(policyState.tileProjectionEmbeddings());
      }
      return new InferenceReadouts(policyState, valueStateEmbedding);
    }
  }

  private EpsilonMahjongStateEncoder.EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encodeMemory(
        parameterStore, stateCategories, stateNumerics, null, training, runtimeParameters);
  }

  private EpsilonMahjongStateEncoder.EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      DecisionInferenceExecution.Forward inferenceForward,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return stateEncoder.encodeMemory(
        parameterStore,
        stateCategories,
        stateNumerics,
        playerMemoryPresentIndices,
        inferenceForward == null ? null : inferenceForward.playerMemory(),
        inferenceForward == null ? null : inferenceForward.tileRelation(),
        inferenceForward == null ? null : inferenceForward.strategic(),
        training,
        runtimeParameters);
  }

  private EpsilonMahjongStateEncoder.EncodedMemory encodeMemory(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    return encodeMemory(
        parameterStore,
        stateCategories,
        stateNumerics,
        playerMemoryPresentIndices,
        null,
        training,
        runtimeParameters);
  }

  /** 固定 GRP の期待効用を Gaussian 事前予測にし、101 区間の状態残差接続を加える。 */
  private NDArray forwardValueHeads(
      ParameterStore parameterStore,
      NDArray stateEmbedding,
      NDArray stateCategories,
      NDArray boundaryMarginals,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    NDArray valueHiddenEmbedding =
        valueHiddenHead
            .forward(parameterStore, new NDList(stateEmbedding), training, runtimeParameters)
            .singletonOrThrow();
    valueHiddenEmbedding = EpsilonMahjongStateEncoder.silu(valueHiddenEmbedding);
    NDArray residualLogits =
        valueHead
            .forward(parameterStore, new NDList(valueHiddenEmbedding), training, runtimeParameters)
            .singletonOrThrow()
            .reshape(stateEmbedding.getShape().get(0), EpsilonDecisionHlGauss.BIN_COUNT);
    NDArray centeredResidual = residualLogits.sub(residualLogits.mean(new int[] {1}, true));
    NDArray rankProbabilities = selectPlayerBoundaryMarginal(stateCategories, boundaryMarginals);
    EpsilonDecisionHlGauss.DeviceConstants constants =
        runtimeParameters == null
            ? null
            : (EpsilonDecisionHlGauss.DeviceConstants)
                runtimeParameters.get(EpsilonDecisionHlGauss.DEVICE_CONSTANTS);
    NDArray priorUtility =
        constants == null
            ? rankProbabilities
                .mul(rankProbabilities.getManager().create(utilityWeights))
                .sum(new int[] {1})
            : constants.priorUtility(rankProbabilities);
    NDArray priorProbabilities =
        constants == null
            ? EpsilonDecisionHlGauss.targetProbabilities(priorUtility, utilityProfile)
            : constants.targetProbabilities(priorUtility);
    // ログだけを下限値し、遠い区間にも有限の残差接続で確率を移せるようにする。
    NDArray prior =
        priorProbabilities
            .maximum(EpsilonDecisionHlGauss.PRIOR_PROBABILITY_FLOOR)
            .log()
            .stopGradient();
    return prior.add(centeredResidual);
  }

  /** 重みを固定したGRP の局境界周辺分布を学習グラフから分離する。 */
  private static NDArray detachedGrpBoundaryMarginals(NDArray boundaryPrior) {
    return boundaryPrior
        .get(
            new NDIndex(
                ":, {}:{}",
                DecisionBoundaryContext.GRP_RANK_OFFSET,
                DecisionBoundaryContext.GRP_RANK_OFFSET + DecisionBoundaryContext.GRP_RANK_SIZE))
        .stopGradient();
  }

  private static NDArray selectPlayerBoundaryMarginal(
      NDArray stateCategories, NDArray flatBoundaryMarginals) {
    long batch = stateCategories.getShape().get(0);
    NDArray playerSeats = stateCategories.get(new NDIndex(":, 0")).sub(1).reshape(batch, 1);
    NDArray seatIds =
        stateCategories
            .getManager()
            .arange(EpsilonDecisionConstants.PLAYERS)
            .reshape(1, EpsilonDecisionConstants.PLAYERS);
    NDArray seatMask =
        seatIds
            .eq(playerSeats)
            .toType(flatBoundaryMarginals.getDataType(), false)
            .reshape(batch, EpsilonDecisionConstants.PLAYERS, 1);
    return flatBoundaryMarginals
        .reshape(batch, EpsilonDecisionConstants.PLAYERS, EpsilonDecisionConstants.PLAYERS)
        .mul(seatMask)
        .sum(new int[] {1});
  }

  private record InferenceReadouts(
      EpsilonMahjongStateEncoder.EncodedState policyState, NDArray valueStateEmbedding) {}

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore,
      NDList inputs,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    DecisionNetworkInputs networkInputs = DecisionNetworkInputs.fromNDList(inputs);
    return forwardDecision(parameterStore, networkInputs, training, runtimeParameters).toNDList();
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    long rowCount = inputShapes == null || inputShapes.length < 2 ? -1 : inputShapes[0].get(0);
    long legalActionCapacity =
        inputShapes == null || inputShapes.length < 2 ? -1 : inputShapes[1].get(1);
    return new Shape[] {
      new Shape(rowCount, DecisionAlternative.NETWORK_SIZE),
      new Shape(rowCount, legalActionCapacity),
      new Shape(rowCount, legalActionCapacity),
      new Shape(rowCount, EpsilonDecisionHlGauss.BIN_COUNT)
    };
  }

  /**
   * パラメーター名が方策スコア計算処理群に属するか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 方策出力層配下なら {@code true}
   */
  public static boolean isPolicyHeadParameterName(String name) {
    return name.contains("policyHead");
  }

  /**
   * パラメーター名が共有状態エンコーダーに属するか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 状態エンコーダー配下なら {@code true}
   */
  public static boolean isSharedStateParameterName(String name) {
    return name.contains("stateEncoder");
  }

  /**
   * パラメーター名が方策所有の共有エンコーダー、特徴量の集約、スコア計算処理に属するか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 方策オプティマイザーが所有するブロック配下なら {@code true}
   */
  public static boolean isPolicyParameterName(String name) {
    return isSharedStateParameterName(name)
        || name.contains("policyStateReadout")
        || isPolicyHeadParameterName(name);
  }

  /**
   * パラメーター名を自己対局による方策オプティマイザーが所有するか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 自己対局による方策の更新対象なら {@code true}
   */
  public static boolean isOnlineActorParameterName(String name) {
    return isPolicyParameterName(name);
  }

  /**
   * パラメーター名が価値専用特徴量の集約または効用-区間出力層に属するか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 価値オプティマイザーの更新対象なら {@code true}
   */
  public static boolean isValueParameterName(String name) {
    return name.contains("valueStateReadout")
        || name.contains("valueHiddenHead")
        || name.contains("valueHead");
  }

  /**
   * 保存データによる一つに事前学習で更新可能なパラメーターか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 方策または価値の更新対象なら {@code true}
   */
  public static boolean isOfflinePretrainingParameterName(String name) {
    return isPolicyParameterName(name) || isValueParameterName(name);
  }

  /**
   * 保存データによる方策のみ事前学習で更新するパラメーターか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 方策のみ事前学習の更新対象なら {@code true}
   */
  public static boolean isOfflinePolicyParameterName(String name) {
    return isPolicyParameterName(name);
  }

  /**
   * 保存データによる価値のみ事前学習で更新するパラメーターか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 価値のみ事前学習の更新対象なら {@code true}
   */
  public static boolean isOfflineValueParameterName(String name) {
    return isValueParameterName(name);
  }

  /**
   * 自己対局による一つに学習のオプティマイザーが所有するパラメーターか判定する。
   *
   * @param name DJL パラメーターの階層名
   * @return 方策または価値の更新対象なら {@code true}
   */
  public static boolean isOnlineTrainingParameterName(String name) {
    return isOnlineActorParameterName(name) || isValueParameterName(name);
  }

  /**
   * 方策・価値のブロックを保存データによる事前学習対象に応じて切り替える。
   *
   * <p>共有状態エンコーダーは方策所有であり、価値だけを有効にしても学習可能なにしない。
   *
   * @param policyEnabled 共有状態エンコーダー、方策特徴量の集約、スコア計算処理を更新するか
   * @param valueEnabled 価値特徴量の集約と効用-区間出力層を更新するか
   */
  public synchronized void prepareForOfflinePretraining(
      boolean policyEnabled, boolean valueEnabled) {
    if (!policyEnabled && !valueEnabled) {
      throw new IllegalArgumentException("offline pretraining requires Policy or Decision Value");
    }
    stateEncoder.freezeParameters(!policyEnabled);
    policyStateReadout.freezeParameters(!policyEnabled);
    policyHead.freezeParameters(!policyEnabled);
    valueStateReadout.freezeParameters(!valueEnabled);
    valueHiddenHead.freezeParameters(!valueEnabled);
    valueHead.freezeParameters(!valueEnabled);
  }

  /** 自己対局による一つに学習で更新する方策・価値ブロックをすべて学習可能なに戻す。 */
  public synchronized void prepareForOnlineTraining() {
    stateEncoder.freezeParameters(false);
    policyStateReadout.freezeParameters(false);
    policyHead.freezeParameters(false);
    valueStateReadout.freezeParameters(false);
    valueHiddenHead.freezeParameters(false);
    valueHead.freezeParameters(false);
  }

  /**
   * スコア計算処理/価値出力と、共有メモリから独立に読んだ集約表現をまとめた監査用出力。
   *
   * @param policyScores 方策グラフ用の未正規化スコア
   * @param valueLogits 効用-区間ロジット [バッチ,101]
   * @param policyStateEmbedding 方策特徴量の集約の {@code [batch, hidden]} 表現
   * @param valueStateEmbedding 価値特徴量の集約の {@code [batch, hidden]} 表現
   */
  public record DiagnosticOutput(
      DecisionPolicyScores policyScores,
      NDArray valueLogits,
      NDArray policyStateEmbedding,
      NDArray valueStateEmbedding) {}

  /**
   * 価値のみ検証が返す効用-区間ロジットと価値特徴量の集約表現。
   *
   * @param valueLogits 効用区間の未正規化ロジット。形状は {@code [batch,101]}
   * @param valueStateEmbedding 共有メモリから価値専用特徴量の集約が集約した表現。形状は {@code [batch, hidden]}
   */
  public record ValueDiagnosticOutput(NDArray valueLogits, NDArray valueStateEmbedding) {}
}
