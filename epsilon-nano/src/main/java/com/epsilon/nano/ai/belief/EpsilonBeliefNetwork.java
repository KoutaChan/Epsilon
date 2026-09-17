package com.epsilon.nano.ai.belief;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.belief.AbstractBeliefNetwork;
import com.epsilon.ai.belief.EpsilonBeliefLayout;
import com.epsilon.config.settings.BeliefSettings;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import com.epsilon.nano.ai.model.EpsilonMahjongStateReadout;
import com.epsilon.nano.ai.model.EpsilonTileRelationEncoder;
import com.epsilon.nano.config.settings.EpsilonSettings;

/** Decision と共通の麻雀状態の入力形式を使い、公開情報から他家の手牌や待ちを推定する Belief モデル。 */
public final class EpsilonBeliefNetwork extends AbstractBeliefNetwork {

  /** チェックポイント内のパラメータと入力形式を識別するモデル構造 ID。 */
  public static final String ARCHITECTURE_ID =
      "belief-mahjong-entity-tile-rel2h4-key-attention-c64-"
          + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
          + "-readout-pertile-v11-"
          + DecisionInputSchema.fingerprint()
          + "-"
          + EpsilonTileRelationEncoder.FINGERPRINT;

  private final EpsilonMahjongStateEncoder stateEncoder;
  private final EpsilonMahjongStateReadout stateReadout;

  /** 現在の{@link BeliefSettings}でネットワークを構築する。 */
  public EpsilonBeliefNetwork() {
    this(EpsilonSettings.defaults().bind(BeliefSettings.class).hidden());
  }

  /**
   * 指定隠れ層幅で公開状態のエンコーダーとBelief 出力層を構築する。
   *
   * @param hiddenSize 注意ヘッド数で割り切れる正の隠れ層幅
   */
  public EpsilonBeliefNetwork(int hiddenSize) {
    this(
        hiddenSize,
        new EpsilonMahjongStateEncoder(hiddenSize),
        new EpsilonMahjongStateReadout(hiddenSize));
  }

  private EpsilonBeliefNetwork(
      int hiddenSize,
      EpsilonMahjongStateEncoder stateEncoder,
      EpsilonMahjongStateReadout stateReadout) {
    super(hiddenSize, stateEncoder, stateReadout);
    this.stateEncoder = stateEncoder;
    this.stateReadout = stateReadout;
  }

  public static String architectureSummary() {
    return architectureSummary(EpsilonSettings.defaults().bind(BeliefSettings.class).hidden());
  }

  public static String architectureSummary(int hidden) {
    return ARCHITECTURE_ID
        + " input=round+player4+tile34+river96+meld16"
        + " encoder=tile-rel2h4-key+shared-memory+"
        + EpsilonMahjongStateEncoder.STRATEGIC_CONTEXT_FINGERPRINT
        + "+c64-query-readout"
        + " tileHead=state-conditioned-per-tile"
        + " hidden="
        + hidden
        + " output="
        + EpsilonBeliefLayout.OUTPUT_SIZE;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape stateCategories =
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_CATEGORIES);
    Shape stateNumerics =
        DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS);
    stateEncoder.initialize(manager, dataType, stateCategories, stateNumerics);
    stateReadout.initialize(manager, dataType, new Shape(-1, hiddenSize()));
    super.initializeChildBlocks(manager, dataType, inputShapes);
  }

  /**
   * 標準形式の状態テンソルからBelief ロジットを計算する。
   *
   * @param parameterStore パラメーター取得先
   * @param stateCategories 形状 {@code [batch, STATE_INT_COUNT]}
   * @param stateNumerics 形状 {@code [batch, STATE_FLOAT_COUNT]}
   * @param training 学習モードならtrue
   * @param runtimeParameters DJL Blockへ渡す実行時パラメーター
   * @return 形状 {@code [batch, BELIEF_OUTPUT_SIZE]} のロジット
   */
  @Override
  public NDArray forwardBelief(
      ParameterStore parameterStore,
      NDArray stateCategories,
      NDArray stateNumerics,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    EpsilonMahjongStateEncoder.EncodedMemory memory =
        stateEncoder.encodeMemory(
            parameterStore, stateCategories, stateNumerics, training, runtimeParameters);
    NDArray stateEmbedding = stateReadout.read(parameterStore, memory, training, runtimeParameters);
    return projectBelief(
        parameterStore, memory.tileEmbeddings(), stateEmbedding, training, runtimeParameters);
  }
}
