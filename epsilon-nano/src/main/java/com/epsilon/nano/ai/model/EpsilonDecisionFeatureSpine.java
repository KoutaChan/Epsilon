package com.epsilon.nano.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.nano.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.nano.ai.decision.policy.DecisionCandidateExecution;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyExecution;
import com.epsilon.nano.ai.decision.policy.EpsilonDecisionCandidateEncoder;
import com.epsilon.nano.ai.decision.policy.TransitionPooling;
import com.epsilon.nano.ai.model.fusion.EpsilonMahjongStateReadoutGroupExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonPlayerMemoryFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonStrategicContextFusionExecution;
import com.epsilon.nano.ai.model.fusion.EpsilonTileRelationFusionExecution;

/**
 * Nanoの局面メモリ、方策用集約、合法候補表現をまとめた目的非依存の特徴抽出器。
 *
 * <p>方策グラフ、価値、後悔回帰、履歴追加は所有しない。各ネットワークは公開された部品を自身の Blockへ登録するので、Nanoの既存チェックポイント名を変えずに同じ実装を利用できる。
 */
public final class EpsilonDecisionFeatureSpine {
  public static final int POLICY_READOUT_MAXIMUM_FEED_FORWARD_WIDTH = 384;
  private final EpsilonMahjongStateEncoder stateEncoder;
  private final EpsilonMahjongStateReadout stateReadout;
  private final EpsilonDecisionCandidateEncoder candidates;
  private final TransitionPooling transitionPooling;

  public EpsilonDecisionFeatureSpine(int hiddenSize) {
    this(hiddenSize, TransitionPooling.SELECTED_DISCARD_POLICY);
  }

  public EpsilonDecisionFeatureSpine(int hiddenSize, TransitionPooling transitionPooling) {
    this.transitionPooling = transitionPooling;
    stateEncoder = new EpsilonMahjongStateEncoder(hiddenSize);
    stateReadout =
        new EpsilonMahjongStateReadout(hiddenSize, POLICY_READOUT_MAXIMUM_FEED_FORWARD_WIDTH);
    candidates = new EpsilonDecisionCandidateEncoder(hiddenSize, transitionPooling);
  }

  public EpsilonMahjongStateEncoder stateEncoder() {
    return stateEncoder;
  }

  public EpsilonMahjongStateReadout stateReadout() {
    return stateReadout;
  }

  public EpsilonDecisionCandidateEncoder candidates() {
    return candidates;
  }

  /** Owns an independent frozen feature copy without changing the source checkpoint or RNG. */
  public EpsilonDecisionFrozenFeatures newFrozenInference(NDManager owner, DataType computeType) {
    return new EpsilonDecisionFrozenFeatures(this, owner, computeType, transitionPooling);
  }

  /** Shared state-plan construction; the caller transfers an optional policy plan on success. */
  DecisionInferenceExecution newInferenceExecution(
      NDManager manager,
      ParameterStore parameterStore,
      DecisionInferenceFusionSettings fusionSettings,
      DataType parameterType,
      DataType inputType,
      int maxBatch,
      int executionSlots,
      EpsilonMahjongStateReadout valueReadout,
      DecisionPolicyExecution policy,
      DecisionCandidateExecution candidateExecution) {
    EpsilonPlayerMemoryFusionExecution playerMemory = null;
    EpsilonTileRelationFusionExecution tileRelation = null;
    EpsilonStrategicContextFusionExecution strategic = null;
    EpsilonMahjongStateReadoutGroupExecution readout = null;
    try {
      playerMemory =
          fusionSettings.playerMemory() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newPlayerMemoryInferenceExecution(
                  manager, parameterStore, parameterType, inputType, maxBatch, executionSlots)
              : null;
      tileRelation =
          fusionSettings.tileRelation() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newTileRelationInferenceExecution(
                  manager, parameterStore, parameterType, maxBatch, executionSlots)
              : null;
      strategic =
          fusionSettings.strategicContext() == DecisionInferenceFusionMode.FUSION
              ? stateEncoder.newStrategicInferenceExecution(
                  manager, parameterStore, parameterType, maxBatch, executionSlots)
              : null;
      readout =
          fusionSettings.stateReadout() == DecisionInferenceFusionMode.FUSION
              ? EpsilonMahjongStateReadoutGroupExecution.create(
                  manager,
                  maxBatch,
                  executionSlots,
                  stateReadout.frozenParameterView(parameterStore, manager),
                  valueReadout == null
                      ? null
                      : valueReadout.frozenParameterView(parameterStore, manager))
              : null;
      return new DecisionInferenceExecution(
          playerMemory, tileRelation, strategic, readout, policy, candidateExecution);
    } catch (RuntimeException | Error failure) {
      for (AutoCloseable resource :
          new AutoCloseable[] {readout, strategic, tileRelation, playerMemory}) {
        if (resource == null) continue;
        try {
          resource.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  public EpsilonMahjongStateEncoder.EncodedState encodeState(
      ParameterStore store,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> parameters) {
    return encodeState(
        store,
        inputs.stateCategories(),
        inputs.stateNumerics(),
        playerMemoryPresentIndices,
        training,
        parameters);
  }

  /** 候補容量から独立した盤面行を一度に符号化する。 */
  public EpsilonMahjongStateEncoder.EncodedState encodeState(
      ParameterStore store,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> parameters) {
    return readState(
        store,
        stateEncoder.encodeMemory(
            store,
            stateCategories,
            stateNumerics,
            playerMemoryPresentIndices,
            training,
            parameters),
        playerMemoryPresentIndices,
        training,
        parameters);
  }

  public EpsilonMahjongStateEncoder.EncodedState readState(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedMemory memory,
      boolean training,
      PairList<String, Object> parameters) {
    return readState(store, memory, null, training, parameters);
  }

  /** Reuses the transferred presence indices when projecting the state readout's entities. */
  public EpsilonMahjongStateEncoder.EncodedState readState(
      ParameterStore store,
      EpsilonMahjongStateEncoder.EncodedMemory memory,
      NDArray playerMemoryPresentIndices,
      boolean training,
      PairList<String, Object> parameters) {
    var summary = stateReadout.summarize(memory, playerMemoryPresentIndices);
    return memory.withStateEmbedding(
        stateReadout.read(store, memory, summary, training, parameters));
  }
}
