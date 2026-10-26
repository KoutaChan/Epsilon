package com.epsilon.nano.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.nano.ai.decision.input.DecisionInferenceInputs;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.input.DecisionPolicyInputs;
import com.epsilon.nano.ai.decision.policy.EpsilonDecisionCandidateEncoder;
import com.epsilon.nano.ai.decision.policy.TransitionPooling;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Model-owned frozen Nano features and constants, independent from the original training weights.
 * Per-batch execution plans borrow these parameters and must drain and close before this owner.
 */
public final class EpsilonDecisionFrozenFeatures implements AutoCloseable {
  private static final DecisionInferenceFusionSettings FEATURE_FUSION =
      new DecisionInferenceFusionSettings(
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.FUSION,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER,
          DecisionInferenceFusionMode.EAGER);

  private final NDManager storage;
  private final EpsilonDecisionFeatureSpine spine;
  private final PairList<String, Object> runtimeConstants;
  private final DataType inputType;
  private final boolean fusionEnabled;
  private boolean closed;

  EpsilonDecisionFrozenFeatures(
      EpsilonDecisionFeatureSpine source,
      NDManager owner,
      DataType computeType,
      TransitionPooling transitionPooling) {
    Objects.requireNonNull(computeType);
    if (computeType != DataType.FLOAT32
        && computeType != DataType.BFLOAT16
        && computeType != DataType.FLOAT16)
      throw new IllegalArgumentException("Unsupported frozen feature precision: " + computeType);
    inputType = owner.getDevice().isGpu() ? computeType : DataType.FLOAT32;
    storage = owner.newSubManager();
    int hidden = source.candidates().hiddenSize();
    spine = new EpsilonDecisionFeatureSpine(hidden, transitionPooling);
    // The production 256-wide fusion plans are validated independently. Other shapes keep the
    // same frozen parameters and use the ordinary eager execution, without exception fallback.
    fusionEnabled =
        storage.getDevice().isGpu()
            && hidden == 256
            && (inputType == DataType.BFLOAT16 || inputType == DataType.FLOAT16);
    try {
      var originals = parameters(source);
      var copies = parameters(spine);
      if (!originals.keySet().equals(copies.keySet()))
        throw new IllegalArgumentException("Frozen feature parameter structure differs");
      for (Parameter parameter : originals.values()) {
        if (parameter.requiresGradient())
          throw new IllegalArgumentException("Frozen feature source is trainable");
      }
      for (Parameter parameter : copies.values()) {
        parameter.setInitializer(new ConstantInitializer(0));
        parameter.freeze(true);
      }
      spine
          .stateEncoder()
          .initialize(
              storage,
              inputType,
              DecisionNetworkInputs.initializationShape(
                  DecisionInputLayout.Tensor.STATE_CATEGORIES),
              DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS));
      spine
          .stateReadout()
          .initialize(
              storage,
              inputType,
              new Shape(-1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, hidden));
      spine.candidates().initialize(storage, inputType);
      for (var entry : originals.entrySet()) {
        NDArray from = entry.getValue().getArray();
        NDArray to = copies.get(entry.getKey()).getArray();
        if (!from.getShape().equals(to.getShape()))
          throw new IllegalArgumentException("Frozen feature parameter shape differs");
        from.copyTo(to);
      }
      runtimeConstants = EpsilonTileRelationEncoder.createRuntimeParameters(storage);
    } catch (RuntimeException | Error failure) {
      try {
        storage.close();
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** The returned store also resolves unrelated owner parameters in their original precision. */
  public ParameterStore parameters(NDManager workingManager) {
    ensureOpen();
    if (!workingManager.getDevice().equals(storage.getDevice()))
      throw new IllegalArgumentException("Frozen features belong to a different device");
    return new ParameterStore(workingManager, false);
  }

  /** Floating input slabs use the same precision as the independent feature parameters. */
  public DataType inputNumericType() {
    return inputType;
  }

  public boolean fusionEnabled() {
    return fusionEnabled;
  }

  public EpsilonDecisionCandidateEncoder candidates() {
    ensureOpen();
    return spine.candidates();
  }

  /** Creates independent execution slots; the caller owns their drain and close lifecycle. */
  public DecisionInferenceExecution newExecution(int maxBatch, int executionSlots) {
    ensureOpen();
    if (maxBatch < 1 || executionSlots < 1)
      throw new IllegalArgumentException("Invalid frozen inference capacity");
    var parameters = parameters(storage);
    var settings = fusionEnabled ? FEATURE_FUSION : DecisionInferenceFusionSettings.eager();
    var candidates =
        spine
            .candidates()
            .newInferenceExecution(
                storage, parameters, settings, inputType, maxBatch, executionSlots);
    try {
      return spine.newInferenceExecution(
          storage,
          parameters,
          settings,
          inputType,
          inputType,
          maxBatch,
          executionSlots,
          null,
          null,
          candidates);
    } catch (RuntimeException | Error failure) {
      try {
        candidates.close();
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** 候補表現の共有計画を使い、方策固有の分岐や出力ヘッドを起動せずに推論する。 */
  public NDArray encodeCandidates(
      ParameterStore parameters,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionNetworkInputs inputs,
      NDArray presentTransitions,
      DecisionInferenceExecution.Forward forward,
      PairList<String, Object> runtimeParameters) {
    ensureOpen();
    return spine
        .candidates()
        .encodeInferenceRepresentation(
            parameters,
            state,
            inputs,
            presentTransitions,
            forward == null ? null : forward.candidates(),
            runtimeParameters);
  }

  /** CPUで集約済みの遷移をそのまま使い、密な入力への展開とデバイス上の再集約を省く。 */
  public NDArray encodeCandidates(
      ParameterStore parameters,
      EpsilonMahjongStateEncoder.EncodedState state,
      DecisionInferenceInputs inputs,
      NDArray presentTransitions,
      DecisionInferenceExecution.Forward forward,
      PairList<String, Object> runtimeParameters) {
    ensureOpen();
    return spine
        .candidates()
        .encodeInferenceRepresentation(
            parameters,
            state,
            inputs,
            presentTransitions,
            forward == null ? null : forward.candidates(),
            runtimeParameters);
  }

  /** Produces the shared state with optional per-batch frozen execution plans. */
  public EpsilonMahjongStateEncoder.EncodedState encodeState(
      ParameterStore parameters,
      DecisionPolicyInputs inputs,
      NDArray playerMemoryPresentIndices,
      DecisionInferenceExecution.Forward forward,
      PairList<String, Object> runtimeParameters) {
    return encodeState(
        parameters,
        inputs.stateCategories(),
        inputs.stateNumerics(),
        playerMemoryPresentIndices,
        forward,
        runtimeParameters);
  }

  /** 候補の容量区分を分けずに、全盤面へ同じ凍結計画を適用する。 */
  public EpsilonMahjongStateEncoder.EncodedState encodeState(
      ParameterStore parameters,
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray playerMemoryPresentIndices,
      DecisionInferenceExecution.Forward forward,
      PairList<String, Object> runtimeParameters) {
    ensureOpen();
    runtimeParameters = withRuntimeConstants(runtimeParameters);
    var memory =
        spine
            .stateEncoder()
            .encodeMemory(
                parameters,
                stateCategories,
                stateNumerics,
                playerMemoryPresentIndices,
                forward == null ? null : forward.playerMemory(),
                forward == null ? null : forward.tileRelation(),
                forward == null ? null : forward.strategic(),
                false,
                runtimeParameters);
    return forward != null && forward.readout() != null
        ? memory.withStateEmbedding(forward.readout().read(memory).policyStateEmbedding())
        : spine.readState(parameters, memory, playerMemoryPresentIndices, false, runtimeParameters);
  }

  private PairList<String, Object> withRuntimeConstants(PairList<String, Object> requested) {
    if (requested == null || requested.isEmpty()) return runtimeConstants;
    var merged = new PairList<String, Object>(requested.size() + runtimeConstants.size());
    for (var entry : requested) merged.add(entry.getKey(), entry.getValue());
    for (var constant : runtimeConstants) {
      boolean supplied = false;
      for (var entry : requested) {
        if (entry.getKey().equals(constant.getKey())) {
          supplied = true;
          break;
        }
      }
      if (!supplied) merged.add(constant.getKey(), constant.getValue());
    }
    return merged;
  }

  private static Map<String, Parameter> parameters(EpsilonDecisionFeatureSpine spine) {
    var parameters = new LinkedHashMap<String, Parameter>();
    append(parameters, "stateEncoder", spine.stateEncoder());
    append(parameters, "stateReadout", spine.stateReadout());
    spine.candidates().registerWith((name, block) -> append(parameters, name, block));
    return parameters;
  }

  private static void append(Map<String, Parameter> parameters, String prefix, Block block) {
    for (var entry : block.getParameters())
      parameters.put(prefix + "/" + entry.getKey(), entry.getValue());
  }

  private void ensureOpen() {
    if (closed) throw new IllegalStateException("Frozen feature owner is closed");
  }

  @Override
  public void close() {
    if (closed) return;
    closed = true;
    storage.close();
  }
}
