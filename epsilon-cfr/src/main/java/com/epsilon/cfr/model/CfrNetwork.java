package com.epsilon.cfr.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionGroupedInferenceBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputLayout;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.policy.TransitionPooling;
import com.epsilon.nano.ai.model.DecisionInferenceExecution;
import com.epsilon.nano.ai.model.EpsilonDecisionFeatureSpine;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import com.epsilon.nano.ai.model.EpsilonDecisionNetwork;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.util.ArrayList;
import java.util.List;

/** Nanoと同じ盤面・候補表現を使い、各行動のスカラー値を出力する。モデルごとに後悔または平均方策の一方を扱う。 */
final class CfrNetwork extends AbstractBlock {
  static final String ARCHITECTURE = "epsilon-cfr-nano-board-feature-spine-v3";
  private final CfrModelConfig config;
  private final EpsilonDecisionFeatureSpine featureSpine;
  private final Linear candidateHidden, output;

  CfrNetwork(CfrModelConfig config) {
    this.config = config;
    int hidden = config.hiddenSize();
    featureSpine =
        new EpsilonDecisionFeatureSpine(hidden, TransitionPooling.CANDIDATE_REPRESENTATION);
    addChildBlock("stateEncoder", featureSpine.stateEncoder());
    addChildBlock("stateReadout", featureSpine.stateReadout());
    featureSpine.candidates().registerWith(this::addChildBlock);
    candidateHidden = addChildBlock("candidateHidden", Linear.builder().setUnits(hidden).build());
    output = addChildBlock("output", Linear.builder().setUnits(1).build());
    // 累積後悔ゼロの後悔マッチングから始めるため、初期方策は一様分布になる。
    // 平均方策も一様分布で初期化し、表現の重みはすべて学習可能なままにする。
    output.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0));
    output.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0));
  }

  EpsilonDecisionFrozenFeatures frozenFeatures(NDManager owner, DataType computeType) {
    return featureSpine.newFrozenInference(owner, computeType);
  }

  boolean isOutputParameter(Parameter parameter) {
    return output.getDirectParameters().values().contains(parameter);
  }

  CfrModel.NanoInitializationReport initializeRepresentationFromNano(Block source) {
    if (!(source instanceof EpsilonDecisionNetwork nano))
      throw new IllegalArgumentException("Source must be an Epsilon-Nano decision model");
    if (nano.hiddenSize() != config.hiddenSize())
      throw new IllegalArgumentException(
          "Nano/CFR hidden sizes differ: " + nano.hiddenSize() + " / " + config.hiddenSize());
    List<ParameterCopy> copies = new ArrayList<>();
    planCopies(child(nano, "stateEncoder"), featureSpine.stateEncoder(), copies);
    planCopies(child(nano, "policyStateReadout"), featureSpine.stateReadout(), copies);
    Block policy = child(nano, "policyHead");
    featureSpine
        .candidates()
        .registerWith((name, target) -> planCopies(child(policy, name), target, copies));
    // 全ブロックの存在・形状を検証してから変更し、非互換な入力による部分コピーを避ける。
    for (ParameterCopy copy : copies) copy.source().copyTo(copy.target());
    return new CfrModel.NanoInitializationReport(
        copies.size(),
        copies.stream().mapToLong(copy -> copy.target().size()).sum(),
        getParameters().values().stream()
            .mapToLong(parameter -> parameter.getArray().size())
            .sum());
  }

  private record ParameterCopy(NDArray source, NDArray target) {}

  private static Block child(Block parent, String name) {
    Block result = null;
    for (var child : parent.getChildren()) {
      // DJLが登録順序の接頭辞を付けるため、親ごとに意味上の名前を解決する。
      if (!child.getKey().endsWith(name)) continue;
      if (result != null) throw new IllegalArgumentException("Ambiguous Nano block: " + name);
      result = child.getValue();
    }
    if (result == null) throw new IllegalArgumentException("Missing Nano block: " + name);
    return result;
  }

  private static void planCopies(Block source, Block target, List<ParameterCopy> copies) {
    var sourceParameters = source.getParameters();
    var targetParameters = target.getParameters();
    if (!sourceParameters.keys().equals(targetParameters.keys()))
      throw new IllegalArgumentException("Nano/CFR representation parameter names differ");
    for (var entry : targetParameters) {
      NDArray from = sourceParameters.get(entry.getKey()).getArray();
      NDArray to = entry.getValue().getArray();
      if (!from.getShape().equals(to.getShape()) || from.getDataType() != to.getDataType())
        throw new IllegalArgumentException(
            "Nano/CFR representation shape or dtype differs: " + entry.getKey());
      copies.add(new ParameterCopy(from, to));
    }
  }

  NDArray scores(ParameterStore store, DecisionDeviceBatch batch, boolean training) {
    var parameters = new PairList<String, Object>();
    var state =
        featureSpine.encodeState(
            store, batch.inputs(), batch.playerMemoryPresentIndices(), training, parameters);
    NDArray candidate =
        featureSpine
            .candidates()
            .encodeRepresentation(
                store,
                state,
                batch.inputs(),
                batch.transitionPresentIndices(),
                training,
                parameters);
    return scoreCandidates(
        store, candidate, batch.bucket().legalActionCapacity(), training, parameters);
  }

  /** 全盤面・遷移を共有し、合法候補のスコアだけを元の行動順で返す。 */
  NDArray inference(
      ParameterStore store,
      DecisionGroupedInferenceBatch batch,
      EpsilonDecisionFrozenFeatures frozen,
      DecisionInferenceExecution.Forward forward) {
    var parameters = new PairList<String, Object>();
    var state =
        frozen == null
            ? featureSpine.encodeState(
                store,
                batch.stateCategories(),
                batch.stateNumerics(),
                batch.playerMemoryPresentIndices(),
                false,
                parameters)
            : frozen.encodeState(
                store,
                batch.stateCategories(),
                batch.stateNumerics(),
                batch.playerMemoryPresentIndices(),
                forward,
                parameters);
    var encoder = frozen == null ? featureSpine.candidates() : frozen.candidates();
    NDArray candidates =
        encoder.encodeGroupedInferenceRepresentation(
            store, state, batch, forward == null ? null : forward.candidates(), parameters);
    return scoreCandidates(store, candidates, 1, false, parameters).reshape(-1);
  }

  private NDArray scoreCandidates(
      ParameterStore store,
      NDArray candidate,
      int actionCapacity,
      boolean training,
      PairList<String, Object> parameters) {
    NDArray hidden =
        candidateHidden.forward(store, new NDList(candidate), training, parameters).getFirst();
    return output
        .forward(store, new NDList(EpsilonMahjongStateEncoder.silu(hidden)), training, parameters)
        .getFirst()
        .reshape(candidate.getShape().get(0), actionCapacity);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType type, Shape... ignored) {
    int hidden = config.hiddenSize();
    featureSpine
        .stateEncoder()
        .initialize(
            manager,
            type,
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_CATEGORIES),
            DecisionNetworkInputs.initializationShape(DecisionInputLayout.Tensor.STATE_NUMERICS));
    featureSpine
        .stateReadout()
        .initialize(
            manager, type, new Shape(-1, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT, hidden));
    featureSpine.candidates().initialize(manager, type);
    candidateHidden.initialize(manager, type, new Shape(-1, -1, hidden));
    output.initialize(manager, type, new Shape(-1, -1, hidden));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore store, NDList inputs, boolean training, PairList<String, Object> parameters) {
    throw new UnsupportedOperationException("Use the typed CFR scores entrypoint");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] shapes) {
    return new Shape[] {new Shape(-1, -1)};
  }
}
