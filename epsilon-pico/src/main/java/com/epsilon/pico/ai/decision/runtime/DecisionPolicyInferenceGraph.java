package com.epsilon.pico.ai.decision.runtime;

import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtAcceleratorGraph;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.pico.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.pico.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.pico.ai.model.EpsilonDecisionNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 固定形状の方策グラフと、その再利用可能な入力・出力バッファを一体で所有する。 */
final class DecisionPolicyInferenceGraph implements AutoCloseable {

  private static final Logger LOGGER = LoggerFactory.getLogger(DecisionPolicyInferenceGraph.class);
  private static final String DENSE_GRAPH_CAPTURE = "denseGraphCapture";

  private final PtNDManager manager;
  private final DecisionBatchTransfer.Workspace workspace;
  private final DecisionBatchTransfer.GraphInput input;
  private final PtAcceleratorGraph graph;
  private final NDArray packedScoreTensor;

  private DecisionPolicyInferenceGraph(
      PtNDManager manager,
      DecisionBatchTransfer.Workspace workspace,
      DecisionBatchTransfer.GraphInput input,
      PtAcceleratorGraph graph,
      NDArray packedScoreTensor) {
    this.manager = manager;
    this.workspace = workspace;
    this.input = input;
    this.graph = graph;
    this.packedScoreTensor = packedScoreTensor;
  }

  static DecisionPolicyInferenceGraph capture(
      NDManager owner,
      EpsilonDecisionNetwork network,
      ParameterStore parameterStore,
      PairList<String, Object> runtimeParameters,
      DecisionComputePrecision computePrecision,
      DecisionHostBatch.RowSlice hostRows,
      DataType inputNumericDataType) {
    PtNDManager graphManager = (PtNDManager) owner.newSubManager();
    DecisionBatchTransfer.Workspace graphWorkspace =
        new DecisionBatchTransfer.Workspace(graphManager);
    DecisionBatchTransfer.GraphInput graphInput =
        DecisionBatchTransfer.prepareGraphInput(
            graphManager, hostRows, graphWorkspace, inputNumericDataType);
    PtAcceleratorGraph graph =
        ((PtEngine) owner.getEngine()).newAcceleratorGraph(owner.getDevice());
    NDArray packedScoreTensor;
    try (Autocast ignored = EpsilonDecisionAutocast.open(graphManager, computePrecision)) {
      graph.beginCapture();
      DecisionDeviceBatch deviceBatch = graphInput.bind();
      PairList<String, Object> graphParameters = new PairList<>();
      graphParameters.addAll(runtimeParameters);
      graphParameters.add(DENSE_GRAPH_CAPTURE, true);
      DecisionPolicyScores rawScores =
          network.forwardPolicy(parameterStore, deviceBatch, false, graphParameters);
      packedScoreTensor =
          DecisionInferenceOutputCodec.packFloat32Scores(
              new NDList(
                  rawScores.alternativeScores(),
                  rawScores.actionCandidateScores(),
                  rawScores.riichiGateScores()));
      graphManager.attachAll(new NDList(packedScoreTensor));
      graph.endCapture();
    }
    LOGGER.info(
        "Decision accelerator graph captured: device={} bucket={} rows={}",
        owner.getDevice(),
        hostRows.bucket(),
        hostRows.size());
    return new DecisionPolicyInferenceGraph(
        graphManager, graphWorkspace, graphInput, graph, packedScoreTensor);
  }

  void refresh(DecisionHostBatch.RowSlice hostRows) {
    input.refresh(hostRows);
  }

  void replay() {
    graph.replay();
  }

  NDArray packedScoreTensor() {
    return packedScoreTensor;
  }

  float[] packedScores() {
    return packedScoreTensor.toFloatArray();
  }

  @Override
  public void close() {
    graph.close();
    workspace.close();
    manager.close();
  }
}
