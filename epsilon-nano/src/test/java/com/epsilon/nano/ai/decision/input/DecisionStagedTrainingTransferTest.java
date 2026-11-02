package com.epsilon.nano.ai.decision.input;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtNDManager;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionPretrainBenchmark;
import com.epsilon.network.NetworkRuntime;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 分離したホスト準備とH2Dが、既存の標準形式転送と同じ入力・索引・教師値を渡す。 */
public class DecisionStagedTrainingTransferTest {
  @Test(groups = "native")
  public void stagingRequiresAGpuWorkspace() {
    NetworkRuntime.initialize();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        var workspace = new DecisionBatchTransfer.Workspace(manager)) {
      DecisionHostBatch source = fixture("training");
      Assert.expectThrows(
          IllegalStateException.class,
          () -> workspace.stageTraining(source.sliceRows(0, source.size())));
    }
  }

  @DataProvider
  public Object[][] layouts() {
    return new Object[][] {
      {"training", DataType.FLOAT32},
      {"training", DataType.BFLOAT16},
      {"dense-inference", DataType.FLOAT32},
      {"dense-inference", DataType.BFLOAT16},
      {"packed-inference", DataType.FLOAT32},
      {"packed-inference", DataType.BFLOAT16}
    };
  }

  @Test(groups = "rocm", dataProvider = "layouts")
  public void stagedTransferMatchesDirectTransferAndReusesBuffers(String kind, DataType type) {
    NetworkRuntime.initialize();
    Engine engine = Engine.getInstance();
    if (engine.getGpuCount() == 0) throw new SkipException("This test requires a GPU.");
    try (NDManager manager = engine.newBaseManager(Device.gpu());
        var workspace = new DecisionBatchTransfer.Workspace(manager);
        var other = new DecisionBatchTransfer.Workspace(manager)) {
      DecisionHostBatch source = fixture(kind);
      for (int count : new int[] {3, 1}) {
        var rows = source.sliceRows(1, count);
        try (NDManager expectedScope = manager.newSubManager();
            NDManager actualScope = manager.newSubManager()) {
          var expected =
              DecisionBatchTransfer.transferTrainingToDevice(
                  expectedScope, rows, null, DecisionTensorTransfer.DIRECT_BUFFER, type);
          var staged = workspace.stageTraining(rows);
          Assert.assertEquals(staged.rows(), rows.size());
          Assert.assertEquals(staged.bucket(), rows.bucket());
          Assert.expectThrows(
              IllegalArgumentException.class,
              () -> other.enqueueStagedTraining((PtNDManager) actualScope, staged, type));
          // stage後は元のホスト行を再読込みしない。教師値も準備時点の値を保持する。
          if (source.hasTrainingTargets()) source.setSampleWeight(1, count * 7);
          var actual = workspace.enqueueStagedTraining((PtNDManager) actualScope, staged, type);
          assertSameBatch(actual, expected);
        }
      }
    }
  }

  private static DecisionHostBatch fixture(String kind) {
    if (kind.equals("training"))
      return EpsilonDecisionPretrainBenchmark.syntheticBatch(4, new DecisionBucket(4, 1));
    var engine = new GameEngine(1871);
    var step = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = step.decisions().getFirst();
    var builder =
        DecisionBatchBuilder.inference(
            4, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
    for (int row = 0; row < 4; row++)
      builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
    DecisionHostBatch packed = builder.build();
    return kind.equals("packed-inference") ? packed : packed.copyRows(0, packed.size());
  }

  private static void assertSameBatch(DecisionDeviceBatch actual, DecisionDeviceBatch expected) {
    Assert.assertEquals(actual.rowCount(), expected.rowCount());
    Assert.assertEquals(actual.bucket(), expected.bucket());
    assertSameArray(actual.categoricalSlab(), expected.categoricalSlab());
    assertSameArray(actual.numericSlab(), expected.numericSlab());
    assertSameArray(actual.playerMemoryPresentIndices(), expected.playerMemoryPresentIndices());
    assertSameArray(actual.transitionPresentIndices(), expected.transitionPresentIndices());
    Assert.assertEquals(actual.hasTrainingTargets(), expected.hasTrainingTargets());
    if (actual.hasTrainingTargets()) {
      assertSameArray(
          actual.trainingTargets().categoricalSlab(), expected.trainingTargets().categoricalSlab());
      assertSameArray(
          actual.trainingTargets().numericSlab(), expected.trainingTargets().numericSlab());
    }
  }

  private static void assertSameArray(NDArray actual, NDArray expected) {
    Assert.assertEquals(actual.getShape(), expected.getShape());
    Assert.assertEquals(actual.getDataType(), expected.getDataType());
    Assert.assertEquals(
        actual.toType(DataType.FLOAT32, false).toFloatArray(),
        expected.toType(DataType.FLOAT32, false).toFloatArray());
  }
}
