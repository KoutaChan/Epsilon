package com.epsilon.nano.ai.decision.policy;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.policy.fusion.DecisionPolicyFusionCandidateContextExecution;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public final class CandidateContextSlotTest {

  @DataProvider
  public Object[][] dtypes() {
    return new Object[][] {
      {DataType.FLOAT32, DataType.FLOAT32},
      {DataType.BFLOAT16, DataType.BFLOAT16},
      {DataType.FLOAT32, DataType.BFLOAT16}
    };
  }

  /** 片方の出力を借用したまま、もう片方の枠で全候補幅と端数バッチを切り替える。 */
  @Test(groups = "rocm", dataProvider = "dtypes")
  public void bucketSwitchPreservesOtherOutstandingOutput(DataType scoresType, DataType valuesType)
      throws Exception {
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(), "PyTorch");
        var execution =
            new DecisionPolicyFusionCandidateContextExecution(manager, 32, valuesType, 5, 2);
        NDManager heldManager = manager.newSubManager();
        var heldForward = execution.beginForward(heldManager)) {
      var held = pool(heldManager, heldForward, 5, 32, scoresType, valuesType);
      float[] heldValues = held.alternativeContexts().toFloatArray();
      heldForward.seal();
      int batch = 0;
      for (int capacity : DecisionInputSchema.LEGAL_ACTION_BUCKETS) {
        try (NDManager working = manager.newSubManager();
            var forward = execution.beginForward(working)) {
          Assert.assertThrows(IllegalStateException.class, () -> execution.beginForward(working));
          var actual = pool(working, forward, 1 + batch++ % 5, capacity, scoresType, valuesType);
          forward.seal();
          // D2H完了後にだけ利用権を返す。別枠のセッション差し替えで保持中の出力は変わらない。
          actual.alternativeContexts().toFloatArray();
          Assert.assertEquals(held.alternativeContexts().toFloatArray(), heldValues);
        }
      }
      Assert.assertFalse(execution.hasIncompleteWork());
      Assert.assertNull(execution.failure());
    }
  }

  private static DecisionPolicyContextPool.MappedCandidateContexts pool(
      NDManager manager,
      DecisionPolicyCandidateContextExecution.Forward forward,
      int rows,
      int capacity,
      DataType scoresType,
      DataType valuesType) {
    int groups = Action.Type.values().length;
    float[] scores = new float[rows * capacity];
    float[] values = new float[rows * capacity * 32];
    float[] masks = new float[rows * capacity * groups];
    for (int row = 0; row < rows; row++) {
      for (int action = 0; action < capacity; action++) {
        int item = row * capacity + action;
        scores[item] = (action % 7 - 3) * 0.25f;
        for (int column = 0; column < 32; column++) {
          values[item * 32 + column] = (row + action + column % 4) * 0.125f;
        }
        // 空集合と末尾のパディングを含め、同じ種類に複数候補がある集約も通す。
        if (row != 1 && (action + 1 < capacity || capacity == 1)) {
          masks[item * groups + (action / 2 + row) % groups] = 1;
        }
      }
    }
    NDArray embeddings =
        manager.create(values).reshape(rows, capacity, 32).toType(valuesType, false);
    NDArray candidateScores =
        manager.create(scores).reshape(rows, capacity).toType(scoresType, false);
    var candidateMasks =
        new DecisionPolicyCandidates.CandidateMasks(
            manager.create(masks).reshape(rows, capacity, groups).toType(valuesType, false), null);
    var expected =
        DecisionPolicyContextPool.poolMappedCandidateTypes(
            embeddings, candidateScores, candidateMasks);
    var actual = forward.pool(embeddings, candidateScores, candidateMasks);
    Assert.assertEquals(actual.alternativeContexts().getDataType(), DataType.FLOAT32);
    Assert.assertEquals(actual.alternativePresence().getDataType(), valuesType);
    assertClose(actual.alternativeContexts(), expected.alternativeContexts());
    assertClose(actual.alternativePresence(), expected.alternativePresence());
    assertClose(actual.passContexts(), expected.passContexts());
    assertClose(actual.passPresence(), expected.passPresence());
    assertClose(actual.context(DecisionAlternative.RON), expected.context(DecisionAlternative.RON));
    assertClose(
        actual.context(DecisionAlternative.KYUSHU), expected.context(DecisionAlternative.KYUSHU));
    assertClose(
        actual.present(DecisionAlternative.KYUSHU), expected.present(DecisionAlternative.KYUSHU));
    assertClose(
        actual.context(DecisionAlternative.TSUMO), expected.context(DecisionAlternative.TSUMO));
    return actual;
  }

  private static void assertClose(NDArray actual, NDArray expected) {
    Assert.assertEquals(actual.getShape(), expected.getShape());
    float[] got = actual.toType(DataType.FLOAT32, false).toFloatArray();
    float[] want = expected.toType(DataType.FLOAT32, false).toFloatArray();
    for (int index = 0; index < want.length; index++) {
      Assert.assertEquals(got[index], want[index], 2e-5f, "element=" + index);
    }
  }
}
