package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * {@link ValueGraphHead#aggregate} のテスト。遷移が 1 つの手をそのまま使い、残りの手だけをパディングして集約しても、全ての手を同じ幅に
 * パディングして重み付きの和を取る式とビットで同じになる。遷移が 1 つの手、連続した遷移の手、遷移を配列で渡した手、幅の下限が手の遷移の
 * 最大の数より大きいときを確かめる。
 */
public class ValueGraphHeadTest {

  private static final int TRANSITIONS = 40;
  private static final int HIDDEN = 8;

  @Test(timeOut = 120_000)
  public void aggregateMatchesPaddedSum() {
    int[][] members = {{3}, {5, 6, 7}, {11}, {20, 21}, {0}, {9, 14, 30, 2}, {39}};
    for (int minimumWidth : new int[] {1, 6}) {
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Random random = new Random(7 + minimumWidth);
        float[] z = new float[TRANSITIONS * HIDDEN];
        float[] b = new float[TRANSITIONS];
        for (int i = 0; i < z.length; i++) {
          z[i] = (float) random.nextGaussian();
        }
        for (int i = 0; i < b.length; i++) {
          b[i] = (float) random.nextGaussian() * 3f;
        }
        NDArray transitions = manager.create(z, new Shape(TRANSITIONS, HIDDEN));
        NDArray values = manager.create(b);

        Staging staging = new Staging();
        BatchIndex.Aggregation.Builder builder = new BatchIndex.Aggregation.Builder(members.length);
        for (int[] member : members) {
          if (member.length > 1 && member[1] == member[0] + 1) {
            builder.add(member[0], member.length);
          } else {
            builder.add(member, 0, member.length);
          }
        }
        BatchIndex.Aggregation aggregation = builder.build(minimumWidth, staging);
        staging.send(PinnedUploads.on(manager), manager);
        Assert.assertEquals(aggregation.directCount, 4);
        Assert.assertEquals(aggregation.width, Math.max(4, minimumWidth));

        float[] actual = ValueGraphHead.aggregate(transitions, values, aggregation).toFloatArray();
        float[] expected =
            padded(transitions, values, members, aggregation.width, manager).toFloatArray();
        Assert.assertEquals(actual.length, members.length * HIDDEN);
        for (int i = 0; i < actual.length; i++) {
          Assert.assertEquals(
              Float.floatToIntBits(actual[i]), Float.floatToIntBits(expected[i]), "at " + i);
        }
      }
    }
  }

  /** 全ての手を同じ幅にパディングして求めた e = Σ_t sg(softmax_t(B(t)/τ))·z_t。パディングは遷移 0 を指し、マスクは 0。 */
  private static NDArray padded(
      NDArray transitions, NDArray values, int[][] members, int width, NDManager manager) {
    long[] chosen = new long[members.length * width];
    float[] mask = new float[members.length * width];
    for (int m = 0; m < members.length; m++) {
      for (int t = 0; t < members[m].length; t++) {
        chosen[m * width + t] = members[m][t];
        mask[m * width + t] = 1f;
      }
    }
    NDArray index = manager.create(chosen);
    NDArray logits =
        NDArrays.gatherRows(values, index).reshape(members.length, width).div(ValueGraphHead.TAU);
    NDArray weights =
        NDArrays.maskedSoftmax(logits, manager.create(mask, new Shape(members.length, width)), 1);
    NDArray rows = NDArrays.gatherRows(transitions, index).reshape(members.length, width, HIDDEN);
    return rows.mul(weights.expandDims(2)).sum(new int[] {1});
  }

  /**
   * 証明辺のあるツモの行だけ {@code peak}（打牌・立直の最大 M）に {@code stopGradient} を掛ける VJP の確かめ。順伝播はビット同一、
   * 証明辺ツモの行では和了の損失から M へ勾配が流れず（Δ には 1、打牌自身の損失からは 1）、同じバッチ内の非証明行では M へ勾配 1 が
   * 流れる。証明辺の無いバッチでは {@link BatchIndex#certifiedTurnRows} が null になり従来の道と同一になる。
   */
  @Test(timeOut = 120_000)
  public void certifiedTurnRowsStopPeakGradientWithoutChangingForward() {
    Batches.Batch batch = Batches.build(List.of(Batches.turn(2), Batches.turn(2)), 13L);
    boolean[] measured = new boolean[batch.actions()];
    Arrays.fill(measured, true);
    float[] rawValues = new float[batch.transitions()];
    Arrays.fill(rawValues, 0.1f);
    // 各行で行動 1（遷移 1 と 8）を唯一の最大 M にする
    rawValues[1] = 2.0f;
    rawValues[8] = 2.0f;
    float[] rawDelta = new float[8];
    Arrays.fill(rawDelta, -0.5f);
    rawDelta[2] = 0.7f; // 行 0 の TSUMO_AGARI
    rawDelta[6] = 0.9f; // 行 1 の TSUMO_AGARI

    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      NetworkInputs inputs = batch.inputs(manager);
      BatchIndex legacyIndex =
          BatchIndex.send(inputs, Pass.training(Precision.FLOAT32, measured, null, null, null));
      BatchIndex emptyCertIndex =
          BatchIndex.send(
              inputs,
              Pass.training(Precision.FLOAT32, measured, null, null, new boolean[] {false, false}));
      BatchIndex certIndex =
          BatchIndex.send(
              inputs,
              Pass.training(Precision.FLOAT32, measured, null, null, new boolean[] {true, false}));
      Assert.assertNull(legacyIndex.certifiedTurnRows);
      Assert.assertNull(emptyCertIndex.certifiedTurnRows);
      Assert.assertNotNull(certIndex.certifiedTurnRows);
      Assert.assertEquals(certIndex.gateCount, 8);

      float[] legacyForward =
          ValueGraphHead.compose(
                  manager, legacyIndex, manager.create(rawValues), manager.create(rawDelta))
              .toFloatArray();
      float[] certForward =
          ValueGraphHead.compose(
                  manager, certIndex, manager.create(rawValues), manager.create(rawDelta))
              .toFloatArray();
      Assert.assertEquals(certForward.length, legacyForward.length);
      for (int i = 0; i < legacyForward.length; i++) {
        Assert.assertEquals(
            Float.floatToRawIntBits(certForward[i]),
            Float.floatToRawIntBits(legacyForward[i]),
            "forward at " + i);
      }

      // 両行の TSUMO_AGARI（スロット 5）の和から逆伝播する
      NDArray values = manager.create(rawValues);
      NDArray delta = manager.create(rawDelta);
      values.setRequiresGradient(true);
      delta.setRequiresGradient(true);
      try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
        NDArray scores = ValueGraphHead.compose(manager, certIndex, values, delta);
        NDArray loss = scores.get("0, 5").add(scores.get("1, 5"));
        gc.backward(loss);
      }
      float[] vGrad = values.getGradient().toFloatArray();
      float[] dGrad = delta.getGradient().toFloatArray();
      Assert.assertEquals(vGrad[1], 0f, 0f, "certified row peak gradient must be blocked");
      Assert.assertEquals(vGrad[8], 1f, 1e-6f, "uncertified row peak gradient must flow");
      Assert.assertEquals(dGrad[2], 1f, 1e-6f, "certified row tsumo delta gradient");
      Assert.assertEquals(dGrad[6], 1f, 1e-6f, "uncertified row tsumo delta gradient");

      // 証明辺ツモ行でも打牌自身のスコアからは B へ勾配 1 が流れる
      NDArray valuesDiscard = manager.create(rawValues);
      NDArray deltaDiscard = manager.create(rawDelta);
      valuesDiscard.setRequiresGradient(true);
      deltaDiscard.setRequiresGradient(true);
      try (GradientCollector gc = manager.getEngine().newGradientCollector()) {
        NDArray scores = ValueGraphHead.compose(manager, certIndex, valuesDiscard, deltaDiscard);
        gc.backward(scores.get("0, 1"));
      }
      Assert.assertEquals(valuesDiscard.getGradient().toFloatArray()[1], 1f, 1e-6f);
    }
  }
}
