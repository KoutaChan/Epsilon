package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.runtime.PinnedUploads;
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
}
