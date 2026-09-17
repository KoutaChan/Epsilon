package com.epsilon.ai.belief;

import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Beliefの教師配列と推論結果の受け渡し、バッチからの行抽出、予測値の意味と有限性を検証する。 */
public class EpsilonBeliefValuesTest {

  @Test
  public void targetTakesGeneratedArraysAndWritesDirectlyAtBatchOffset() {
    float[] hands = new float[102];
    float[] hidden = new float[34];
    float[] shanten = {2, 0, 1};
    float[] tenpai = {0, 1, 0};
    float[] waits = new float[102];
    hands[0] = 3;
    hands[101] = 1;
    hidden[0] = 4;
    waits[34] = 1;
    EpsilonBeliefTarget target = new EpsilonBeliefTarget(hands, hidden, shanten, tenpai, waits);
    Assert.assertSame(target.opponentHandCounts(), hands);
    Assert.assertSame(target.hiddenTileCounts(), hidden);
    Assert.assertSame(target.opponentShanten(), shanten);
    Assert.assertSame(target.opponentTenpai(), tenpai);
    Assert.assertSame(target.opponentWaitMask(), waits);

    float[] batch = new float[217];
    Arrays.fill(batch, -7);
    target.writeTo(batch, 3);
    Assert.assertEquals(batch[2], -7.0f);
    Assert.assertEquals(batch[3], 3.0f);
    Assert.assertEquals(batch[104], 1.0f);
    Assert.assertEquals(batch[105], 0.0f);
    Assert.assertEquals(batch[139], 1.0f);
    Assert.assertEquals(Arrays.copyOfRange(batch, 207, 213), new float[] {2, 0, 0, 1, 1, 0});
    Assert.assertEquals(batch[213], -7.0f);
    Assert.assertEquals(batch[216], -7.0f);
  }

  @Test
  public void priorExtractsOneRowWithoutRetainingBatchOutput() {
    float[] batch = new float[425];
    Arrays.fill(batch, -3);
    int offset = 213;
    batch[offset] = 1.5f;
    batch[offset + 101] = 2.5f;
    batch[offset + 102] = 3.5f;
    batch[offset + 203] = 4.5f;
    batch[offset + 204] = 5.5f;
    batch[offset + 209] = 6.5f;
    EpsilonBeliefPrior prior = EpsilonBeliefPrior.fromFlatOutput(batch, offset);
    Arrays.fill(batch, 99);
    Assert.assertEquals(prior.opponentHandLogits()[0], 1.5f);
    Assert.assertEquals(prior.opponentHandLogits()[101], 2.5f);
    Assert.assertEquals(prior.opponentWaitLogits()[0], 3.5f);
    Assert.assertEquals(prior.opponentWaitLogits()[101], 4.5f);
    Assert.assertEquals(prior.opponentScalar(0, EpsilonBeliefLayout.SCALAR_SHANTEN), 5.5f);
    Assert.assertEquals(prior.opponentScalar(2, EpsilonBeliefLayout.SCALAR_TENPAI), 6.5f);
  }

  @Test
  public void priorPreservesCountTiltWaitLogProbabilityAndTenpaiLogit() {
    float[] hands = new float[102];
    float[] waits = new float[102];
    hands[0] = 2;
    waits[0] = 1;
    EpsilonBeliefTarget target =
        new EpsilonBeliefTarget(
            hands, new float[34], new float[] {2, 0, 1}, new float[] {0, 1, 0}, waits);
    EpsilonBeliefPrior prior = EpsilonBeliefPrior.fromTarget(target);
    Assert.assertEquals(prior.opponentHandLogits()[0], (float) Math.log(2.001f), 1.0e-6f);
    Assert.assertEquals(prior.opponentHandLogits()[1], (float) Math.log(0.001f), 1.0e-6f);
    Assert.assertEquals(prior.opponentWaitLogits()[0], 0.0f);
    Assert.assertEquals(prior.opponentWaitLogits()[1], (float) Math.log(1.0e-4f), 1.0e-6f);
    Assert.assertEquals(prior.opponentScalar(0, EpsilonBeliefLayout.SCALAR_SHANTEN), 2.0f);
    Assert.assertEquals(
        prior.opponentScalar(0, EpsilonBeliefLayout.SCALAR_TENPAI), -9.21024f, 1.0e-4f);
    Assert.assertEquals(
        prior.opponentScalar(1, EpsilonBeliefLayout.SCALAR_TENPAI), 9.21007f, 1.0e-4f);
  }

  @Test
  public void priorRejectsNonFiniteNetworkOutput() {
    float[] output = new float[210];
    output[102] = Float.NaN;
    Assert.expectThrows(
        IllegalArgumentException.class, () -> EpsilonBeliefPrior.fromFlatOutput(output, 0));
  }
}
