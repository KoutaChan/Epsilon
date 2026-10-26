package com.epsilon.spi;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 牌譜解析の評価の検証、softmax の正規化、損失を確かめる。 */
public class ReviewEvaluationTest {
  @DataProvider(name = "invalidProbabilities")
  public Object[][] invalidProbabilities() {
    return new Object[][] {
      {new float[] {}},
      {new float[] {0.5f, 0.4f}},
      {new float[] {0.6f, 0.6f}},
      {new float[] {1.5f, -0.5f}},
      {new float[] {Float.NaN, 1}},
      {new float[] {Float.POSITIVE_INFINITY, 0}}
    };
  }

  @Test(dataProvider = "invalidProbabilities")
  public void rejectsProbabilitiesThatAreNotADistribution(float[] probabilities) {
    Assert.expectThrows(
        IllegalArgumentException.class, () -> ReviewEvaluation.policy(probabilities));
  }

  @Test
  public void rejectsNonFiniteValuesAndALengthMismatch() {
    float[] probabilities = {0.25f, 0.75f};
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReviewEvaluation.valued(new float[] {0, Float.NaN}, probabilities));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReviewEvaluation.valued(new float[] {0, Float.NEGATIVE_INFINITY}, probabilities));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReviewEvaluation.valued(new float[] {0, 1, 2}, probabilities));
    Assert.expectThrows(
        NullPointerException.class, () -> ReviewEvaluation.valued(null, probabilities));
  }

  @Test
  public void softmaxIsNormalizedOrderedAndStableForLargeValues() {
    float[] probabilities = ReviewEvaluation.softmax(new float[] {1000, 1000.02f, 999.9f}, 0.02);
    double sum = 0;
    for (float probability : probabilities) sum += probability;
    Assert.assertEquals(sum, 1.0, 1e-6);
    Assert.assertTrue(probabilities[1] > probabilities[0]);
    Assert.assertTrue(probabilities[0] > probabilities[2]);
    // 0.02 劣る手は最善手の約 e^-1 倍。
    Assert.assertEquals(probabilities[0] / probabilities[1], Math.exp(-1), 1e-2);
    Assert.assertEquals(ReviewEvaluation.softmax(new float[] {3, 3}, 1), new float[] {0.5f, 0.5f});
    Assert.expectThrows(
        IllegalArgumentException.class, () -> ReviewEvaluation.softmax(new float[] {0}, 0));
  }

  @Test
  public void lossIsTheGapToTheBestValue() {
    float[] values = {0.1f, 0.4f, -0.2f};
    var evaluation = ReviewEvaluation.valued(values, ReviewEvaluation.softmax(values, 0.1));
    Assert.assertTrue(evaluation.valued());
    Assert.assertEquals(evaluation.actions(), 3);
    float[] losses = evaluation.losses();
    Assert.assertEquals(losses[1], 0f);
    Assert.assertEquals(losses[0], 0.3f, 1e-6f);
    Assert.assertEquals(losses[2], 0.6f, 1e-6f);
    var policy = ReviewEvaluation.policy(new float[] {1});
    Assert.assertFalse(policy.valued());
    Assert.expectThrows(IllegalStateException.class, policy::losses);
  }
}
