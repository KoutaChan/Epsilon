package com.epsilon.ai.decision.duel;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 既知の標本統計と観測範囲で、共有正規分布信頼区間を検証する。 */
public class EpsilonDecisionFixedSampleConfidenceIntervalTest {

  @DataProvider
  public Object[][] knownConfidenceBounds() {
    return new Object[][] {
      {0.05, 1.23484868811834, 3.76515131188166},
      {0.01, 0.8373093341136766, 4.162690665886323}
    };
  }

  @Test(dataProvider = "knownConfidenceBounds")
  public void knownSamplesProduceUnbiasedStandardErrorAndGaussianBounds(
      double alpha, double lower, double upper) {
    var interval = new EpsilonDecisionFixedSampleConfidenceInterval(0.0, 5.0);
    interval.add(1.0);
    interval.add(2.0);
    interval.add(3.0);
    interval.add(4.0);

    var estimate = interval.estimate(alpha);
    Assert.assertEquals(estimate.samples(), 4);
    Assert.assertEquals(estimate.mean(), 2.5, 1.0e-12);
    Assert.assertEquals(estimate.standardError(), 0.6454972243679028, 1.0e-12);
    Assert.assertEquals(estimate.lower(), lower, 1.0e-8);
    Assert.assertEquals(estimate.upper(), upper, 1.0e-8);
  }

  @Test
  public void oneObservationRetainsTheFullKnownRange() {
    var interval = new EpsilonDecisionFixedSampleConfidenceInterval(-2.0, 2.0);
    interval.add(1.5);

    var estimate = interval.estimate(0.05);
    Assert.assertEquals(estimate.mean(), 1.5);
    Assert.assertEquals(estimate.lower(), -2.0);
    Assert.assertEquals(estimate.upper(), 2.0);
  }

  @Test
  public void wideIntervalClipsToTheKnownObservationBounds() {
    var interval = new EpsilonDecisionFixedSampleConfidenceInterval(-2.0, 2.0);
    interval.add(-2.0);
    interval.add(2.0);

    var estimate = interval.estimate(0.05);
    Assert.assertEquals(estimate.mean(), 0.0);
    Assert.assertEquals(estimate.standardError(), 2.0);
    Assert.assertEquals(estimate.lower(), -2.0);
    Assert.assertEquals(estimate.upper(), 2.0);
  }

  @DataProvider
  public Object[][] endpoints() {
    return new Object[][] {{-2.0}, {2.0}};
  }

  @Test(dataProvider = "endpoints")
  public void repeatedEndpointHasZeroVarianceAndAnExactPointInterval(double endpoint) {
    var interval = new EpsilonDecisionFixedSampleConfidenceInterval(-2.0, 2.0);
    interval.add(endpoint);
    interval.add(endpoint);
    interval.add(endpoint);

    var estimate = interval.estimate(0.05);
    Assert.assertEquals(estimate.mean(), endpoint);
    Assert.assertEquals(estimate.standardError(), 0.0);
    Assert.assertEquals(estimate.lower(), endpoint);
    Assert.assertEquals(estimate.upper(), endpoint);
  }
}
