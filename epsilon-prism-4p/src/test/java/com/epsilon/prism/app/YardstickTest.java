package com.epsilon.prism.app;

import org.testng.Assert;
import org.testng.annotations.Test;

public class YardstickTest {
  @Test
  public void pairwiseRatingUsesTheCandidatePlacementAndItsClusteredError() {
    Assert.assertEquals(Yardstick.elo(4, 2.5, 0.01), "+0.0000 +/- 2.3162");
    Assert.assertEquals(Yardstick.elo(4, 2, 0), "+120.4120 +/- 0.0000");
    Assert.assertEquals(Yardstick.elo(4, 3, 0), "-120.4120 +/- 0.0000");
    Assert.assertEquals(Yardstick.elo(4, 1, 0), "+infinity (SE undefined)");
    Assert.assertEquals(Yardstick.elo(4, 4, 0), "-infinity (SE undefined)");
    Assert.expectThrows(IllegalArgumentException.class, () -> Yardstick.elo(4, Double.NaN, 0));
    Assert.assertEquals(Yardstick.elo(3, 2, 0.01), "+0.0000 +/- 3.4744");
    Assert.assertEquals(Yardstick.elo(3, 1.5, 0), "+190.8485 +/- 0.0000");
    Assert.assertEquals(Yardstick.elo(3, 3, 0), "-infinity (SE undefined)");
    Assert.expectThrows(IllegalArgumentException.class, () -> Yardstick.elo(3, 3.5, 0));
  }
}
