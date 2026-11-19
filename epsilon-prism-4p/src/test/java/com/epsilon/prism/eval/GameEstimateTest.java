package com.epsilon.prism.eval;

import com.epsilon.prism.core.DecisionClass;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class GameEstimateTest {
  @Test
  public void clusterResidualUsesTheGameSize() {
    var first = new GameEstimate.Game();
    var second = new GameEstimate.Game();
    first.add(DecisionClass.CALL, 2);
    first.add(DecisionClass.CALL, 4);
    second.add(DecisionClass.THREAT, 0);
    var games = List.of(first, second);
    var overall = GameEstimate.of(games, null);
    Assert.assertEquals(overall.mean(), 2.0);
    Assert.assertEquals(overall.se(), 4.0 / 3, 1e-14);
    Assert.assertNull(GameEstimate.of(games, DecisionClass.KAN).mean());
    Assert.assertNull(GameEstimate.of(List.of(first), null).se());
    Assert.assertNull(GameEstimate.of(games, DecisionClass.CALL).se());
  }
}
