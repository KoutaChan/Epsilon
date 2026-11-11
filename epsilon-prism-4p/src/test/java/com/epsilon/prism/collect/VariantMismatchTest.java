package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.TwinDiscard;
import java.io.IOException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 四人打ちでないルールを、コホート、Φ、半荘の記録のどこから渡されても拒むことを確かめる。 */
public class VariantMismatchTest {

  private static void assertRefused(Variant cohort, Variant phi) throws IOException {
    var plan =
        new Collector.Cohort(
            cohort, 1, 2, List.of(10L), DrawOrder.SEAT_STREAMS, Sampling.OFF, null, CollectorTest.URA,
            new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE), TwinDiscard.TSUMOGIRI, null);
    try (var table = new TableLoop(new TableSettings(1, 1, 1000L, 4));
        var temporary = new CollectorTest.TemporaryCohort(null)) {
      IllegalStateException refused =
          Assert.expectThrows(
              IllegalStateException.class,
              () ->
                  Collector.collect(
                      table,
                      plan,
                      new CollectorTest.HashedScorePolicy(),
                      Phi.scoreRank(phi, RankUtility.TENHOU),
                      temporary.sink));
      Assert.assertTrue(refused.getMessage().startsWith("Variant mismatch: "), refused.getMessage());
    }
  }

  @Test
  public void collectionRefusesOtherVariants() throws IOException {
    assertRefused(TENHOU_3P, TENHOU_3P);
    assertRefused(TENHOU_4P, TENHOU_3P);
    assertRefused(TENHOU_3P, TENHOU_4P);
  }

  @Test
  public void gameRecordsRefuseMixedVariants() {
    GrpState three = new GrpState(TENHOU_3P, 1, 0, 0, new int[] {35000, 35000, 35000});
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new GameRecord(TENHOU_4P, 1, List.of(three), new int[] {25000, 25000, 25000, 25000}));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> new GameRecord(TENHOU_4P, 1, List.of(), new int[] {35000, 35000, 35000}));
  }
}
