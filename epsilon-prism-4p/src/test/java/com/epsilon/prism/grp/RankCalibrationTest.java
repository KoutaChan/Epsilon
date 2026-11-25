package com.epsilon.prism.grp;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class RankCalibrationTest {
  /** 次局は南場の最後の局の 1 つ前・最後の局・西場に分かれ、席の数はルールから取る。 */
  @Test
  public void boundariesUseTheNextRoundAndTheSettledGap() {
    measure(
        Variant.TENHOU_4P,
        new int[] {30500, 28500, 24500, 16500},
        new int[] {20000, 30000, 32000, 18000});
    measure(Variant.TENHOU_3P, new int[] {38000, 36000, 31000}, new int[] {30000, 35000, 40000});
    Assert.assertEquals(RankCalibration.Gap.of(1999), RankCalibration.Gap.BELOW_2000);
    Assert.assertEquals(RankCalibration.Gap.of(-500), RankCalibration.Gap.BELOW_2000);
    Assert.assertEquals(RankCalibration.Gap.of(2000), RankCalibration.Gap.FROM_2000);
    Assert.assertEquals(RankCalibration.Gap.of(4000), RankCalibration.Gap.FROM_4000);
    Assert.assertEquals(RankCalibration.Gap.of(8000), RankCalibration.Gap.FROM_8000);
  }

  /** 席 0 が 1 位・席 1 が 2 位で、2 位の上との点差は 2,000 点台、下との点差は 4,000 点台。終局では席 0 が 3 位、席 1 が 2 位。 */
  private static void measure(Variant variant, int[] scores, int[] finals) {
    int seats = variant.seats(), last = variant.lastRound();
    var ends =
        List.of(
            new GrpState(variant, last - 1, 0, 0, scores),
            new GrpState(variant, last, 0, 0, scores),
            new GrpState(variant, last + 1, 0, 0, scores));
    var game = new GameRecord(variant, 1L, ends, finals);
    var hands = HandEnds.of(variant, RankUtility.TENHOU, List.of(game));
    var lines =
        RankCalibration.measure(
            HandEndValues.of(hands, Phi.scoreRank(variant, RankUtility.TENHOU)));
    Double[] second = new Double[seats];
    Arrays.fill(second, 0.);
    second[1] = 1.;
    for (var round : RankCalibration.Round.values()) {
      for (var direction : RankCalibration.Direction.values()) {
        var cells =
            lines.stream().filter(l -> l.round() == round && l.direction() == direction).toList();
        Assert.assertEquals(
            cells.stream().mapToLong(RankCalibration.Line::seatHands).sum(), seats - 1L);
        var middle =
            cells.stream()
                .filter(l -> l.rank() == 2 && l.seatHands() > 0)
                .findFirst()
                .orElseThrow();
        Assert.assertEquals(
            middle.gap(),
            direction == RankCalibration.Direction.ABOVE
                ? RankCalibration.Gap.FROM_2000
                : RankCalibration.Gap.FROM_4000);
        Assert.assertEquals(middle.predicted(), second);
        Assert.assertEquals(middle.observed(), second);
      }
      var leader =
          lines.stream()
              .filter(l -> l.round() == round && l.rank() == 1 && l.seatHands() > 0)
              .findFirst()
              .orElseThrow();
      Assert.assertEquals(leader.observed()[2], 1.0);
      Assert.assertEquals(leader.predicted()[2], 0.0);
    }
  }
}
