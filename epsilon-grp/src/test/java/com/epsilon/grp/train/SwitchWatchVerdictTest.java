package com.epsilon.grp.train;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * GRP を切り替えた後の監視の判定を確かめる。重ならない 8 行の平均が +2 SE（SE = √Σ SE² / 8）を超えたらアラートを出し、モデルを
 * 戻す。τ を 2 倍にしている間にまたアラートが出たら、再学習を止める。行の作り方も確かめる。
 */
public class SwitchWatchVerdictTest {

  private static final double SE = 2e-5;

  @Test
  public void quietWatchesKeepTheRefits() {
    Assert.assertEquals(SwitchWatchVerdict.judge(lines(0, SE), false), SwitchWatchVerdict.QUIET);
    Assert.assertEquals(SwitchWatchVerdict.judge(lines(-3e-5, SE), true), SwitchWatchVerdict.QUIET);
  }

  @Test
  public void aWorseWatchRollsBackAndThenHalts() {
    // 8 行の平均の SE は √8 × 2e-5 / 8 ≈ 7.07e-6 なので、アラートのしきい値は ≈ 1.414e-5
    Assert.assertEquals(SwitchWatchVerdict.judge(lines(1.5e-5, SE), false), SwitchWatchVerdict.ROLLBACK);
    Assert.assertEquals(SwitchWatchVerdict.judge(lines(1.5e-5, SE), true), SwitchWatchVerdict.HALT);
    Assert.assertEquals(SwitchWatchVerdict.judge(lines(1.3e-5, SE), false), SwitchWatchVerdict.QUIET);
  }

  @Test
  public void oneBadCohortIsAveragedWithTheOthers() {
    List<SwitchWatchLine> lines = new ArrayList<>(lines(0, SE));
    lines.set(3, new SwitchWatchLine(13, "used", "previous", 8e-5, SE, 0.01));
    Assert.assertEquals(SwitchWatchVerdict.judge(lines, false), SwitchWatchVerdict.QUIET);
    lines.set(5, new SwitchWatchLine(15, "used", "previous", 8e-5, SE, 0.01));
    Assert.assertEquals(SwitchWatchVerdict.judge(lines, false), SwitchWatchVerdict.ROLLBACK);
  }

  @Test
  public void verdictsNeedEightAscendingLines() {
    Assert.assertThrows(
        IllegalArgumentException.class, () -> SwitchWatchVerdict.judge(lines(0, SE).subList(0, 7), false));
    List<SwitchWatchLine> swapped = new ArrayList<>(lines(0, SE));
    SwitchWatchLine held = swapped.get(2);
    swapped.set(2, swapped.get(3));
    swapped.set(3, held);
    Assert.assertThrows(IllegalArgumentException.class, () -> SwitchWatchVerdict.judge(swapped, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> new SwitchWatchLine(1, "a", "b", Double.NaN, SE, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> new SwitchWatchLine(1, "a", "b", 0, -SE, 0));
    Assert.assertThrows(IllegalArgumentException.class, () -> new SwitchWatchLine(1, "a", "b", 0, SE, Double.NaN));
  }

  @Test
  public void linesCompareTheUsedModelWithThePreviousOne() {
    HandEnds ends = HandEnds.of(
        Variant.TENHOU_4P, RankUtility.TENHOU, TrainFixtures.uniform(2_000, 6, new SplittableRandom(41)));
    HandEndValues previous = shifted(ends, 0);
    HandEndValues used = shifted(ends, 0.05);
    SwitchWatchLine line = SwitchWatchLine.of(44, "g044", "g043", ends, used, previous);
    Assert.assertEquals(line.cohort(), 44);
    Assert.assertEquals(line.usedId(), "g044");
    Assert.assertEquals(line.mseDifference(), GrpMetrics.mse(ends, used) - GrpMetrics.mse(ends, previous), 1e-12);
    Assert.assertTrue(line.mseDifference() > 0);
    Assert.assertEquals(line.wobble(), 0.05, 1e-12);
  }

  /** コホート 10〜17 の 8 行を作る。 */
  private static List<SwitchWatchLine> lines(double difference, double standardError) {
    List<SwitchWatchLine> lines = new ArrayList<>();
    for (int cohort = 10; cohort < 10 + SwitchWatchVerdict.LINES; cohort++) {
      lines.add(new SwitchWatchLine(cohort, "used", "previous", difference, standardError, 0.01));
    }
    return lines;
  }

  private static HandEndValues shifted(HandEnds ends, double shift) {
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    double[] uniform = new double[16];
    Arrays.fill(uniform, 0.25);
    double[] phi = new double[4];
    Arrays.fill(phi, shift);
    for (int h = 0; h < ends.handEnds(); h++) {
      builder.set(h, phi, uniform);
    }
    return builder.build();
  }
}
