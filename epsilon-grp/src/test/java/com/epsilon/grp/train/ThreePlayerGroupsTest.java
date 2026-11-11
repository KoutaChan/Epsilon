package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの物差しの区分と、三人打ちの効用 [1, 0, −1] での Φ を確かめる。区分は終わった局で決め、東 1〜3 局、南 1・2 局と、
 * 南 3 局以降（西入を含む）をまとめた S3+ の 6 つ。終盤の 2 区分は S2 と S3+。どの Φ の項目でも、席ごとの値は効用の最小と最大の
 * 間にあり、席の和は効用の和（0）になる。
 */
public class ThreePlayerGroupsTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final RankUtility UTILITY = RankUtility.TENHOU;

  /** 三人打ちの区分の名前（区分の番号の順）。 */
  private static final List<String> NAMES = List.of("E1", "E2", "E3", "S1", "S2", "S3+");

  /** 区分の名前を作る。場の頭文字に局の番号を続け、最後の区分には + を付ける。 */
  private static String name(Variant variant, int group) {
    String name = "ESW".charAt(group / variant.seats()) + Integer.toString(group % variant.seats() + 1);
    return group == variant.lastRound() ? name + "+" : name;
  }

  @Test
  public void threePlayerGroupsAreE1ToS3Plus() {
    // 東 1 局から西 3 局の前まで 1 局ずつ進む半荘。局末 k で終わった局は k
    List<GrpState> handEnds = new ArrayList<>();
    for (int round = 1; round <= THREE.finalRound(); round++) {
      handEnds.add(new GrpState(THREE, round, 0, 0, new int[] {40_000, 35_000, 30_000}));
    }
    GameRecord game = new GameRecord(THREE, 1, handEnds, new int[] {50_000, 35_000, 20_000});
    HandEnds ends = HandEnds.of(THREE, UTILITY, List.of(game));
    Assert.assertEquals(ends.groups(), NAMES.size());
    List<String> names = new ArrayList<>();
    for (int group = 0; group < ends.groups(); group++) {
      names.add(name(THREE, group));
    }
    Assert.assertEquals(names, NAMES);
    List<String> byHand = new ArrayList<>();
    for (int h = 0; h < ends.handEnds(); h++) {
      Assert.assertEquals(game.endedRound(h), h);
      byHand.add(NAMES.get(ends.group(h)));
    }
    // 南 3 局、西 1 局、西 2 局が終わった局末は S3+
    Assert.assertEquals(byHand, List.of("E1", "E2", "E3", "S1", "S2", "S3+", "S3+", "S3+"));
  }

  /**
   * 物差しの区分は 6 つで、全ての席と局末をどれかに分ける。僅差のセルは、終盤の 2 区分（S2・S3+）での最下位と、その 1 つ上の席
   * だけ。
   */
  @Test
  public void theReportUsesTheThreePlayerGroups() {
    List<GameRecord> games = TrainFixtures.simulated(THREE, 1_500, 0, 53);
    HandEnds ends = HandEnds.of(THREE, UTILITY, games);
    GrpReport report = GrpMetrics.measure(ends, HandEndValues.of(ends, Phi.scoreRank(THREE, UTILITY)));
    Assert.assertEquals(report.groups().size(), NAMES.size());
    long[] counted = new long[NAMES.size()];
    for (int h = 0; h < ends.handEnds(); h++) {
      counted[ends.group(h)] += THREE.seats();
    }
    long total = 0;
    for (int group = 0; group < NAMES.size(); group++) {
      Assert.assertEquals(report.groups().get(group).seatHands(), counted[group], NAMES.get(group));
      Assert.assertTrue(counted[group] > 0, NAMES.get(group) + " has no hand ends");
      total += counted[group];
    }
    Assert.assertEquals(total, ends.seatHands());
    Assert.assertEquals(report.seatHands(), ends.seatHands());
    Assert.assertEquals(report.closeCells().size(), 8);
    Set<String> groups = new TreeSet<>();
    Set<Integer> ranks = new TreeSet<>();
    for (GrpReport.CloseCell cell : report.closeCells()) {
      groups.add(NAMES.get(cell.group()));
      ranks.add(cell.rank());
    }
    Assert.assertEquals(groups, Set.of("S2", "S3+"));
    Assert.assertEquals(ranks, Set.of(1, 2));
    Assert.assertTrue(report.tails().minPhi() >= -2 && report.tails().maxPhi() <= 2, report.tails().toString());
  }

  /** 点数の順位、ネットワーク、終盤の数え上げのどの Φ も、三人打ちの効用では [−2, 2] に入り、席の和が 0 になる。 */
  @Test
  public void phiStaysWithinTheThreePlayerUtilities() {
    List<GameRecord> games = TrainFixtures.simulated(THREE, 300, 0, 59);
    HandEnds ends = HandEnds.of(THREE, UTILITY, games);
    GrpNetwork leaf = GrpFixtures.network(GrpFixtures.shape(THREE, 12, false), 61, 1.0);
    OutcomeShape outcomes = new OutcomeShape(THREE, GrpFixtures.plainFields(), 8);
    FinalRound late = FinalRound.of(GrpFixtures.outcomes(outcomes, 67, 0.5), leaf, 1e-9);
    Map<String, Phi> phis =
        Map.of(
            "score rank", Phi.scoreRank(THREE, UTILITY),
            "network", Phi.of(leaf, UTILITY),
            "late", Phi.late(late, UTILITY, 64));
    double lowest = UTILITY.value(THREE, 2);
    double highest = UTILITY.value(THREE, 0);
    double sum = UTILITY.value(THREE, 0) + UTILITY.value(THREE, 1) + UTILITY.value(THREE, 2);
    Assert.assertEquals(lowest, -1.0);
    Assert.assertEquals(highest, 1.0);
    Assert.assertEquals(sum, 0.0);
    for (Map.Entry<String, Phi> entry : phis.entrySet()) {
      HandEndValues values = HandEndValues.of(ends, entry.getValue());
      Assert.assertTrue(values.finite(), entry.getKey());
      for (int h = 0; h < ends.handEnds(); h++) {
        double seats = 0;
        for (int seat = 0; seat < 3; seat++) {
          double phi = values.phi(h, seat);
          Assert.assertTrue(
              phi >= lowest - 1e-12 && phi <= highest + 1e-12, entry.getKey() + " Φ " + phi + " at " + ends.state(h));
          seats += phi;
          double places = 0;
          double expected = 0;
          for (int place = 0; place < 3; place++) {
            places += values.marginal(h, seat, place);
            expected += values.marginal(h, seat, place) * UTILITY.value(THREE, place);
          }
          Assert.assertEquals(places, 1, 1e-9, entry.getKey() + " marginals at " + ends.state(h));
          Assert.assertEquals(phi, expected, 1e-9, entry.getKey() + " Φ is the expected utility at " + ends.state(h));
        }
        Assert.assertEquals(seats, sum, 1e-9, entry.getKey() + " ΣΦ at " + ends.state(h));
      }
    }
  }
}
