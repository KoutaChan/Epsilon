package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.FinalRoundWorkspace;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局末の列を確かめる。局末から、その局末を含む半荘を引ける。局末の無い半荘を間に挟んでも、正しい半荘に戻る。選んだ局末の列で
 * 測った指標は、元の列で同じ局末を選んで測った指標と同じになる。数え上げの値は、葉のネットワークと FinalRound の値に合う。
 */
public class HandEndsTest {

  @Test
  public void handEndsMapBackToTheirGames() {
    int[] even = {25_000, 25_000, 25_000, 25_000};
    int[] hands = {0, 2, 0, 0, 3, 1, 0};
    List<GameRecord> games = new ArrayList<>();
    for (int game = 0; game < hands.length; game++) {
      List<GrpState> handEnds = new ArrayList<>();
      for (int hand = 0; hand < hands[game]; hand++) {
        handEnds.add(new GrpState(Variant.TENHOU_4P, hand + 1, 0, 0, even));
      }
      games.add(new GameRecord(Variant.TENHOU_4P, game, handEnds, even));
    }
    HandEnds ends = HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU, games);
    Assert.assertEquals(ends.handEnds(), 6);
    for (int game = 0; game < ends.games(); game++) {
      for (int handEnd = ends.firstHand(game); handEnd < ends.firstHand(game + 1); handEnd++) {
        Assert.assertEquals(ends.game(handEnd), game, "hand end " + handEnd);
      }
    }
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> ends.game(6));
  }

  /** 局末の履歴は、半荘の始めの卓と、同じ半荘でその局末より前にある局末の卓からなる。選んだ列でも、履歴は元の半荘から作る。 */
  @Test
  public void historiesHoldTheRoundStartsOfTheGame() {
    int[] even = {25_000, 25_000, 25_000, 25_000};
    int[] hands = {2, 0, 3};
    List<GameRecord> games = new ArrayList<>();
    for (int game = 0; game < hands.length; game++) {
      List<GrpState> handEnds = new ArrayList<>();
      for (int hand = 0; hand < hands[game]; hand++) {
        handEnds.add(new GrpState(Variant.TENHOU_4P, hand + 1, game, 0, even));
      }
      games.add(new GameRecord(Variant.TENHOU_4P, game, handEnds, even));
    }
    HandEnds ends = HandEnds.of(Variant.TENHOU_4P, RankUtility.TENHOU, games);
    RoundHistory history = new RoundHistory(Variant.TENHOU_4P);
    int[] sizes = {1, 2, 1, 2, 3};
    for (int h = 0; h < ends.handEnds(); h++) {
      ends.history(h, history);
      Assert.assertEquals(history.size(), sizes[h], "hand end " + h);
      Assert.assertEquals(history.round(0), 0);
      Assert.assertEquals(history.score(0, 0), Variant.TENHOU_4P.startPoints());
      for (int index = 1; index < history.size(); index++) {
        GrpState earlier = ends.state(h - history.size() + index);
        Assert.assertEquals(history.round(index), earlier.round());
        Assert.assertEquals(history.honba(index), earlier.honba());
      }
    }
    HandEnds last = ends.select(h -> h == 4);
    last.history(0, history);
    Assert.assertEquals(history.size(), 3);
    Assert.assertEquals(history.round(2), 2);
    Assert.assertEquals(history.honba(2), 2);
  }

  /**
   * 全ての局末を選んだ列で測ると、元の列で測ったのとビット単位で同じになる。数え上げる卓を選んだ列で測ると、元の列で同じ局末を
   * 選んで測ったのとビット単位で同じになる。値をコピーできるのは、その値を求めた列から直接選んだ列だけ。数え上げの値の Φ は、
   * 数え上げない卓では葉のネットワークの Φ と、数え上げる卓では FinalRound の Φ と、ビット単位で同じになる。局末ごとの、半荘が
   * 終わる組の確率の和は、FinalRound の作業領域の値と同じになる。
   */
  @Test
  public void selectedHandEndsMeasureLikeTheChosenHandEnds() {
    Variant variant = Variant.TENHOU_4P;
    HandEnds ends = HandEnds.of(variant, RankUtility.TENHOU, TrainFixtures.simulated(variant, 600, 0, 31));
    GrpNetwork leaf = GrpFixtures.network(GrpFixtures.shape(variant, 8, false), 13, 0.5);
    GrpNetwork other = GrpFixtures.network(GrpFixtures.shape(variant, 8, false), 17, 0.5);
    HandEndValues values = HandEndValues.of(ends, leaf, GrpWorkspace.of(leaf.shape()));
    HandEndValues reference = HandEndValues.of(ends, other, GrpWorkspace.of(other.shape()));

    HandEnds all = ends.select(h -> true);
    Assert.assertEquals(all.handEnds(), ends.handEnds());
    Assert.assertEquals(GrpMetrics.measure(all, values.select(all)), GrpMetrics.measure(ends, values));
    Assert.assertEquals(
        GrpMetrics.pairedMse(all, reference.select(all), values.select(all)),
        GrpMetrics.pairedMse(ends, reference, values));

    IntPredicate covers = h -> FinalRound.covers(ends.state(h));
    HandEnds covered = ends.select(covers);
    Assert.assertTrue(covered.handEnds() > 100 && covered.handEnds() < ends.handEnds(), covered.handEnds() + " covered");
    Assert.assertEquals(GrpMetrics.measure(covered, values.select(covered)), GrpMetrics.measure(ends, values, covers));
    Assert.assertThrows(IllegalArgumentException.class, () -> values.select(covered.select(h -> true)));

    OutcomeShape outcomes = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    FinalRound round = FinalRound.of(GrpFixtures.outcomes(outcomes, 11, 0.5), leaf, 1e-4);
    HandEndValues enumerated = HandEndValues.of(ends, round, () -> FinalRoundWorkspace.of(round));
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    double[] phi = new double[variant.seats()];
    for (int h = 0; h < ends.handEnds(); h++) {
      GrpState state = ends.state(h);
      round.phi(state, RankUtility.TENHOU, workspace, phi);
      Assert.assertEquals(
          Double.doubleToRawLongBits(enumerated.terminalMass(h)), Double.doubleToRawLongBits(workspace.terminalMass()),
          state.toString());
      for (int seat = 0; seat < phi.length; seat++) {
        long bits = Double.doubleToRawLongBits(enumerated.phi(h, seat));
        Assert.assertEquals(bits, Double.doubleToRawLongBits(phi[seat]), state.toString());
        if (!FinalRound.covers(state)) {
          Assert.assertEquals(bits, Double.doubleToRawLongBits(values.phi(h, seat)), state.toString());
        }
      }
    }
  }
}
