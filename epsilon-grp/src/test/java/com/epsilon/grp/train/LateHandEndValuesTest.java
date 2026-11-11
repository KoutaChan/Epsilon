package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.FinalRoundWorkspace;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.Phi;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 終盤の数え上げの Φ から局末の値を作るときの振り分け（HandEndValues.of）を確かめる。数え上げない局末の Φ・周辺確率・実際の順列の
 * 対数確率は、葉のネットワークで求めた値とビット単位で同じになる。数え上げる局末では、数え上げで求めた値とビット単位で同じになる。
 * 埋めた確率、半荘が終わる組の確率、精算した組の数、半荘が続く葉の数は、数え上げる局末にだけ入る。Φ は収集で使う Phi.values と
 * ビット単位で同じ。数え上げない局末で実際の順列の確率が 0 に潰れても、対数確率はロジットから求め直すので有限になる。
 */
public class LateHandEndValuesTest {

  private static final RankUtility UTILITY = RankUtility.TENHOU;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void coveredHandEndsTakeTheEnumerationAndOthersTheNetwork(Variant variant) {
    HandEnds ends = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 300, 0, 41));
    FinalRound round = round(variant, 0.5);
    Phi late = Phi.late(round, UTILITY, 256);
    HandEndValues values = HandEndValues.of(ends, late);
    HandEndValues network = HandEndValues.of(ends, round.leaf(), GrpWorkspace.of(round.leaf().shape()));
    HandEndValues enumerated = HandEndValues.of(ends, round, () -> FinalRoundWorkspace.of(round));
    Assert.assertTrue(values.enumerated());
    Assert.assertFalse(network.enumerated());
    Assert.assertTrue(values.joint());
    Assert.assertTrue(values.finite());
    int seats = variant.seats();
    double[] phi = new double[seats];
    int covered = 0;
    double filled = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      GrpState state = ends.state(h);
      HandEndValues expected = FinalRound.covers(state) ? enumerated : network;
      assertSame(values, expected, h);
      if (FinalRound.covers(state)) {
        covered++;
        filled += values.filledMass(h);
        assertBits(values.filledMass(h), enumerated.filledMass(h), state);
        assertBits(values.terminalMass(h), enumerated.terminalMass(h), state);
        Assert.assertTrue(values.terminalMass(h) + values.filledMass(h) <= 1 + 1e-12, state.toString());
      } else {
        Assert.assertEquals(values.filledMass(h), 0.0, state.toString());
        Assert.assertEquals(values.terminalMass(h), 0.0, state.toString());
      }
      late.values(state, phi);
      for (int seat = 0; seat < seats; seat++) {
        assertBits(values.phi(h, seat), phi[seat], state);
      }
    }
    Assert.assertTrue(covered > 0 && covered < ends.handEnds(), covered + " of " + ends.handEnds());
    Assert.assertTrue(filled > 0, "no filled mass in " + covered + " covered hand ends");

    HandEndValues again = HandEndValues.of(ends, late);
    for (int h = 0; h < ends.handEnds(); h++) {
      assertSame(again, values, h);
    }
    HandEnds selected = ends.select(end -> end % 3 == 1);
    HandEndValues chosen = values.select(selected);
    HandEndValues direct = HandEndValues.of(selected, late);
    Assert.assertTrue(chosen.enumerated());
    for (int index = 0; index < selected.handEnds(); index++) {
      assertSame(chosen, direct, index);
      assertBits(chosen.phi(index, 0), values.phi(3 * index + 1, 0), selected.state(index));
      assertBits(chosen.filledMass(index), values.filledMass(3 * index + 1), selected.state(index));
      assertBits(chosen.terminalMass(index), values.terminalMass(3 * index + 1), selected.state(index));
    }
  }

  /**
   * 葉の重みを大きくして、数え上げない局末で実際の順列の確率を double の 0 に潰す。そこでも LATE の対数確率は有限で、葉の
   * ネットワークでロジットから求め直した値とビット単位で同じになる。
   */
  @Test(dataProvider = "variants")
  public void collapsedProbabilitiesKeepFiniteLogs(Variant variant) {
    HandEnds ends = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 200, 0, 43));
    FinalRound round = round(variant, 8);
    GrpNetwork leaf = round.leaf();
    HandEndValues values = HandEndValues.of(ends, Phi.late(round, UTILITY, 256));
    HandEndValues network = HandEndValues.of(ends, leaf, GrpWorkspace.of(leaf.shape()));
    GrpWorkspace workspace = GrpWorkspace.of(leaf.shape());
    double[] p = new double[RankPermutations.of(variant).count()];
    int collapsed = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      GrpState state = ends.state(h);
      if (FinalRound.covers(state)) {
        continue;
      }
      leaf.permutations(state, workspace, p);
      if (p[ends.permutation(h)] == 0) {
        collapsed++;
        Assert.assertTrue(Double.isFinite(values.jointLog(h)), state.toString());
      }
      assertSame(values, network, h);
    }
    Assert.assertTrue(collapsed > 0, "no collapsed hand end among " + ends.handEnds());
  }

  /**
   * 数え上げる局末の、精算した組の数と半荘が続く葉の数は、その局末だけを新しい作業領域で数え上げたときの数と同じになる。半荘が続く
   * 葉の数は、精算した組の数から半荘が終わった葉の数を引いた数。葉のメモに当たったかどうかでは変わらないので、もう 1 度求めても同じ
   * になる。数え上げない局末と、数え上げで求めていない値では 0。選んだ列にもそのまま移る。
   */
  @Test(dataProvider = "variants")
  public void enumerationCountsMatchAFreshWorkspace(Variant variant) {
    HandEnds ends = HandEnds.of(variant, UTILITY, TrainFixtures.simulated(variant, 200, 0, 47));
    FinalRound round = round(variant, 0.5);
    Phi late = Phi.late(round, UTILITY, 256);
    HandEndValues values = HandEndValues.of(ends, late);
    HandEndValues again = HandEndValues.of(ends, late);
    HandEndValues network = HandEndValues.of(ends, round.leaf(), GrpWorkspace.of(round.leaf().shape()));
    double[] p = new double[RankPermutations.of(variant).count()];
    long leaves = 0;
    long terminal = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      GrpState state = ends.state(h);
      Assert.assertEquals(network.settled(h), 0, state.toString());
      Assert.assertEquals(network.leaves(h), 0, state.toString());
      Assert.assertEquals(again.settled(h), values.settled(h), state.toString());
      Assert.assertEquals(again.leaves(h), values.leaves(h), state.toString());
      if (!FinalRound.covers(state)) {
        Assert.assertEquals(values.settled(h), 0, state.toString());
        Assert.assertEquals(values.leaves(h), 0, state.toString());
        continue;
      }
      FinalRoundWorkspace fresh = FinalRoundWorkspace.of(round);
      round.permutations(state, fresh, p);
      Assert.assertEquals(values.settled(h), fresh.settled(), state.toString());
      Assert.assertEquals(values.leaves(h), fresh.networkLeaves() + fresh.memoHits(), state.toString());
      Assert.assertEquals(values.settled(h) - values.leaves(h), fresh.terminalLeaves(), state.toString());
      leaves += values.leaves(h);
      terminal += fresh.terminalLeaves();
    }
    Assert.assertTrue(leaves > 0 && terminal > 0, leaves + " leaves, " + terminal + " terminal leaves");
    HandEnds selected = ends.select(end -> end % 3 == 1);
    HandEndValues chosen = values.select(selected);
    for (int index = 0; index < selected.handEnds(); index++) {
      Assert.assertEquals(chosen.settled(index), values.settled(3 * index + 1));
      Assert.assertEquals(chosen.leaves(index), values.leaves(3 * index + 1));
    }
  }

  static FinalRound round(Variant variant, double leafScale) {
    OutcomeShape outcomes = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    return FinalRound.of(
        GrpFixtures.outcomes(outcomes, 31, 0.5),
        GrpFixtures.network(GrpFixtures.shape(variant, 8, false), 37, leafScale),
        1e-5);
  }

  /** 局末 h の Φ・周辺確率・対数確率が、ビット単位で同じことを確かめる。 */
  private static void assertSame(HandEndValues actual, HandEndValues expected, int h) {
    HandEnds ends = actual.ends();
    GrpState state = ends.state(h);
    int seats = ends.seats();
    assertBits(actual.jointLog(h), expected.jointLog(h), state);
    for (int seat = 0; seat < seats; seat++) {
      assertBits(actual.phi(h, seat), expected.phi(h, seat), state);
      for (int place = 0; place < seats; place++) {
        assertBits(actual.marginal(h, seat, place), expected.marginal(h, seat, place), state);
      }
    }
  }

  private static void assertBits(double actual, double expected, GrpState state) {
    Assert.assertEquals(
        Double.doubleToRawLongBits(actual), Double.doubleToRawLongBits(expected), String.valueOf(state));
  }
}
