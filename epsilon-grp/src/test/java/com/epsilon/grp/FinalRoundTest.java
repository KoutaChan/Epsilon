package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 最後の局の数え上げを確かめる。確かめるのは、順列の確率の和が 1 になること、終局の葉の値が厳密なこと、半荘が続く葉では葉の
 * ネットワークの値を元の卓の相対席に直して足すこと、Σ Φ = Σ u、数え上げない卓では葉のネットワークの値になること、値がメモと作業領域に
 * 依らないこと。
 */
public class FinalRoundTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final int NONE = HandOutcomes.NONE;
  private static final OutcomeShape SHAPE = new OutcomeShape(FOUR, GrpFixtures.plainFields(), 8);
  private static final GrpShape LEAF = GrpFixtures.shape(FOUR, 8, false);

  /**
   * 打ち切る組が無ければ、足した確率だけで和が 1 になる。打ち切った確率を葉のネットワークで埋めても、和はやはり 1。半荘が終わる組の
   * 確率の和は、半荘が続く組がある卓では 1 未満で、打ち切った確率と足しても 1 を超えない。数え上げない卓では 0 に戻る。
   */
  @Test
  public void theEnumerationSumsToOne() {
    FinalRound full = FinalRound.of(GrpFixtures.outcomes(SHAPE, 11, 0.5), GrpFixtures.network(LEAF, 13, 0.5), 0);
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(full);
    GrpState south4 = new GrpState(FOUR, 7, 1, 1, new int[] {28_000, 26_000, 24_000, 21_000});
    double[] p = new double[24];
    Assert.assertEquals(full.permutations(south4, workspace, p), 0.0);
    assertDistribution(p);
    Assert.assertTrue(workspace.networkLeaves() > 0, "the dealer's wins continue the game");
    Assert.assertTrue(workspace.terminalLeaves() > 0, "children's wins end it");
    double terminal = workspace.terminalMass();
    Assert.assertTrue(terminal > 0 && terminal < 1, "terminal mass " + terminal);

    FinalRound pruned = FinalRound.of(full.outcomes(), full.leaf(), 1e-4);
    FinalRoundWorkspace small = FinalRoundWorkspace.of(pruned);
    double dropped = pruned.permutations(south4, small, p);
    Assert.assertTrue(dropped > 0 && dropped < 1, "dropped " + dropped);
    assertDistribution(p);
    Assert.assertTrue(small.settled() < workspace.settled());
    Assert.assertTrue(small.terminalMass() <= terminal + 1e-12 && small.terminalMass() + dropped <= 1 + 1e-12);

    full.permutations(new GrpState(FOUR, 2, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000}), workspace, p);
    Assert.assertEquals(workspace.terminalMass(), 0.0);
  }

  /** 数え上げ（精算、局の進行、葉のネットワークとメモ、打ち切った確率の埋め合わせ）も、ウォームアップの後はメモリを確保しない。 */
  @Test
  public void steadyEnumerationsDoNotAllocate() {
    FinalRound round = FinalRound.of(GrpFixtures.outcomes(SHAPE, 11, 0.5), GrpFixtures.network(LEAF, 13, 0.5), 1e-4);
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    Random random = new Random(23);
    List<GrpState> covered = new ArrayList<>();
    while (covered.size() < 100) {
      GrpState state = GrpFixtures.state(FOUR, random);
      if (FinalRound.covers(state)) {
        covered.add(state);
      }
    }
    GrpState[] states = covered.toArray(GrpState[]::new);
    double[] p = new double[24];
    Allocations.assertSteady(states.length + " enumerations", 10, () -> {
      for (GrpState state : states) {
        round.permutations(state, workspace, p);
      }
    });
    Assert.assertTrue(workspace.settled() > 0 && workspace.networkLeaves() > 0, workspace.settled() + " settled");
  }

  /**
   * 西 4 で、q を相対席 2（絶対席 1）の満貫のツモ・立直なしに寄せる。親（絶対席 3）が 4,000 点、子が 2,000 点ずつ払い、最終の点
   * 24,000・33,000・22,000・21,000 で終わる。そのためその組の葉は厳密で、p はその順列に、Φ はその順位の効用に一致する。ほかの組は
   * 打ち切る。打ち切った確率は 1e-12 未満で、半荘が終わる組の確率の和は 1 から打ち切った確率を引いたものになる。打ち切りのしきい値が
   * 0 なら、親が続く組（親の和了・親の聴牌・途中流局）は西 4 でも続くので、葉のネットワークも呼ぶ。
   */
  @Test
  public void terminalLeavesAreExact() {
    HandOutcomes hands = HandOutcomes.of(FOUR);
    double[] logits = quiet(hands);
    logits[hands.mainTerm(GrpFixtures.main(hands, OutcomeSection.TSUMO, 2, NONE, NONE, 0))] = 40;
    logits[hands.gradeTerm(2, WinMethod.TSUMO, BaseClass.MANGAN.ordinal())] = 40;
    GrpState west4 = new GrpState(FOUR, 11, 0, 0, new int[] {26_000, 25_000, 24_000, 25_000});
    double[] p = new double[24];
    FinalRound every = FinalRound.of(GrpFixtures.biased(SHAPE, logits), GrpFixtures.network(LEAF, 17, 0.5), 0);
    FinalRoundWorkspace everyWorkspace = FinalRoundWorkspace.of(every);
    every.permutations(west4, everyWorkspace, p);
    Assert.assertTrue(everyWorkspace.networkLeaves() > 0);

    FinalRound round = FinalRound.of(GrpFixtures.biased(SHAPE, logits), GrpFixtures.network(LEAF, 17, 0.5), 1e-9);
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    double dropped = round.permutations(west4, workspace, p);
    Assert.assertTrue(dropped < 1e-12);
    Assert.assertEquals(workspace.terminalMass(), 1 - dropped, 1e-12);
    int expected = RankPermutations.of(FOUR).index(new int[] {3, 1, 0, 2});
    Assert.assertEquals(p[expected], 1, 1e-12);
    Assert.assertEquals(workspace.networkLeaves(), 0);
    Assert.assertEquals(workspace.terminalLeaves(), workspace.settled());
    double[] phi = new double[4];
    round.phi(west4, RankUtility.TENHOU, workspace, phi);
    double[] utilities = {1.0 / 3, 2.0 / 3, 0, -1};
    for (int seat = 0; seat < 4; seat++) {
      Assert.assertEquals(phi[seat], utilities[seat], 1e-12, "seat " + seat);
    }
  }

  /**
   * 南 4（親は絶対席 3）でトップが返し点に届かない卓。q を相対席 1（絶対席 0）の、相対席 2（絶対席 1）からの 1 翻 30 符のロンに
   * 寄せると、半荘は西 1（親は絶対席 0）へ続き、p は葉のネットワークが西 1 の卓に出す p(π|s′) を 1 席ずらしたものになる。半荘が
   * 終わる組の確率の和はほぼ 0。
   */
  @Test
  public void continuingLeavesUseTheLeafNetworkInTheStartFrame() {
    HandOutcomes hands = HandOutcomes.of(FOUR);
    double[] logits = quiet(hands);
    logits[hands.mainTerm(GrpFixtures.main(hands, OutcomeSection.RON, 1, NONE, 2, 0))] = 40;
    logits[hands.gradeTerm(1, WinMethod.RON, BaseClass.B240.ordinal())] = 40;
    GrpNetwork leaf = GrpFixtures.network(LEAF, 19, 0.7);
    FinalRound round = FinalRound.of(GrpFixtures.biased(SHAPE, logits), leaf, 0);
    double[] p = new double[24];
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    round.permutations(new GrpState(FOUR, 7, 0, 0, new int[] {28_000, 26_000, 24_000, 22_000}), workspace, p);
    Assert.assertTrue(workspace.terminalMass() < 1e-12, "terminal mass " + workspace.terminalMass());
    double[] west1 = new double[24];
    leaf.permutations(
        new GrpState(FOUR, 8, 0, 0, new int[] {29_000, 25_000, 24_000, 22_000}), GrpWorkspace.of(LEAF), west1);
    RankPermutations table = RankPermutations.of(FOUR);
    for (int permutation = 0; permutation < 24; permutation++) {
      Assert.assertEquals(p[table.rotate(permutation, 1)], west1[permutation], 1e-12, "permutation " + permutation);
    }
  }

  /**
   * 局の始めの点が 1,000 点未満の席は立直できないので、q がその席の立直に確率を置いても、立直していないものとして数える。南 4
   * （親は絶対席 3）で絶対席 0（相対席 1）が 500 点の卓で、q を親の 500 点オールのツモに寄せ、相対席 1 が支払う側で立直した確率を
   * ほぼ 1 にする。絶対席 0 は 0 点で飛ばず、親はトップでないので、半荘は南 4 の 1 本場へ続く。p は葉のネットワークがその卓に出す
   * p(π|s′) になり、半荘が終わる組の確率の和はほぼ 0。
   */
  @Test
  public void seatsBelowTheDepositDeclareNoRiichi() {
    HandOutcomes hands = HandOutcomes.of(FOUR);
    double[] logits = quiet(hands);
    logits[hands.mainTerm(GrpFixtures.main(hands, OutcomeSection.TSUMO, 0, NONE, NONE, 0))] = 40;
    logits[hands.gradeTerm(0, WinMethod.TSUMO, BaseClass.B240.ordinal())] = 40;
    logits[hands.riichiTerm(1, RiichiRole.PAYER, true)] = 40;
    GrpNetwork leaf = GrpFixtures.network(LEAF, 19, 0.7);
    FinalRound round = FinalRound.of(GrpFixtures.biased(SHAPE, logits), leaf, 0);
    double[] p = new double[24];
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    round.permutations(new GrpState(FOUR, 7, 0, 0, new int[] {500, 44_500, 30_000, 25_000}), workspace, p);
    Assert.assertTrue(workspace.terminalMass() < 1e-12, "terminal mass " + workspace.terminalMass());
    double[] renchan = new double[24];
    leaf.permutations(
        new GrpState(FOUR, 7, 1, 0, new int[] {0, 44_000, 29_500, 26_500}), GrpWorkspace.of(LEAF), renchan);
    for (int permutation = 0; permutation < 24; permutation++) {
      Assert.assertEquals(p[permutation], renchan[permutation], 1e-12, "permutation " + permutation);
    }
  }

  @Test
  public void phiSumsToTheUtilitiesAndStaysInRange() {
    FinalRound round = FinalRound.of(GrpFixtures.outcomes(SHAPE, 19, 0.5), GrpFixtures.network(LEAF, 23, 0.5), 1e-5);
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    Random random = new Random(29);
    double[] phi = new double[4];
    for (int trial = 0; trial < 40; trial++) {
      GrpState state = covered(random);
      round.phi(state, RankUtility.TENHOU, workspace, phi);
      double sum = 0;
      for (double value : phi) {
        Assert.assertTrue(value >= -1 - 1e-12 && value <= 2.0 / 3 + 1e-12, state + " " + value);
        sum += value;
      }
      Assert.assertEquals(sum, 0, 1e-12, state.toString());
    }
  }

  @Test
  public void uncoveredTablesUseTheLeafNetwork() {
    GrpNetwork leaf = GrpFixtures.network(LEAF, 31, 0.5);
    FinalRound round = FinalRound.of(GrpFixtures.outcomes(SHAPE, 37, 0.5), leaf, 1e-6);
    GrpState east = new GrpState(FOUR, 2, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    Assert.assertFalse(FinalRound.covers(east));
    Assert.assertTrue(FinalRound.covers(new GrpState(FOUR, 7, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000})));
    Assert.assertFalse(FinalRound.covers(new GrpState(Variant.TENHOU_3P, 4, 0, 0, new int[] {35_000, 35_000, 35_000})));
    Assert.assertTrue(FinalRound.covers(new GrpState(Variant.TENHOU_3P, 5, 0, 0, new int[] {35_000, 35_000, 35_000})));
    double[] p = new double[24];
    double[] direct = new double[24];
    Assert.assertEquals(round.permutations(east, FinalRoundWorkspace.of(round), p), 0.0);
    leaf.permutations(east, GrpWorkspace.of(LEAF), direct);
    Assert.assertEquals(p, direct);
    FinalRound other = FinalRound.of(round.outcomes(), leaf, 1e-6);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> other.permutations(east, FinalRoundWorkspace.of(round), p));
    Assert.assertThrows(IllegalArgumentException.class, () -> FinalRound.of(round.outcomes(), leaf, 1));
  }

  /** 葉の値はメモに当たっても外れてもビット単位で同じなので、作業領域と呼び出しの順に依らない。 */
  @Test
  public void valuesDoNotDependOnTheMemoOrTheWorkspace() {
    FinalRound round = FinalRound.of(GrpFixtures.outcomes(SHAPE, 41, 0.5), GrpFixtures.network(LEAF, 43, 0.5), 1e-6);
    Random random = new Random(47);
    List<GrpState> states = new ArrayList<>();
    for (int trial = 0; trial < 20; trial++) {
      states.add(covered(random));
    }
    FinalRoundWorkspace first = FinalRoundWorkspace.of(round);
    double[][] expected = new double[states.size()][24];
    for (int index = 0; index < states.size(); index++) {
      round.permutations(states.get(index), first, expected[index]);
    }
    long hits = first.memoHits();
    FinalRoundWorkspace second = FinalRoundWorkspace.of(round);
    double[] p = new double[24];
    for (int pass = 0; pass < 2; pass++) {
      for (int index = states.size() - 1; index >= 0; index--) {
        round.permutations(states.get(index), pass == 0 ? second : first, p);
        for (int k = 0; k < 24; k++) {
          Assert.assertEquals(Double.doubleToRawLongBits(p[k]), Double.doubleToRawLongBits(expected[index][k]));
        }
      }
    }
    Assert.assertTrue(first.memoHits() > hits, "the second pass reads the memo");
  }

  /** 立直の確率を 0 に寄せたロジット（ほかは 0）。 */
  private static double[] quiet(HandOutcomes hands) {
    double[] logits = new double[hands.logits()];
    for (int seat = 0; seat < 4; seat++) {
      for (RiichiRole role : RiichiRole.values()) {
        logits[hands.riichiTerm(seat, role, true)] = -40;
      }
    }
    return logits;
  }

  /** 南 4 から西 4 までの、半荘が続く卓。 */
  private static GrpState covered(Random random) {
    GrpState state = GrpFixtures.state(FOUR, random);
    return new GrpState(FOUR, 7 + random.nextInt(5), state.honba(), state.sticks(), state.scores());
  }

  private static void assertDistribution(double[] p) {
    double sum = 0;
    for (double value : p) {
      Assert.assertTrue(value >= 0, "negative probability " + value);
      sum += value;
    }
    Assert.assertEquals(sum, 1, 1e-12);
  }
}
