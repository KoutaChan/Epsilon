package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの最後の局の数え上げを確かめる。確かめるのは、結果の表の大きさ、覆う卓（南 3 から）、終局しかない卓での厳密な値、打ち切りの
 * 無い数え上げが全ての組を別に足し上げた値と合うこと（これは四人打ちでも確かめる）。
 */
public class ThreePlayerFinalRoundTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final int NONE = HandOutcomes.NONE;

  /**
   * 終わり方はツモ 3・ロン 6・ダブロン 3・荒牌 8・途中流局 1 の 21 通り。打点の項は 3 席 × ツモ・ロン × 26 段の 156、立直の項は
   * 席 × 立場の数。1 つの結果が持つ項は最大 6（ダブロンの終わり方・打点 2・立直 3）で、順列は 6 通り。
   */
  @Test
  public void threePlayerTablesHaveTheDeclaredSizes() {
    HandOutcomes hands = HandOutcomes.of(THREE);
    Assert.assertEquals(hands.mainCount(), 21);
    int[] counts = new int[OutcomeSection.values().length];
    for (int main = 0; main < hands.mainCount(); main++) {
      counts[hands.section(main).ordinal()]++;
      int expected = switch (hands.section(main)) {
        case TSUMO, RON, DOUBLE_RON, ABORTIVE -> 3;
        case DRAW -> Integer.bitCount(hands.tenpai(main));
      };
      Assert.assertEquals(hands.factorCount(main), expected, "main " + main);
    }
    Assert.assertEquals(counts, new int[] {3, 6, 3, 8, 1});
    int roles = RiichiRole.values().length;
    Assert.assertEquals(hands.termStart(OutcomeTerm.GRADE), 21);
    Assert.assertEquals(hands.termStart(OutcomeTerm.RIICHI), 21 + 156);
    Assert.assertEquals(hands.logits(), 21 + 156 + 3 * roles);
    Assert.assertEquals(hands.terms(), 21 + 156 + 2 * 3 * roles);
    Assert.assertEquals(hands.maxTerms(), 6);
    Assert.assertEquals(RankPermutations.of(THREE).count(), 6);
    Assert.assertEquals(FinalRound.VARIANTS, EnumSet.of(Variant.TENHOU_4P, THREE));
    OutcomeShape shape = new OutcomeShape(THREE, GrpFixtures.plainFields(), 8);
    Assert.assertSame(shape.outcomes(), hands);
    int[] scores = {35_000, 35_000, 35_000};
    Assert.assertFalse(FinalRound.covers(new GrpState(THREE, 4, 0, 0, scores)));
    for (int round = 5; round <= THREE.finalRound(); round++) {
      Assert.assertTrue(FinalRound.covers(new GrpState(THREE, round, 0, 0, scores)), "round " + round);
    }
  }

  /**
   * 西 3（親は絶対席 2）で、q を相対席 1（絶対席 0）の満貫のツモ・立直なしに寄せる。ツモ損で親が 4,000、子が 2,000 を払い、最終の点
   * 46,000・33,000・26,000 で終わるので、p はその順列に、Φ はその順位の効用に一致し、葉のネットワークは呼ばない。
   */
  @Test
  public void terminalOnlyTablesAreExact() {
    HandOutcomes hands = HandOutcomes.of(THREE);
    double[] logits = new double[hands.logits()];
    for (int seat = 0; seat < 3; seat++) {
      for (RiichiRole role : RiichiRole.values()) {
        logits[hands.riichiTerm(seat, role, true)] = -40;
      }
    }
    logits[hands.mainTerm(GrpFixtures.main(hands, OutcomeSection.TSUMO, 1, NONE, NONE, 0))] = 40;
    logits[hands.gradeTerm(1, WinMethod.TSUMO, BaseClass.MANGAN.ordinal())] = 40;
    OutcomeShape shape = new OutcomeShape(THREE, GrpFixtures.plainFields(), 8);
    GrpNetwork leaf = GrpFixtures.network(GrpFixtures.shape(THREE, 8, false), 17, 0.5);
    FinalRound round = FinalRound.of(GrpFixtures.biased(shape, logits), leaf, 1e-9);
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    GrpState west3 = new GrpState(THREE, 8, 0, 0, new int[] {40_000, 35_000, 30_000});
    Assert.assertEquals(west3.dealer(), 2);
    double[] p = new double[6];
    double dropped = round.permutations(west3, workspace, p);
    Assert.assertTrue(dropped < 1e-12, "dropped " + dropped);
    Assert.assertEquals(workspace.terminalMass(), 1 - dropped, 1e-12);
    Assert.assertEquals(workspace.networkLeaves(), 0);
    Assert.assertEquals(workspace.terminalLeaves(), workspace.settled());
    Assert.assertEquals(p[RankPermutations.of(THREE).index(new int[] {2, 0, 1})], 1, 1e-12);
    double[] phi = new double[3];
    round.phi(west3, RankUtility.TENHOU, workspace, phi);
    double[] utilities = {2, 0, -2};
    for (int seat = 0; seat < 3; seat++) {
      Assert.assertEquals(phi[seat], utilities[seat], 1e-12, "seat " + seat);
    }
  }

  /**
   * 打ち切りの値 0 の数え上げは、全ての組（終わり方 × 打点 × 立直）を q の確率で重み付けし、精算と局の進行の後の卓の順列の分布
   * （終局は実際の順列に 1、続けば葉のネットワークを元の卓の相対席に直したもの）を別に足し上げた値と合う。
   */
  @Test
  public void anUncutEnumerationMatchesABruteForceSum() {
    for (Variant variant : List.of(THREE, Variant.TENHOU_4P)) {
      OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
      FinalRound round = FinalRound.of(
          GrpFixtures.outcomes(shape, 51, 0.5), GrpFixtures.network(GrpFixtures.shape(variant, 8, false), 53, 0.5), 0);
      FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
      Random random = new Random(55);
      int tables = variant == THREE ? 12 : 2;
      int count = RankPermutations.of(variant).count();
      double[] p = new double[count];
      for (int table = 0; table < tables; table++) {
        GrpState state = GrpFixtures.state(variant, random);
        int last = variant.lastRound();
        state = new GrpState(
            variant, last + random.nextInt(variant.finalRound() - last + 1), state.honba(), state.sticks(),
            state.scores());
        Assert.assertEquals(round.permutations(state, workspace, p), 0.0);
        double[] expected = bruteForce(round, state);
        double sum = 0;
        for (int k = 0; k < count; k++) {
          Assert.assertEquals(p[k], expected[k], 1e-12, state + " permutation " + k);
          sum += expected[k];
        }
        Assert.assertEquals(sum, 1, 1e-12, state.toString());
      }
    }
  }

  /**
   * 全ての組を足し上げた p(π|s)。FinalRound の打ち切りと葉のメモを使わない。局の始めの点が 1,000 点未満の席は立直できないので、
   * その席が立直した組の重みは 0 にし、立直していない組にはその席の立直の確率を掛けない。
   */
  private static double[] bruteForce(FinalRound round, GrpState state) {
    Variant variant = state.variant();
    int seats = variant.seats();
    HandOutcomes hands = HandOutcomes.of(variant);
    RankPermutations table = RankPermutations.of(variant);
    int count = table.count();
    OutcomeWorkspace outcomes = OutcomeWorkspace.of(round.outcomes().shape());
    TableState start = new TableState(variant);
    state.copyInto(start);
    round.outcomes().forward(start, outcomes);
    GrpWorkspace leafWorkspace = GrpWorkspace.of(round.leaf().shape());
    Map<Long, double[]> leaves = new HashMap<>();
    RoundOutcome outcome = new RoundOutcome(variant);
    TableState after = new TableState(variant);
    int[] delta = new int[seats];
    int[] scores = new int[seats];
    double[] out = new double[count];
    int dealer = start.dealer();
    for (int main = 0; main < hands.mainCount(); main++) {
      OutcomeSection section = hands.section(main);
      boolean win = section == OutcomeSection.TSUMO || section == OutcomeSection.RON
          || section == OutcomeSection.DOUBLE_RON;
      WinMethod method = section == OutcomeSection.TSUMO ? WinMethod.TSUMO : WinMethod.RON;
      int bases = win ? HandOutcomes.BASES : 1;
      int others = section == OutcomeSection.DOUBLE_RON ? HandOutcomes.BASES : 1;
      int factors = hands.factorCount(main);
      for (int base = 0; base < bases; base++) {
        for (int other = 0; other < others; other++) {
          for (int mask = 0; mask < 1 << factors; mask++) {
            double q = outcomes.probability(hands.mainTerm(main));
            if (win) {
              q *= outcomes.probability(hands.gradeTerm(hands.winner(main), method, base));
            }
            if (others > 1) {
              q *= outcomes.probability(hands.gradeTerm(hands.secondWinner(main), WinMethod.RON, other));
            }
            int riichi = 0;
            for (int index = 0; index < factors; index++) {
              int seat = hands.factorSeat(main, index);
              int absolute = Seats.absolute(dealer, seat, seats);
              boolean declared = (mask & 1 << index) != 0;
              if (start.score(absolute) < Variant.DEPOSIT) {
                q *= declared ? 0 : 1;
              } else {
                q *= outcomes.probability(hands.riichiTerm(seat, hands.role(main, seat), declared));
              }
              if (declared) {
                riichi |= 1 << absolute;
              }
            }
            hands.place(main, win ? base : NONE, others > 1 ? other : NONE, riichi, dealer, outcome);
            Settlement.apply(variant, start, outcome, delta);
            Progression.next(start, outcome, delta, after);
            if (after.gameOver()) {
              for (int seat = 0; seat < seats; seat++) {
                scores[seat] = after.score(seat);
              }
              out[table.realized(scores, dealer)] += q;
              continue;
            }
            GrpState next = GrpState.of(after);
            double[] leaf = leaves.get(next.key());
            if (leaf == null || next.key() == GrpKey.NONE) {
              leaf = new double[count];
              round.leaf().permutations(next, leafWorkspace, leaf);
              leaves.put(next.key(), leaf);
            }
            int shift = Math.floorMod(after.dealer() - dealer, seats);
            for (int k = 0; k < count; k++) {
              out[table.rotate(k, shift)] += q * leaf[k];
            }
          }
        }
      }
    }
    return out;
  }
}
