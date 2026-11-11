package com.epsilon.prism.collect;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ObservedDecisions;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.policy.RegimeRepresentatives;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 候補集合 C(o)（{@link CandidateSet#build}）が、O、π_t の s、p0、ハッシュから作る乱数だけで決まることのテスト。優先の順、ε_C で足す
 * 1 手、打った手と同値な手の置き換えも確かめる。
 */
public class CandidateSetTest {
  private static final int M1 = 0, M2 = 1, M4 = 3, M5 = 4, M6 = 5, M7 = 6;

  private static int dahai(int kind) {
    return Action.discard(kind, false, false, false);
  }

  /** 全ての合法手から作る候補集合は打った手を含み、どの合法手もどれかの候補と同値になる。測る手は候補そのもの。 */
  @Test
  public void allActionsCoversEveryLegalAction() {
    int[] legal = {
      dahai(M5), Action.discard(M5, false, true, false), dahai(M1), Action.discard(M1, false, false, true), dahai(M2)
    };
    for (int played = 0; played < legal.length; played++) {
      CandidateSet all = CandidateSet.allActions(legal, played);
      int playedAction = played;
      Assert.assertTrue(Arrays.stream(all.support()).anyMatch(a -> a == playedAction));
      Assert.assertEquals(all.measured(), all.support());
      for (int a = 0; a < legal.length; a++) {
        int action = a;
        Assert.assertTrue(
            Arrays.stream(all.support()).anyMatch(c -> Twins.same(legal, c, action)), "action " + a);
      }
      Assert.assertFalse(all.explored());
    }
  }

  /**
   * 新しいエンジンで打った判断の O で確かめる。同じ入力なら O を読み直しても C は同じで、C は s の 1 位と、最初の体制の両側の代表を含む。
   * 一様に混ぜた分で C の外の手を打っても C は変わらず、測る手に打った手が加わる。
   */
  @Test
  public void dependsOnlyOnTheObservationScoresPriorAndCoins() {
    SeatFacts facts = new SeatFacts(Variant.TENHOU_4P);
    Random random = new Random(3);
    int threatened = 0, outside = 0;
    for (ObservedDecisions.Decision decision : ObservedDecisions.collect(29, 2, TableWalk.Policy.GREEDY, 5_000)) {
      Observation o = decision.observation();
      int[] legal = decision.legal();
      int n = legal.length;
      float[] scores = new float[n], prior = new float[n];
      for (int a = 0; a < n; a++) {
        scores[a] = random.nextFloat();
        prior[a] = random.nextFloat();
      }
      facts.load(o);
      int threats = Threats.count(o, facts);
      if (threats > 0) threatened++;
      CandidateSet built = CandidateSet.build(o, legal, scores, prior, threats, 0, 6, 0.1, 0.9, 0.5);
      CandidateSet again =
          CandidateSet.build(decision.observation(), legal, scores, prior, threats, 0, 6, 0.1, 0.9, 0.5);
      Assert.assertEquals(again.support(), built.support());
      Assert.assertEquals(again.measured(), built.measured());
      Assert.assertFalse(built.explored());
      int top = Greedy.argmax(scores, prior, n);
      Assert.assertTrue(covered(legal, built.support(), top), "the best s");
      int[] regimes = RegimeRepresentatives.of(o, legal, scores, prior, threats);
      for (int r = 0; r < Math.min(2, regimes.length); r++)
        Assert.assertTrue(covered(legal, built.support(), regimes[r]), "the first regime");
      for (int played = 0; played < n; played++) {
        if (covered(legal, built.support(), played)) continue;
        CandidateSet off = CandidateSet.build(o, legal, scores, prior, threats, played, 6, 0.1, 0.9, 0.5);
        // 同値な手のどれが代表になるかは打った手で替わる（0 を打てば、0 と同値な手の代表は 0）ので、同値な手は同じと見て比べる
        Assert.assertEquals(off.support().length, built.support().length);
        for (int c : off.support()) Assert.assertTrue(covered(legal, built.support(), c), "candidate " + c);
        Assert.assertFalse(off.playedInside(played));
        int[] measured = Arrays.copyOf(off.support(), off.support().length + 1);
        measured[measured.length - 1] = played;
        Arrays.sort(measured);
        Assert.assertEquals(off.measured(), measured);
        outside++;
        break;
      }
    }
    Assert.assertTrue(threatened > 100, "threatened " + threatened);
    Assert.assertTrue(outside > 100, "outside " + outside);
  }

  /** ε_C の乱数が当たれば上限を 1 つ空け、ここまでの手と同値でない手のうち floor(draw × 数) 番目の手を 1 つ足す。 */
  @Test
  public void exploreCoinAddsOneUniformAction() {
    int[] legal = {dahai(M1), dahai(M2), dahai(M4), dahai(M5), dahai(M6), dahai(M7)};
    float[] scores = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f};
    float[] prior = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.5f};
    CandidateSet plain = CandidateSet.build(null, legal, scores, prior, 0, 2, 3, 0.1, 0.9, 0.6);
    Assert.assertEquals(plain.support(), new int[] {0, 1, 5});
    Assert.assertFalse(plain.explored());
    // 上限 2 で {0, 5}。残りの {1, 2, 3, 4} から floor(0.6 × 4) = 2 番目の 3 を足す
    CandidateSet explored = CandidateSet.build(null, legal, scores, prior, 0, 2, 3, 0.1, 0.05, 0.6);
    Assert.assertEquals(explored.support(), new int[] {0, 3, 5});
    Assert.assertEquals(explored.measured(), new int[] {0, 2, 3, 5});
    Assert.assertTrue(explored.explored());
    Assert.assertFalse(explored.playedInside(2));
  }

  /** 同じ牌の手出しとツモ切りは 1 つの候補にまとめる。打った手がそのどちらかなら、候補を打った手に置き換える（分岐が要らない）。 */
  @Test
  public void twinOfThePlayedActionReplacesTheCandidate() {
    int[] legal = {dahai(M5), Action.discard(M5, false, true, false), dahai(M1)};
    float[] scores = {0.30f, 0.29f, 0.00f};
    float[] prior = {0.4f, 0.4f, 0.2f};
    CandidateSet tedashi = CandidateSet.build(null, legal, scores, prior, 0, 0, 6, 0.1, 0.9, 0);
    Assert.assertEquals(tedashi.support(), new int[] {0, 2});
    CandidateSet tsumogiri = CandidateSet.build(null, legal, scores, prior, 0, 1, 6, 0.1, 0.9, 0);
    Assert.assertEquals(tsumogiri.support(), new int[] {1, 2});
    Assert.assertEquals(tsumogiri.measured(), new int[] {1, 2});
    Assert.assertTrue(tsumogiri.playedInside(1));
  }

  /**
   * 順位は値の降順。同点なら同点を分ける値（s の順位なら p0）が大きい手を、それも同じなら番号の小さい手を上にする（貪欲な方策の argmax と
   * 同じ比べ方）。初期化したままのネットワーク（s ≡ 0）では、s の順位は p0 の順になる。
   */
  @Test
  public void rankingBreaksTiesByPriorThenIndex() {
    Assert.assertEquals(
        CandidateSet.ranking(new int[] {0, 1, 2, 3}, new float[] {0.2f, 0.5f, 0.5f, -1f}),
        new int[] {1, 2, 0, 3});
    // 返すのは、渡した手の配列の中のインデックス
    Assert.assertEquals(
        CandidateSet.ranking(new int[] {4, 1}, new float[] {0f, 0.3f, 0f, 0f, 0.7f}), new int[] {0, 1});
    Assert.assertEquals(
        CandidateSet.ranking(new int[] {0, 1, 2, 3}, new float[4], new float[] {0.1f, 0.4f, 0.1f, 0.4f}),
        new int[] {1, 3, 0, 2});
    Assert.assertEquals(Greedy.argmax(new float[4], new float[] {0.1f, 0.2f, 0.4f, 0.3f}, 4), 2);
  }

  private static boolean covered(int[] legal, int[] support, int action) {
    for (int c : support) if (Twins.same(legal, c, action)) return true;
    return false;
  }
}
