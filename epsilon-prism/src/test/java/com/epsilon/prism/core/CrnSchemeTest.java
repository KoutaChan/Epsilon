package com.epsilon.prism.core;

import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.util.SeedMixer;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 判断の共通乱数の使い方（{@link CrnScheme}）: V1 は {@link CrnKeys} の式と同じ値、V2 の 2 段の床は分布がちょうど b で、兄弟の分岐（σ だけ
 * 違う）が同じ鍵で一緒に発火して同じ手を引く。V2 の種別類は鳴いた後の打牌を分ける。
 */
public class CrnSchemeTest {

  private static final List<Action> LEGAL =
      List.of(Action.dahai(0), Action.dahai(9), Action.dahai(18), Action.dahai(27), Action.dahai(31));

  /** V1 は鍵・種別類・引いた手が {@link CrnKeys} の式と同じ。 */
  @Test
  public void firstSchemeKeepsTheCurrentDraws() {
    float[] sigma = {0, 0, 1, 0, 0};
    float[] b = new Behaviour(0.1, 0, CrnScheme.V1).policy(sigma, LEGAL.size());
    for (int k = 0; k < 10_000; k++) {
      long hand = SeedMixer.mix64(k);
      int seat = k & 3, ordinal = k % 37;
      for (EngineDecisionKind kind : EngineDecisionKind.values()) {
        int kindClass = CrnScheme.V1.kindClass(kind);
        Assert.assertEquals(kindClass, CrnKeys.kindClass(kind));
        long key = CrnScheme.V1.decisionKey(hand, seat, kindClass, ordinal);
        Assert.assertEquals(key, CrnKeys.decisionKey(hand, seat, kindClass, ordinal));
        Assert.assertEquals(
            CrnScheme.V1.select(key, LEGAL, sigma, b, 0.1), CrnKeys.select(key, LEGAL, b, 0));
      }
    }
  }

  /** V2 の 2 段の床: 10^5 個の鍵で選んだ頻度が b = (1 − ε)·σ + ε/n に合う（z < 4.5）。σ の和が 0 なら一様。 */
  @Test
  public void twoStageFloorDrawsBehaviour() {
    double epsilon = 0.1;
    int n = LEGAL.size(), keys = 100_000;
    for (float[] sigma : new float[][] {{0, 0, 1, 0, 0}, {0.5f, 0, 0.25f, 0.25f, 0}, {0, 0, 0, 0, 0}}) {
      float[] b = new Behaviour(epsilon, 0, CrnScheme.V2).policy(sigma, n);
      long[] counts = new long[n];
      for (int k = 0; k < keys; k++) {
        counts[CrnScheme.V2.select(SeedMixer.mix64(31L * k + 7), LEGAL, sigma, b, epsilon)]++;
      }
      for (int a = 0; a < n; a++) {
        double expected = keys * (double) b[a];
        double z = (counts[a] - expected) / Math.sqrt(Math.max(expected * (1 - b[a]), 1));
        Assert.assertTrue(Math.abs(z) < 4.5, "action " + a + " count " + counts[a] + " expected " + expected);
      }
    }
  }

  /**
   * 兄弟の分岐: 同じ鍵・同じ合法手で σ（貪欲の手）だけが違うとき、V2 は床の発火が同じ鍵で起き、発火した判断では同じ手を引く。V1 は貪欲の手で
   * 競争の相手が変わるので、床の手が揃わない判断がある。
   */
  @Test
  public void siblingsFireTogether() {
    double epsilon = 0.2;
    int n = LEGAL.size();
    float[] first = {1, 0, 0, 0, 0};
    float[] second = {0, 0, 0, 1, 0};
    float[] firstB = new Behaviour(epsilon, 0, CrnScheme.V2).policy(first, n);
    float[] secondB = new Behaviour(epsilon, 0, CrnScheme.V2).policy(second, n);
    int fired = 0, firstSchemeMismatch = 0;
    for (int k = 0; k < 20_000; k++) {
      long key = SeedMixer.mix64(k + 1_000_003L);
      int a = CrnScheme.V2.select(key, LEGAL, first, firstB, epsilon);
      int b = CrnScheme.V2.select(key, LEGAL, second, secondB, epsilon);
      boolean deviatedA = a != 0, deviatedB = b != 3;
      if (deviatedA || deviatedB || a == b) {
        // 発火した判断（一様の競争の手が貪欲の手と重なると、発火しても貪欲の手になる）
        fired++;
        Assert.assertEquals(a, b, "key " + key);
      }
      int v1a = CrnScheme.V1.select(key, LEGAL, first, firstB, epsilon);
      int v1b = CrnScheme.V1.select(key, LEGAL, second, secondB, epsilon);
      if ((v1a != 0 || v1b != 3) && v1a != v1b) firstSchemeMismatch++;
    }
    Assert.assertTrue(fired > 3_000, "fired " + fired);
    Assert.assertTrue(firstSchemeMismatch > 100, "v1 mismatches " + firstSchemeMismatch);
  }

  /** V2 の種別類は 3 つで、鳴いた後の打牌は手番と別。席×種別類の鍵は同じ序数で重ならない。 */
  @Test
  public void secondSchemeSeparatesPostCallDiscards() {
    Assert.assertEquals(CrnScheme.V2.kindClasses(), 3);
    Assert.assertNotEquals(
        CrnScheme.V2.kindClass(EngineDecisionKind.POST_CALL_DAHAI),
        CrnScheme.V2.kindClass(EngineDecisionKind.TURN));
    Assert.assertEquals(
        CrnScheme.V2.kindClass(EngineDecisionKind.CHANKAN),
        CrnScheme.V2.kindClass(EngineDecisionKind.RESPONSE));
    Set<Long> keys = new HashSet<>();
    for (int seat = 0; seat < 4; seat++) {
      for (int kindClass = 0; kindClass < 3; kindClass++) {
        Assert.assertTrue(keys.add(CrnScheme.V2.decisionKey(123L, seat, kindClass, 5)));
      }
    }
    CrnKeys.Cursor cursor = new CrnKeys.Cursor(CrnScheme.V2.kindClasses());
    Assert.assertEquals(cursor.next(3, 2), 0);
    Assert.assertEquals(cursor.next(3, 2), 1);
    Assert.assertEquals(cursor.next(3, 0), 0);
    Assert.assertEquals(cursor.copy().next(3, 2), 2);
  }
}
