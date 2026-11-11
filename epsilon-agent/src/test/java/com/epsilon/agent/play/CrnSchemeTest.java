package com.epsilon.agent.play;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.HashSet;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 判断の共通乱数の使い方（{@link CrnScheme}）を確かめる。1 回の競争で引く方式（{@code SINGLE_RACE}）は、{@link CrnKeys} の式と
 * 同じ値になる。2 段で引く方式（{@code TWO_STAGE_FLOOR}）は、引いた手の分布がちょうど一様分布を混ぜた分布になり、打ち手の分布だけが違う兄弟の分岐は、同じキーで
 * 一緒に発火して同じ手を引く。2 段で引く方式では、鳴いた後の打牌を手番と別の判断のクラスにする。
 */
public class CrnSchemeTest {

  private static final int[] LEGAL = {discard(0), discard(9), discard(18), discard(27), discard(31)};

  private static int discard(int kind) {
    return Action.discard(kind, false, false, false);
  }

  /** {@code SINGLE_RACE} では、キー・判断のクラス・引いた手が {@link CrnKeys} の式と同じになる。 */
  @Test
  public void firstSchemeKeepsTheCurrentDraws() {
    float[] sigma = {0, 0, 1, 0, 0};
    float[] b = new Behaviour(0.1, 0, CrnScheme.SINGLE_RACE).policy(sigma, LEGAL.length);
    for (int k = 0; k < 10_000; k++) {
      long hand = SeedMixer.mix64(k);
      int seat = k & 3, ordinal = k % 37;
      for (EngineDecisionKind kind : EngineDecisionKind.values()) {
        int kindClass = CrnScheme.SINGLE_RACE.kindClass(kind);
        Assert.assertEquals(kindClass, CrnKeys.kindClass(kind));
        long key = CrnScheme.SINGLE_RACE.decisionKey(hand, seat, kindClass, ordinal);
        Assert.assertEquals(key, CrnKeys.decisionKey(hand, seat, kindClass, ordinal));
        Assert.assertEquals(
            CrnScheme.SINGLE_RACE.select(key, LEGAL, sigma, b, 0.1), CrnKeys.select(key, LEGAL, b, 0));
      }
    }
  }

  /** {@code TWO_STAGE_FLOOR} で 10^5 個のキーから選んだ頻度が、b = (1 − ε)·σ + ε/n に合う（|z| < 4.5。σ は打ち手の分布、ε は混ぜる割合、n は合法手の数）。σ の和が 0 なら b は一様分布。 */
  @Test
  public void twoStageFloorDrawsBehaviour() {
    double epsilon = 0.1;
    int n = LEGAL.length, keys = 100_000;
    for (float[] sigma : new float[][] {{0, 0, 1, 0, 0}, {0.5f, 0, 0.25f, 0.25f, 0}, {0, 0, 0, 0, 0}}) {
      float[] b = new Behaviour(epsilon, 0, CrnScheme.TWO_STAGE_FLOOR).policy(sigma, n);
      long[] counts = new long[n];
      for (int k = 0; k < keys; k++) {
        counts[CrnScheme.TWO_STAGE_FLOOR.select(SeedMixer.mix64(31L * k + 7), LEGAL, sigma, b, epsilon)]++;
      }
      for (int a = 0; a < n; a++) {
        double expected = keys * (double) b[a];
        double z = (counts[a] - expected) / Math.sqrt(Math.max(expected * (1 - b[a]), 1));
        Assert.assertTrue(Math.abs(z) < 4.5, "action " + a + " count " + counts[a] + " expected " + expected);
      }
    }
  }

  /**
   * 兄弟の分岐のテスト。同じキー・同じ合法手で打ち手の分布（貪欲の手）だけが違うとき、{@code TWO_STAGE_FLOOR} では、一様に引くかどうかが
   * 同じキーで決まり、一様に引いた判断では同じ手を引く。{@code SINGLE_RACE} では貪欲の手によって競争の相手が変わるので、どちらかが
   * 貪欲の手から外れても、2 つの分岐で引いた手が揃わない判断がある。
   */
  @Test
  public void siblingsFireTogether() {
    double epsilon = 0.2;
    int n = LEGAL.length;
    float[] first = {1, 0, 0, 0, 0};
    float[] second = {0, 0, 0, 1, 0};
    float[] firstB = new Behaviour(epsilon, 0, CrnScheme.TWO_STAGE_FLOOR).policy(first, n);
    float[] secondB = new Behaviour(epsilon, 0, CrnScheme.TWO_STAGE_FLOOR).policy(second, n);
    int fired = 0, firstSchemeMismatch = 0;
    for (int k = 0; k < 20_000; k++) {
      long key = SeedMixer.mix64(k + 1_000_003L);
      int a = CrnScheme.TWO_STAGE_FLOOR.select(key, LEGAL, first, firstB, epsilon);
      int b = CrnScheme.TWO_STAGE_FLOOR.select(key, LEGAL, second, secondB, epsilon);
      boolean deviatedA = a != 0, deviatedB = b != 3;
      if (deviatedA || deviatedB || a == b) {
        // 発火した判断。一様に引いた手が貪欲の手と同じなら、発火しても貪欲の手になる。
        fired++;
        Assert.assertEquals(a, b, "key " + key);
      }
      int v1a = CrnScheme.SINGLE_RACE.select(key, LEGAL, first, firstB, epsilon);
      int v1b = CrnScheme.SINGLE_RACE.select(key, LEGAL, second, secondB, epsilon);
      if ((v1a != 0 || v1b != 3) && v1a != v1b) firstSchemeMismatch++;
    }
    Assert.assertTrue(fired > 3_000, "fired " + fired);
    Assert.assertTrue(firstSchemeMismatch > 100, "v1 mismatches " + firstSchemeMismatch);
  }

  /**
   * {@code TWO_STAGE_FLOOR} の判断のクラスは 3 つで、鳴いた後の打牌は手番と別のクラスになる。槍槓と抜いた北への応答は、打牌への応答と
   * 同じクラス（抜いた北への応答は {@code SINGLE_RACE} でも同じ）。席と判断のクラスの組ごとのキーは、序数が同じでも重ならない。
   */
  @Test
  public void secondSchemeSeparatesPostCallDiscards() {
    Assert.assertEquals(CrnScheme.TWO_STAGE_FLOOR.kindClasses(), 3);
    Assert.assertNotEquals(
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.POST_CALL_DAHAI),
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.TURN));
    Assert.assertEquals(
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.CHANKAN),
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.RESPONSE));
    Assert.assertEquals(
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.NUKI_RESPONSE),
        CrnScheme.TWO_STAGE_FLOOR.kindClass(EngineDecisionKind.RESPONSE));
    Assert.assertEquals(
        CrnScheme.SINGLE_RACE.kindClass(EngineDecisionKind.NUKI_RESPONSE),
        CrnScheme.SINGLE_RACE.kindClass(EngineDecisionKind.RESPONSE));
    Set<Long> keys = new HashSet<>();
    for (int seat = 0; seat < 4; seat++) {
      for (int kindClass = 0; kindClass < 3; kindClass++) {
        Assert.assertTrue(keys.add(CrnScheme.TWO_STAGE_FLOOR.decisionKey(123L, seat, kindClass, 5)));
      }
    }
    CrnKeys.Cursor cursor = new CrnKeys.Cursor(4, CrnScheme.TWO_STAGE_FLOOR.kindClasses());
    Assert.assertEquals(cursor.next(3, 2), 0);
    Assert.assertEquals(cursor.next(3, 2), 1);
    Assert.assertEquals(cursor.next(3, 0), 0);
    Assert.assertEquals(cursor.copy().next(3, 2), 2);
  }
}
