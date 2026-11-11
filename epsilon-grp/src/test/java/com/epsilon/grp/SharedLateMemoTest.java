package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 全てのスレッドで共有する終盤のメモ（SharedPhiMemo と、それを持つ Phi.late）を確かめる。共有のメモを通した値は、メモを使わずに
 * 1 卓ずつ求めた値と、スレッドの数・スロットの数・呼ぶ順に依らずビット単位で同じ。多くのスレッドがスロットを書き合っても、別のキーの
 * 値を返さない。q・葉・打ち切りの違う Phi は、互いのメモを読まない。四人打ちと三人打ちの両方で確かめる。スロットの数と席の数の積が
 * int に収まらないメモは、配列を確保する前に拒む。
 */
public class SharedLateMemoTest {

  private static final RankUtility UTILITY = RankUtility.TENHOU;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /**
   * 1・4・16 本のスレッドが 1 つの Phi を共有し、それぞれ別の順に覆う卓を全て呼ぶ（半分のスレッドは TableState から）。スロットが 1・64・
   * 65,536 のどれでも、値はメモを使わずに求めた値とビット単位で同じ。呼び出しの数は全てのスレッドの和で、当たり・外れ・メモを通らなかった
   * 数の和に合い、外れはキーの種類の数以上、メモを通らなかった数はキーに収まらない卓の数のスレッド倍。
   */
  @Test(dataProvider = "variants")
  public void sharedValuesAreTheBitsWithoutAMemo(Variant variant) throws Exception {
    FinalRound round = LatePhiTest.round(variant, 1e-5);
    List<GrpState> covered = covered(LatePhiTest.states(variant, 500, 29));
    double[][] expected = new double[covered.size()][variant.seats()];
    for (int index = 0; index < covered.size(); index++) {
      round.phi(covered.get(index), UTILITY, FinalRoundWorkspace.of(round), expected[index]);
    }
    long keys = covered.stream().mapToLong(GrpState::key).filter(key -> key != GrpKey.NONE).distinct().count();
    long outside = covered.stream().filter(state -> state.key() == GrpKey.NONE).count();
    Assert.assertTrue(covered.size() > 150 && keys > 150, covered.size() + " covered, " + keys + " keys");
    for (int threads : new int[] {1, 4, 16}) {
      ExecutorService pool = Executors.newFixedThreadPool(threads);
      try {
        for (int slots : new int[] {1, 64, 65_536}) {
          Phi late = Phi.late(round, UTILITY, slots);
          CyclicBarrier start = new CyclicBarrier(threads);
          List<Future<double[][]>> results = new ArrayList<>();
          for (int thread = 0; thread < threads; thread++) {
            long seed = 1_000L * threads + slots + thread;
            boolean viaTable = thread % 2 == 1;
            results.add(pool.submit(() -> {
              start.await();
              return shuffledValues(late, covered, seed, viaTable);
            }));
          }
          for (Future<double[][]> result : results) {
            double[][] values = result.get();
            for (int index = 0; index < covered.size(); index++) {
              assertBits(values[index], expected[index], threads + " threads, " + slots + " slots, " + covered.get(index));
            }
          }
          Phi.LateStats stats = late.stats().late();
          String label = threads + " threads, " + slots + " slots: " + stats;
          Assert.assertEquals(stats.requests(), (long) threads * covered.size(), label);
          Assert.assertEquals(stats.hits() + stats.misses() + stats.bypasses(), stats.requests(), label);
          Assert.assertTrue(stats.misses() >= keys, label);
          Assert.assertEquals(stats.bypasses(), threads * outside, label);
          if (slots == 65_536 && threads == 16) {
            Assert.assertTrue(stats.hits() > 0, label);
          }
        }
      } finally {
        pool.shutdownNow();
      }
    }
  }

  /**
   * 16 本のスレッドが 1・4・2,048 スロットのメモ（2,048 ではロックを 2 つのスロットで分け合う）を、512 種類のキーで書き合い、引き合う。
   * 当たったときの Φ は、どれもそのキーの Φ とビット単位で同じ（別のキーの値も、別のキーと混ざった値も返さない）。スレッドごとの当たりと
   * 外れの和は、そのスレッドの呼び出しの数に合う。
   */
  @Test
  public void collidingThreadsNeverReadAnotherKeysValue() throws Exception {
    int threads = 16;
    int calls = 50_000;
    long[] keys = new long[512];
    Random random = new Random(41);
    Set<Long> distinct = new HashSet<>();
    for (int index = 0; index < keys.length; index++) {
      long key;
      do {
        key = random.nextLong() >>> 1;
      } while (key == GrpKey.NONE || !distinct.add(key));
      keys[index] = key;
    }
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      for (int seats : new int[] {4, 3}) {
        for (int slots : new int[] {1, 4, 2_048}) {
          SharedPhiMemo memo = new SharedPhiMemo(slots, seats);
          CyclicBarrier start = new CyclicBarrier(threads);
          List<Future<long[]>> results = new ArrayList<>();
          for (int thread = 0; thread < threads; thread++) {
            long seed = 7_000L * seats + slots * 31L + thread;
            results.add(pool.submit(() -> {
              start.await();
              return hammer(memo, keys, seats, calls, seed);
            }));
          }
          long hits = 0;
          for (Future<long[]> result : results) {
            long[] counts = result.get();
            Assert.assertEquals(counts[0] + counts[1], calls, seats + " seats, " + slots + " slots");
            hits += counts[0];
          }
          Assert.assertTrue(hits > 0, seats + " seats, " + slots + " slots");
        }
      }
    } finally {
      pool.shutdownNow();
    }
  }

  /** 新しいメモは何にも当たらず、書いた卓には当たる。スロットの数は 1 以上の 2 の冪だけを受ける。 */
  @Test
  public void anEmptyMemoMissesAndSlotsArePowersOfTwo() {
    SharedPhiMemo memo = new SharedPhiMemo(8, 4);
    double[] out = {9, 9, 9, 9};
    Assert.assertFalse(memo.find(12_345L, out));
    Assert.assertTrue(Arrays.equals(out, new double[] {9, 9, 9, 9}), Arrays.toString(out));
    memo.store(12_345L, new double[] {1, 2, 3, 4});
    Assert.assertTrue(memo.find(12_345L, out));
    Assert.assertTrue(Arrays.equals(out, new double[] {1, 2, 3, 4}), Arrays.toString(out));
    Assert.assertFalse(memo.find(12_346L, out));
    for (int slots : new int[] {0, -1, 3, 6, -4, Integer.MIN_VALUE, (1 << 20) + 1}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> new SharedPhiMemo(slots, 4));
    }
    FinalRound round = LatePhiTest.round(Variant.TENHOU_4P, 1e-5);
    Assert.assertThrows(IllegalArgumentException.class, () -> Phi.late(round, UTILITY, Integer.MIN_VALUE));
    Assert.assertEquals(Phi.late(round, UTILITY, 1 << 20).kind(), Phi.Kind.LATE);
  }

  /**
   * スロットの数と席の数の積が int に収まらないメモは、配列を確保する前に、スロットと席の数を添えた英語の 1 文で拒む（四人打ちの 2^29
   * スロットは負の長さに、2^30 スロットは長さ 0 に桁あふれする）。Phi.late も同じ文で拒む。
   */
  @Test
  public void slotsBeyondAnIntArrayFailBeforeAllocating() {
    for (int[] memo : new int[][] {{1 << 29, 4}, {1 << 30, 4}, {1 << 30, 3}, {1 << 30, 2}}) {
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> new SharedPhiMemo(memo[0], memo[1]));
      Assert.assertEquals(
          failure.getMessage(),
          "A memo of " + memo[0] + " slots for " + memo[1] + " seats needs more than 2147483647 values");
      Assert.assertTrue(failure.getCause() instanceof ArithmeticException, failure.toString());
    }
    for (Variant variant : FinalRound.VARIANTS) {
      FinalRound round = LatePhiTest.round(variant, 1e-5);
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> Phi.late(round, UTILITY, 1 << 30));
      Assert.assertEquals(
          failure.getMessage(),
          "A memo of " + (1 << 30) + " slots for " + variant.seats() + " seats needs more than 2147483647 values");
    }
  }

  /**
   * 同じ覆う卓を、q・葉・打ち切りの違う 4 つの Phi で交互に呼ぶ。どの Phi の値も、自分の FinalRound でメモを使わずに求めた値とビット単位で
   * 同じで、ほかの 3 つの値とはどこかで違う。初めの 1 巡では、どの Phi もメモに当たらない（ほかの Phi が先に求めた卓でも数え上げる）。
   */
  @Test(dataProvider = "variants")
  public void anotherValueNeverReadsThisMemo(Variant variant) {
    FinalRound base = LatePhiTest.round(variant, 1e-5);
    OutcomeShape outcomes = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    GrpShape leaf = GrpFixtures.shape(variant, 8, false);
    List<FinalRound> rounds = List.of(
        base,
        FinalRound.of(GrpFixtures.outcomes(outcomes, 32, 0.5), base.leaf(), 1e-5),
        FinalRound.of(base.outcomes(), GrpFixtures.network(leaf, 38, 0.5), 1e-5),
        FinalRound.of(base.outcomes(), base.leaf(), 1e-2));
    List<GrpState> covered = distinctKeys(covered(LatePhiTest.states(variant, 160, 43)));
    Assert.assertTrue(covered.size() > 40, covered.size() + " covered");
    double[][][] expected = new double[rounds.size()][covered.size()][variant.seats()];
    for (int which = 0; which < rounds.size(); which++) {
      FinalRound round = rounds.get(which);
      for (int index = 0; index < covered.size(); index++) {
        round.phi(covered.get(index), UTILITY, FinalRoundWorkspace.of(round), expected[which][index]);
      }
    }
    for (int which = 1; which < rounds.size(); which++) {
      boolean differs = false;
      for (int index = 0; index < covered.size(); index++) {
        differs |= !Arrays.equals(expected[which][index], expected[0][index]);
      }
      Assert.assertTrue(differs, rounds.get(which) + " gives the base values");
    }
    double[] values = new double[variant.seats()];
    for (int slots : new int[] {1, 4_096}) {
      List<Phi> phis = new ArrayList<>();
      for (FinalRound round : rounds) {
        phis.add(Phi.late(round, UTILITY, slots));
      }
      for (int pass = 0; pass < 2; pass++) {
        for (int index = 0; index < covered.size(); index++) {
          for (int which = 0; which < phis.size(); which++) {
            phis.get(which).values(covered.get(index), values);
            assertBits(values, expected[which][index], rounds.get(which) + ", " + covered.get(index));
          }
        }
        if (pass == 0) {
          for (Phi phi : phis) {
            Phi.LateStats stats = phi.stats().late();
            Assert.assertEquals(stats.hits(), 0, phi + ": " + stats);
            Assert.assertEquals(stats.misses(), covered.size(), phi + ": " + stats);
          }
        }
      }
    }
  }

  /** 覆う卓のうち、キーに収まらない卓も残す。 */
  private static List<GrpState> covered(List<GrpState> states) {
    List<GrpState> covered = new ArrayList<>();
    for (GrpState state : states) {
      if (FinalRound.covers(state)) {
        covered.add(state);
      }
    }
    return covered;
  }

  /** キーに収まり、キーが重ならない卓だけ。 */
  private static List<GrpState> distinctKeys(List<GrpState> states) {
    Set<Long> seen = new HashSet<>();
    List<GrpState> distinct = new ArrayList<>();
    for (GrpState state : states) {
      if (state.key() != GrpKey.NONE && seen.add(state.key())) {
        distinct.add(state);
      }
    }
    return distinct;
  }

  /** 卓を乱数の種で混ぜた順に呼んだ値を返す（並びは元の順）。{@code viaTable} なら TableState にコピーしてから呼ぶ。 */
  private static double[][] shuffledValues(Phi phi, List<GrpState> states, long seed, boolean viaTable) {
    List<Integer> order = new ArrayList<>();
    for (int index = 0; index < states.size(); index++) {
      order.add(index);
    }
    Collections.shuffle(order, new Random(seed));
    double[][] values = new double[states.size()][phi.variant().seats()];
    TableState table = new TableState(phi.variant());
    for (int index : order) {
      if (viaTable) {
        states.get(index).copyInto(table);
        phi.values(table, values[index]);
      } else {
        phi.values(states.get(index), values[index]);
      }
    }
    return values;
  }

  /**
   * キーを乱数で選んで引き、当たれば Φ がそのキーの値と同じことを確かめ、外れればそのキーの値を書く。
   *
   * @return 当たりと外れの数
   */
  private static long[] hammer(SharedPhiMemo memo, long[] keys, int seats, int calls, long seed) {
    Random random = new Random(seed);
    double[] out = new double[seats];
    double[] phi = new double[seats];
    long hits = 0;
    long misses = 0;
    for (int call = 0; call < calls; call++) {
      long key = keys[random.nextInt(keys.length)];
      if (memo.find(key, out)) {
        hits++;
        for (int seat = 0; seat < seats; seat++) {
          if (Double.doubleToRawLongBits(out[seat]) != Double.doubleToRawLongBits(value(key, seat))) {
            throw new AssertionError("key " + key + " seat " + seat + " read " + out[seat] + ", not " + value(key, seat));
          }
        }
      } else {
        misses++;
        for (int seat = 0; seat < seats; seat++) {
          phi[seat] = value(key, seat);
        }
        memo.store(key, phi);
      }
    }
    return new long[] {hits, misses};
  }

  /** キーと席だけで決まる、席ごとに違う値。 */
  private static double value(long key, int seat) {
    return (SeedMixer.mix64(key * 31 + seat) >>> 11) * 0x1p-53;
  }

  private static void assertBits(double[] actual, double[] expected, String label) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++) {
      Assert.assertEquals(
          Double.doubleToRawLongBits(actual[index]), Double.doubleToRawLongBits(expected[index]), label + " at " + index);
    }
  }
}
