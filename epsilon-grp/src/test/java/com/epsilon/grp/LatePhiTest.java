package com.epsilon.grp;

import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 終盤だけを数え上げる Φ（Phi.late）を確かめる。値は、覆う卓では FinalRound.phi と、ほかの卓では葉の GrpNetwork だけの Phi と、
 * 終局の卓では順位の効用と、ビット単位で同じ。evaluate の Φ は values と同じビットで、周辺確率もそれぞれの求め方と同じ。値はメモの
 * 有無・スロットの数・スレッドの数・呼ぶ順に依らない。終盤のメモは卓の全てのフィールドで引くので、違う卓の値を混ぜない。終盤のメモは
 * 全てのスレッドで共有し、ほかのスレッドが数え上げた卓にも当たる。Stats の終盤の数えは、同じ順に呼んだ FinalRound の作業領域の数えと
 * 合う。四人打ちと三人打ちの両方で確かめる。
 */
public class LatePhiTest {

  private static final RankUtility UTILITY = RankUtility.TENHOU;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** 覆う卓は数え上げ、ほかは葉のネットワーク、終局の卓は効用で求める。GrpState と TableState のどちらから呼んでも同じ。 */
  @Test(dataProvider = "variants")
  public void coveredTablesEnumerateAndOthersMatchTheLeafBits(Variant variant) {
    FinalRound round = round(variant, 1e-5);
    Phi late = Phi.late(round, UTILITY, 64);
    Phi leaf = Phi.of(round.leaf(), UTILITY);
    Assert.assertEquals(late.kind(), Phi.Kind.LATE);
    Assert.assertSame(late.model(), round.leaf());
    Assert.assertSame(late.finalRound(), round);
    Assert.assertNull(leaf.finalRound());
    Assert.assertFalse(late.needsHistory());
    int seats = variant.seats();
    double[] values = new double[seats];
    double[] expected = new double[seats];
    TableState table = new TableState(variant);
    int covered = 0;
    for (GrpState state : states(variant, 300, 3)) {
      reference(round, leaf, state, expected);
      late.values(state, values);
      assertBits(values, expected, state);
      state.copyInto(table);
      late.values(table, values);
      assertBits(values, expected, state);
      if (FinalRound.covers(state)) {
        covered++;
      }
    }
    Assert.assertTrue(covered > 60 && covered < 240, covered + " covered");
    TableState finished = finished(variant);
    late.values(finished, values);
    leaf.values(finished, expected);
    assertBits(values, expected, null);
  }

  /** evaluate の Φ は values と同じビット。周辺確率は、覆う卓では数え上げの p(π|s) と、ほかでは葉の Phi の evaluate と同じ。evaluate は数えに入らない。 */
  @Test(dataProvider = "variants")
  public void evaluateMatchesTheValueBits(Variant variant) {
    FinalRound round = round(variant, 1e-5);
    Phi late = Phi.late(round, UTILITY, 64);
    Phi leaf = Phi.of(round.leaf(), UTILITY);
    int seats = variant.seats();
    RankPermutations table = RankPermutations.of(variant);
    double[] values = new double[seats];
    double[] phi = new double[seats];
    double[] marginals = new double[seats * seats];
    double[] expected = new double[seats * seats];
    double[] leafPhi = new double[seats];
    double[] p = new double[table.count()];
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    List<GrpState> states = states(variant, 200, 5);
    Phi.Stats before = late.stats();
    for (GrpState state : states) {
      late.evaluate(null, state, phi, marginals);
      if (FinalRound.covers(state)) {
        round.permutations(state, workspace, p);
        table.marginals(p, state.dealer(), expected);
      } else {
        leaf.evaluate(null, state, leafPhi, expected);
      }
      assertBits(marginals, expected, state);
    }
    Assert.assertEquals(late.stats(), before);
    for (GrpState state : states) {
      late.evaluate(null, state, phi, marginals);
      late.values(state, values);
      assertBits(phi, values, state);
    }
  }

  /**
   * 終盤のメモが 1 スロット（呼ぶたびに追い出す）でも 16,384 スロットでも、呼ぶ順を変えても、8 本のスレッドが 1 つのメモを共有して
   * 別々の順に 2 度ずつ呼んでも（1 スロットなら全てのスレッドが同じスロットを書き合う）、値は 1 卓ずつ求めた値とビット単位で同じ。
   * 共有したメモの呼び出しの数は全てのスレッドの和で、キーに収まる覆う卓はどれも 1 度は数え上げる。
   */
  @Test(dataProvider = "variants")
  public void memoSlotsThreadsAndOrderDoNotChangeTheBits(Variant variant) throws Exception {
    FinalRound round = round(variant, 1e-5);
    Phi leaf = Phi.of(round.leaf(), UTILITY);
    List<GrpState> states = states(variant, 400, 7);
    double[][] expected = new double[states.size()][variant.seats()];
    for (int index = 0; index < states.size(); index++) {
      reference(round, leaf, states.get(index), expected[index]);
    }
    for (int slots : new int[] {1, 2, 16_384}) {
      Phi late = Phi.late(round, UTILITY, slots);
      for (long seed = 0; seed < 2; seed++) {
        double[][] values = valuesInShuffledOrder(late, states, seed);
        for (int index = 0; index < states.size(); index++) {
          assertBits(values[index], expected[index], states.get(index));
        }
      }
    }
    int threads = 8;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      for (int slots : new int[] {1, 256}) {
        Phi shared = Phi.late(round, UTILITY, slots);
        List<Future<double[][][]>> results = new ArrayList<>();
        for (int thread = 0; thread < threads; thread++) {
          long seed = 100 + thread;
          results.add(pool.submit(() -> new double[][][] {
              valuesInShuffledOrder(shared, states, seed), valuesInShuffledOrder(shared, states, seed + 50)}));
        }
        for (Future<double[][][]> result : results) {
          for (double[][] values : result.get()) {
            for (int index = 0; index < states.size(); index++) {
              assertBits(values[index], expected[index], states.get(index));
            }
          }
        }
        Phi.LateStats late = shared.stats().late();
        Assert.assertEquals(late.requests(), 2L * threads * coveredCount(states));
        Assert.assertEquals(late.hits() + late.misses() + late.bypasses(), late.requests());
        Assert.assertTrue(late.misses() >= coveredKeys(states), slots + " slots: " + late);
      }
    } finally {
      pool.shutdownNow();
    }
  }

  /**
   * 別のスレッドが数え上げた卓は、数え上げずに共有の終盤のメモから読み、値は同じビット。1 スロットのメモでも、別のスレッドが書いた
   * 直後に引けば当たる。別に開いた Phi はメモを共有しない。
   */
  @Test(dataProvider = "variants")
  public void anotherThreadsTableIsAHit(Variant variant) throws Exception {
    FinalRound round = round(variant, 1e-5);
    Phi late = Phi.late(round, UTILITY, 1);
    List<GrpState> covered = new ArrayList<>();
    for (GrpState state : states(variant, 100, 19)) {
      if (FinalRound.covers(state) && state.key() != GrpKey.NONE) {
        covered.add(state);
      }
    }
    int seats = variant.seats();
    double[] theirs = new double[seats];
    double[] ours = new double[seats];
    ExecutorService other = Executors.newSingleThreadExecutor();
    try {
      for (GrpState state : covered) {
        other.submit(() -> late.values(state, theirs)).get();
        late.values(state, ours);
        assertBits(ours, theirs, state);
      }
    } finally {
      other.shutdownNow();
    }
    Phi.LateStats stats = late.stats().late();
    Assert.assertEquals(stats.requests(), 2L * covered.size());
    Assert.assertEquals(stats.hits(), covered.size(), stats.toString());
    Assert.assertEquals(stats.misses(), covered.size(), stats.toString());

    Phi reopened = Phi.late(round, UTILITY, 1);
    reopened.values(covered.get(covered.size() - 1), ours);
    assertBits(ours, theirs, covered.get(covered.size() - 1));
    Assert.assertEquals(reopened.stats().late().misses(), 1);
  }

  /**
   * 覆う卓から 1 つの値（本場・供託・点・局）だけを変えた卓は、キーも値も元の卓と違い、どれも 1 卓ずつ求めた値と同じになる。
   * 1 スロットの終盤のメモで交互に呼んでも、16,384 スロットのメモで 2 度呼んでも混ざらない。キーから卓を作り直すと元の卓に戻る。
   */
  @Test(dataProvider = "variants")
  public void theLateMemoKeepsEveryTableApart(Variant variant) {
    FinalRound round = round(variant, 1e-5);
    Phi leaf = Phi.of(round.leaf(), UTILITY);
    int seats = variant.seats();
    int[] scores = seats == 4 ? new int[] {27_000, 25_000, 24_000, 23_000} : new int[] {36_000, 35_000, 33_000};
    int last = variant.lastRound();
    GrpState base = new GrpState(variant, last, 1, 1, scores);
    int[] moved = scores.clone();
    moved[0] -= 100;
    moved[1] += 100;
    int[] deposited = scores.clone();
    deposited[1] -= Variant.DEPOSIT;
    List<GrpState> family = List.of(
        base,
        new GrpState(variant, last, 2, 1, scores),
        new GrpState(variant, last, 1, 2, deposited),
        new GrpState(variant, last, 1, 1, moved),
        new GrpState(variant, last + 1, 1, 1, scores));
    List<Long> keys = new ArrayList<>();
    double[][] expected = new double[family.size()][seats];
    for (int index = 0; index < family.size(); index++) {
      GrpState state = family.get(index);
      Assert.assertTrue(FinalRound.covers(state), state.toString());
      Assert.assertEquals(GrpState.ofKey(variant, state.key()), state);
      Assert.assertFalse(keys.contains(state.key()), state.toString());
      keys.add(state.key());
      reference(round, leaf, state, expected[index]);
      for (int other = 0; other < index; other++) {
        Assert.assertFalse(Arrays.equals(expected[index], expected[other]), state + " and " + family.get(other));
      }
    }
    double[] values = new double[seats];
    for (int slots : new int[] {1, 16_384}) {
      Phi late = Phi.late(round, UTILITY, slots);
      for (int pass = 0; pass < 3; pass++) {
        for (int index = 0; index < family.size(); index++) {
          late.values(family.get(index), values);
          assertBits(values, expected[index], family.get(index));
          late.values(family.get(index), values);
          assertBits(values, expected[index], family.get(index));
        }
      }
      Phi.LateStats stats = late.stats().late();
      Assert.assertEquals(stats.requests(), 6L * family.size());
      Assert.assertEquals(stats.misses(), slots == 1 ? 3L * family.size() : family.size(), stats.toString());
    }
  }

  /**
   * 覆う卓 N 個を 2 度ずつ、覆わない卓 M 個を 1 度、終局の卓を 1 度、キーに収まらない覆う卓を 2 度呼ぶ。Phi 全体の数えは全ての
   * 呼び出しを、終盤の数えは覆う卓の呼び出しだけを数える。精算・葉のネットワーク・葉のメモの当たり・埋めた確率の和は、数え上げた卓を
   * 同じ順に FinalRound の作業領域 1 つで求めたときの数えと同じ。since は差を返す。ネットワークだけの Phi では、終盤の数えは 0。
   */
  @Test(dataProvider = "variants")
  public void statsCountTheLateCalls(Variant variant) {
    FinalRound round = round(variant, 1e-5);
    Phi late = Phi.late(round, UTILITY, 1_024);
    int seats = variant.seats();
    List<GrpState> covered = new ArrayList<>();
    List<GrpState> uncovered = new ArrayList<>();
    for (GrpState state : states(variant, 300, 11)) {
      (FinalRound.covers(state) ? covered : uncovered).add(state);
    }
    GrpState first = covered.get(0);
    GrpState outside = new GrpState(variant, first.round(), 70, first.sticks(), first.scores());
    Assert.assertEquals(outside.key(), GrpKey.NONE);
    FinalRoundWorkspace mirror = FinalRoundWorkspace.of(round);
    double[] values = new double[seats];
    double filled = 0;
    Phi.Stats start = late.stats();
    for (GrpState state : covered) {
      late.values(state, values);
      late.values(state, values);
      filled += round.phi(state, UTILITY, mirror, values);
    }
    for (GrpState state : uncovered) {
      late.values(state, values);
    }
    late.values(finished(variant), values);
    for (int repeat = 0; repeat < 2; repeat++) {
      late.values(outside, values);
      filled += round.phi(outside, UTILITY, mirror, values);
    }
    Phi.Stats stats = late.stats().since(start);
    int n = covered.size();
    int m = uncovered.size();
    Assert.assertEquals(stats.requests(), 2L * n + m + 3);
    Assert.assertEquals(stats.hits(), n);
    Assert.assertEquals(stats.misses(), (long) n + m);
    Assert.assertEquals(stats.bypasses(), 2);
    Phi.LateStats lateStats = stats.late();
    Assert.assertEquals(lateStats.requests(), 2L * n + 2);
    Assert.assertEquals(lateStats.hits(), n);
    Assert.assertEquals(lateStats.misses(), n);
    Assert.assertEquals(lateStats.bypasses(), 2);
    Assert.assertEquals(lateStats.settled(), mirror.settled());
    Assert.assertEquals(lateStats.leafNetworks(), mirror.networkLeaves());
    Assert.assertEquals(lateStats.leafHits(), mirror.memoHits());
    Assert.assertEquals(lateStats.filled(), filled, 1e-12 * n);
    Assert.assertTrue(lateStats.filled() > 0, lateStats.toString());
    Assert.assertTrue(lateStats.nanos() > 0 && lateStats.nanos() <= stats.nanos(), stats.toString());
    Assert.assertEquals(late.stats().since(late.stats()).late(), new Phi.LateStats(0, 0, 0, 0, 0, 0, 0, 0, 0));

    Phi network = Phi.of(round.leaf(), UTILITY);
    for (GrpState state : covered) {
      network.values(state, values);
    }
    Assert.assertEquals(network.stats().late(), new Phi.LateStats(0, 0, 0, 0, 0, 0, 0, 0, 0));
    Assert.assertEquals(network.stats().misses(), n);
  }

  /** 終盤のメモのスロットは 2 の冪であること。ルールの違う卓と、長さの違う出力先は拒む。 */
  @Test
  public void lateNeedsPowerOfTwoSlots() {
    FinalRound round = round(Variant.TENHOU_4P, 1e-5);
    for (int slots : new int[] {0, 3, -4}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> Phi.late(round, UTILITY, slots));
    }
    Phi late = Phi.late(round, UTILITY, 64);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> late.values(new GrpState(Variant.TENHOU_3P, 5, 0, 0, new int[] {35_000, 35_000, 35_000}), new double[4]));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> late.values(new GrpState(Variant.TENHOU_4P, 7, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000}),
            new double[3]));
    Assert.assertThrows(IllegalStateException.class, () -> Phi.of(round.leaf(), UTILITY).finalRoundWorkspace());
  }

  /**
   * 数え上げは、参照用の実装（FinalRoundReference。葉を 1 卓ずつ求めてその場で足す）とビット単位で同じ。打ち切りの値と、
   * 葉のメモのスロットの数（1・8・64・1,024）に依らない。埋めた確率と、精算の数、半荘が終わる葉の数も同じ。
   */
  @Test(dataProvider = "variants")
  public void finalRoundMatchesTheReferenceCopy(Variant variant) {
    List<GrpState> covered = new ArrayList<>();
    for (GrpState state : states(variant, 400, 13)) {
      if (FinalRound.covers(state) && covered.size() < 40) {
        covered.add(state);
      }
    }
    int count = RankPermutations.of(variant).count();
    double[] p = new double[count];
    double[] expected = new double[count];
    for (double cutoff : new double[] {1e-4, 1e-5, 1e-7}) {
      FinalRound round = round(variant, cutoff);
      FinalRoundReference reference = FinalRoundReference.of(round);
      double[][] references = new double[covered.size()][];
      double[] dropped = new double[covered.size()];
      for (int index = 0; index < covered.size(); index++) {
        dropped[index] = reference.permutations(covered.get(index), expected);
        references[index] = expected.clone();
      }
      for (int slots : new int[] {1, 8, 64, 1_024}) {
        FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round, slots);
        Assert.assertEquals(workspace.leafSlots(), slots);
        for (int index = 0; index < covered.size(); index++) {
          double filled = round.permutations(covered.get(index), workspace, p);
          Assert.assertEquals(Double.doubleToRawLongBits(filled), Double.doubleToRawLongBits(dropped[index]));
          assertBits(p, references[index], covered.get(index));
        }
        Assert.assertEquals(workspace.settled(), reference.settled());
        Assert.assertEquals(workspace.terminalLeaves(), reference.terminalLeaves());
      }
    }
  }

  /**
   * 終盤の Φ は、数え上げ（1 スロットの終盤のメモで毎回数え上げる）・葉のネットワーク・終局の卓のどれでも、GrpState と TableState
   * のどちらからでも、ウォームアップの後はメモリを確保しない。
   */
  @Test(dataProvider = "variants")
  public void steadyLateValuesDoNotAllocate(Variant variant) {
    Phi late = Phi.late(round(variant, 1e-4), UTILITY, 1);
    GrpState[] states = states(variant, 60, 17).toArray(GrpState[]::new);
    TableState table = new TableState(variant);
    TableState finished = finished(variant);
    double[] values = new double[variant.seats()];
    Allocations.assertSteady(3 * states.length + " late values", 5, () -> {
      for (GrpState state : states) {
        late.values(state, values);
        state.copyInto(table);
        late.values(table, values);
      }
      late.values(finished, values);
    });
    Assert.assertTrue(late.stats().late().misses() > 0);
  }

  /** 1 卓ずつ求めた値を書く。覆う卓では新しい作業領域で FinalRound.phi を、ほかの卓では葉の Phi を使う。 */
  private static void reference(FinalRound round, Phi leaf, GrpState state, double[] out) {
    if (FinalRound.covers(state)) {
      round.phi(state, UTILITY, FinalRoundWorkspace.of(round), out);
    } else {
      leaf.values(state, out);
    }
  }

  static FinalRound round(Variant variant, double cutoff) {
    OutcomeShape outcomes = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    GrpShape leaf = GrpFixtures.shape(variant, 8, false);
    return FinalRound.of(GrpFixtures.outcomes(outcomes, 31, 0.5), GrpFixtures.network(leaf, 37, 0.5), cutoff);
  }

  /** 乱数で卓を作る。半分（偶数番目）は覆う卓（南場の最後の局から西場の最後の局まで）にする。 */
  static List<GrpState> states(Variant variant, int count, long seed) {
    Random random = new Random(seed);
    List<GrpState> states = new ArrayList<>(count);
    int last = variant.lastRound();
    for (int index = 0; index < count; index++) {
      GrpState state = GrpFixtures.state(variant, random);
      if (index % 2 == 0) {
        int round = last + random.nextInt(variant.finalRound() - last + 1);
        state = new GrpState(variant, round, state.honba(), state.sticks(), state.scores());
      }
      states.add(state);
    }
    return states;
  }

  /** 南場の最後の局の親が不聴で流れ、トップが返し点に届いて終わった卓。 */
  static TableState finished(Variant variant) {
    int seats = variant.seats();
    int[] scores = seats == 4 ? new int[] {40_000, 20_000, 20_000, 20_000} : new int[] {45_000, 30_000, 30_000};
    TableState start = new TableState(variant);
    start.set(variant.lastRound(), 0, 0, scores);
    RoundOutcome draw = new RoundOutcome(variant);
    int[] delta = new int[seats];
    Settlement.apply(variant, start, draw, delta);
    TableState end = new TableState(variant);
    Progression.next(start, draw, delta, end);
    Assert.assertTrue(end.gameOver(), end.toString());
    return end;
  }

  private static int coveredCount(List<GrpState> states) {
    int count = 0;
    for (GrpState state : states) {
      if (FinalRound.covers(state)) {
        count++;
      }
    }
    return count;
  }

  /** キーに収まる覆う卓の、キーの種類の数。 */
  private static long coveredKeys(List<GrpState> states) {
    return states.stream()
        .filter(state -> FinalRound.covers(state))
        .mapToLong(GrpState::key)
        .filter(key -> key != GrpKey.NONE)
        .distinct()
        .count();
  }

  private static double[][] valuesInShuffledOrder(Phi phi, List<GrpState> states, long seed) {
    List<Integer> order = new ArrayList<>();
    for (int index = 0; index < states.size(); index++) {
      order.add(index);
    }
    Collections.shuffle(order, new Random(seed));
    double[][] values = new double[states.size()][phi.variant().seats()];
    for (int index : order) {
      phi.values(states.get(index), values[index]);
    }
    return values;
  }

  private static void assertBits(double[] actual, double[] expected, GrpState state) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < actual.length; index++) {
      Assert.assertEquals(
          Double.doubleToRawLongBits(actual[index]), Double.doubleToRawLongBits(expected[index]),
          state + " at " + index);
    }
  }
}
