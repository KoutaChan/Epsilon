package com.epsilon.grp;

import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Phi を確かめる。終局の卓の厳密な効用、ネットワークとビット単位で同じ値、メモの当たりと外れ、キーに収まらない卓、スレッドに依らず
 * ビット単位で同じ値、メモリを確保しないこと、モデルごとのメモ。GruSinkhorn では、履歴を読むこと、履歴のバージョンを含むメモの
 * キー、表を重ねることも確かめる。
 */
public class PhiTest {

  @Test
  public void finishedTablesGetExactUtilitiesWithoutTheNetwork() {
    // 南 4 の親（絶対席 3）が不聴で流れ、トップが返し点に届いて終わる。絶対席 1〜3 は同点で、起家に近い順。
    TableState start = new TableState(Variant.TENHOU_4P);
    start.set(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, new int[] {40_000, 20_000, 20_000, 20_000}));
    TableState end = finish(start, new RoundOutcome(Variant.TENHOU_4P));
    Phi phi = Phi.of(GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 8, false), 1, 1.0), RankUtility.TENHOU);
    double[] values = new double[4];
    phi.values(end, values);
    Assert.assertEquals(values, new double[] {2.0 / 3, 1.0 / 3, 0, -1});
    assertUntouched(phi.memo());

    // 三人打ちの飛び。親（絶対席 1）が 8,000 点を放銃して 0 点未満になる。
    TableState three = new TableState(Variant.TENHOU_3P);
    three.set(new RoundStart(Variant.TENHOU_3P, 1, 0, 0, new int[] {50_000, 2_000, 53_000}));
    RoundOutcome ron = new RoundOutcome(Variant.TENHOU_3P);
    ron.setRon(BaseClass.MANGAN.base(), 0, 1, RoundOutcome.NO_SEAT, 0);
    Phi threePhi =
        Phi.of(GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_3P, 8, false), 2, 1.0), RankUtility.TENHOU);
    double[] threeValues = new double[3];
    threePhi.values(finish(three, ron), threeValues);
    Assert.assertEquals(threeValues, new double[] {1, -1, 0});
    assertUntouched(threePhi.memo());
  }

  @Test
  public void continuingTablesMatchTheNetworkBits() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 16, true), 3, 1.0);
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    GrpWorkspace workspace = GrpWorkspace.of(network.shape());
    TableState table = new TableState(Variant.TENHOU_4P);
    Random random = new Random(29);
    for (int trial = 0; trial < 500; trial++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      table.set(new RoundStart(Variant.TENHOU_4P, state.round(), state.honba(), state.sticks(), state.scores()));
      double[] direct = new double[4];
      network.phi(state, RankUtility.TENHOU, workspace, direct);
      double[] fromTable = new double[4];
      phi.values(table, fromTable);
      double[] fromState = new double[4];
      phi.values(state, fromState);
      assertBits(fromTable, direct, state);
      assertBits(fromState, direct, state);
    }
  }

  @Test
  public void memoHitsAndMissesGiveTheSameBits() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 16, false), 5, 1.0);
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    GrpWorkspace workspace = GrpWorkspace.of(network.shape());
    Random random = new Random(31);
    List<GrpState> states = new ArrayList<>();
    for (int trial = 0; trial < 10_000; trial++) {
      states.add(GrpFixtures.state(Variant.TENHOU_4P, random));
    }
    double[] direct = new double[4];
    double[] first = new double[4];
    double[] second = new double[4];
    for (GrpState state : states) {
      network.phi(state, RankUtility.TENHOU, workspace, direct);
      phi.values(state, first);
      phi.values(state, second);
      assertBits(first, direct, state);
      assertBits(second, direct, state);
    }
    PhiMemo memo = phi.memo();
    Assert.assertTrue(memo.hits() >= states.size(), memo.hits() + " hits");
    Assert.assertTrue(memo.misses() > memo.slots(), memo.misses() + " misses");
    Assert.assertEquals(memo.bypasses(), 0);
    // スロットは上書きされているので、もう一度引くと当たりと外れが混ざる。どちらでもビット単位で同じ。
    long hits = memo.hits();
    for (GrpState state : states) {
      network.phi(state, RankUtility.TENHOU, workspace, direct);
      phi.values(state, first);
      assertBits(first, direct, state);
    }
    Assert.assertTrue(memo.hits() > hits);
    Assert.assertEquals(memo.hits() + memo.misses(), 3L * states.size());
  }

  @Test
  public void statesOutsideTheKeyBypassTheMemo() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 16, false), 7, 1.0);
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    GrpState state = new GrpState(Variant.TENHOU_4P, 3, 64, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    double[] direct = new double[4];
    network.phi(state, RankUtility.TENHOU, GrpWorkspace.of(network.shape()), direct);
    double[] values = new double[4];
    phi.values(state, values);
    assertBits(values, direct, state);
    phi.values(state, values);
    assertBits(values, direct, state);
    Assert.assertEquals(phi.memo().bypasses(), 2);
    Assert.assertEquals(phi.memo().hits() + phi.memo().misses(), 0);
  }

  @Test
  public void threadsGiveBitIdenticalValues() throws Exception {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 32, true), 9, 1.0);
    Random random = new Random(37);
    List<GrpState> states = new ArrayList<>();
    for (int trial = 0; trial < 4_000; trial++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      states.add(
          trial % 10 == 0 ? new GrpState(Variant.TENHOU_4P, state.round(), 70, state.sticks(), state.scores()) : state);
    }
    double[][] expected = new double[states.size()][4];
    GrpWorkspace workspace = GrpWorkspace.of(network.shape());
    for (int index = 0; index < states.size(); index++) {
      network.phi(states.get(index), RankUtility.TENHOU, workspace, expected[index]);
    }
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    int threads = 8;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Future<double[][]>> results = new ArrayList<>();
      for (int thread = 0; thread < threads; thread++) {
        long seed = thread;
        results.add(pool.submit(() -> valuesInShuffledOrder(phi, states, seed)));
      }
      for (Future<double[][]> result : results) {
        double[][] values = result.get();
        for (int index = 0; index < states.size(); index++) {
          assertBits(values[index], expected[index], states.get(index));
        }
      }
    } finally {
      pool.shutdownNow();
    }
  }

  /** メモの当たりと外れ、半荘が続く卓（TableState と GrpState）、終局の卓のどれでも、ウォームアップの後はメモリを確保しない。 */
  @Test
  public void steadyValuesDoNotAllocate() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 32, true), 59, 1.0);
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    GrpState[] states = states(3 * phi.memo().slots(), 61);
    TableState start = new TableState(Variant.TENHOU_4P);
    start.set(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, new int[] {40_000, 20_000, 20_000, 20_000}));
    TableState finished = finish(start, new RoundOutcome(Variant.TENHOU_4P));
    TableState table = new TableState(Variant.TENHOU_4P);
    double[] values = new double[4];
    Allocations.assertSteady(
        4 * states.length + " values", 30, () -> valuesOfAll(phi, states, table, finished, values));
  }

  @Test
  public void eachPhiKeepsItsOwnMemo() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 16, false);
    GrpNetwork first = GrpFixtures.network(shape, 41, 1.0);
    GrpNetwork second = GrpFixtures.network(shape, 43, 1.0);
    GrpState state = new GrpState(Variant.TENHOU_4P, 6, 1, 0, new int[] {33_000, 27_000, 22_000, 18_000});
    double[] values = new double[4];
    Phi.of(first, RankUtility.TENHOU).values(state, values);
    double[] direct = new double[4];
    second.phi(state, RankUtility.TENHOU, GrpWorkspace.of(shape), direct);
    Phi.of(second, RankUtility.TENHOU).values(state, values);
    assertBits(values, direct, state);
  }

  @Test
  public void tablesMustMatchTheVariant() {
    Phi phi = Phi.of(GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 8, false), 53, 1.0), RankUtility.TENHOU);
    GrpState three = new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_000, 35_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> phi.values(three, new double[3]));
    GrpState four = new GrpState(Variant.TENHOU_4P, 0, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> phi.values(four, new double[3]));
  }

  /** GrpNetwork は履歴を読まない。履歴を渡しても渡さなくてもビット単位で同じで、メモも履歴に依らず当たる。 */
  @Test
  public void networkEntriesIgnoreTheHistory() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 16, false), 73, 1.0);
    Phi phi = Phi.of(network, RankUtility.TENHOU);
    Random random = new Random(79);
    GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
    double[] direct = new double[4];
    network.phi(state, RankUtility.TENHOU, GrpWorkspace.of(network.shape()), direct);
    double[] values = new double[4];
    phi.values(GrpFixtures.history(Variant.TENHOU_4P, random, 3), state, values);
    assertBits(values, direct, state);
    phi.values(GrpFixtures.history(Variant.TENHOU_4P, random, 5), state, values);
    assertBits(values, direct, state);
    Assert.assertEquals(phi.memo().hits(), 1);
  }

  /**
   * GruSinkhorn は履歴と卓を読み、モデルの値とビット単位で同じ値を返す。履歴を渡さないメソッドは半荘が続く卓を拒み、終局の卓では
   * ネットワークを呼ばずに厳密な効用を返す。
   */
  @Test
  public void gruEntriesReadTheHistory() {
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 8, 2), 83, 1.0);
    Phi phi = Phi.of(gru, RankUtility.TENHOU);
    GruWorkspace workspace = GruWorkspace.of(gru);
    TableState table = new TableState(Variant.TENHOU_4P);
    Random random = new Random(89);
    for (int trial = 0; trial < 200; trial++) {
      RoundHistory history = GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(10));
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      state.copyInto(table);
      double[] direct = new double[4];
      gru.phi(history, state, RankUtility.TENHOU, workspace, direct);
      double[] fromState = new double[4];
      phi.values(history, state, fromState);
      double[] fromTable = new double[4];
      phi.values(history, table, fromTable);
      assertBits(fromState, direct, state);
      assertBits(fromTable, direct, state);
    }
    GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
    state.copyInto(table);
    Assert.assertThrows(IllegalArgumentException.class, () -> phi.values(state, new double[4]));
    Assert.assertThrows(IllegalArgumentException.class, () -> phi.values(table, new double[4]));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> phi.values(new RoundHistory(Variant.TENHOU_4P), state, new double[4]));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> phi.values(GrpFixtures.history(Variant.TENHOU_3P, random, 2), state, new double[4]));

    Phi fresh = Phi.of(gru, RankUtility.TENHOU);
    double[] values = new double[4];
    fresh.values(GrpFixtures.finished(), values);
    Assert.assertEquals(values, new double[] {2.0 / 3, 1.0 / 3, 0, -1});
    fresh.values(GrpFixtures.history(Variant.TENHOU_4P, random, 2), GrpFixtures.finished(), values);
    Assert.assertEquals(values, new double[] {2.0 / 3, 1.0 / 3, 0, -1});
    assertUntouched(fresh.memo());
  }

  /** メモのキーは卓と履歴のバージョンの組。コピーした履歴（同じバージョン）は当たり、同じ卓でも別の履歴は外れる。当たっても外れてもビット単位で同じ。 */
  @Test
  public void gruMemoHitsAndMissesGiveTheSameBits() {
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 8, 2), 97, 1.0);
    Phi phi = Phi.of(gru, RankUtility.TENHOU);
    GruWorkspace workspace = GruWorkspace.of(gru);
    Random random = new Random(101);
    List<RoundHistory> histories = new ArrayList<>();
    for (int index = 0; index < 400; index++) {
      histories.add(GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(8)));
    }
    GrpState[] states = states(30, 103);
    double[] direct = new double[4];
    double[] first = new double[4];
    double[] second = new double[4];
    for (RoundHistory history : histories) {
      RoundHistory fork = history.copy();
      for (int index = 0; index < 4; index++) {
        GrpState state = states[random.nextInt(states.length)];
        gru.phi(history, state, RankUtility.TENHOU, workspace, direct);
        phi.values(history, state, first);
        phi.values(fork, state, second);
        assertBits(first, direct, state);
        assertBits(second, direct, state);
      }
    }
    PhiMemo memo = phi.memo();
    Assert.assertTrue(memo.hits() >= 4L * histories.size(), memo.hits() + " hits");
    Assert.assertTrue(memo.misses() >= histories.size(), memo.misses() + " misses");
    Assert.assertEquals(memo.hits() + memo.misses(), 8L * histories.size());
    Assert.assertEquals(memo.bypasses(), 0);
  }

  @Test
  public void gruThreadsGiveBitIdenticalValues() throws Exception {
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 16, 2), 107, 1.0);
    Random random = new Random(109);
    List<RoundHistory> histories = new ArrayList<>();
    List<GrpState> states = new ArrayList<>();
    for (int trial = 0; trial < 1_500; trial++) {
      histories.add(
          trial % 3 == 0 || histories.isEmpty()
              ? GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(10))
              : histories.get(histories.size() - 1));
      states.add(GrpFixtures.state(Variant.TENHOU_4P, random));
    }
    double[][] expected = new double[states.size()][4];
    GruWorkspace workspace = GruWorkspace.of(gru);
    for (int index = 0; index < states.size(); index++) {
      gru.phi(histories.get(index), states.get(index), RankUtility.TENHOU, workspace, expected[index]);
    }
    Phi phi = Phi.of(gru, RankUtility.TENHOU);
    int threads = 8;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Future<double[][]>> results = new ArrayList<>();
      for (int thread = 0; thread < threads; thread++) {
        long seed = thread;
        results.add(pool.submit(() -> gruValuesInShuffledOrder(phi, histories, states, seed)));
      }
      for (Future<double[][]> result : results) {
        double[][] values = result.get();
        for (int index = 0; index < states.size(); index++) {
          assertBits(values[index], expected[index], states.get(index));
        }
      }
    } finally {
      pool.shutdownNow();
    }
  }

  /**
   * GruSinkhorn でも、メモの当たりと外れ、状態のメモの当たりと外れ、表を重ねること、終局の卓のどれでも、ウォームアップの後は
   * メモリを確保しない。
   */
  @Test
  public void steadyGruValuesDoNotAllocate() {
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 8, 2), 113, 1.0);
    PhiTable phiTable = PhiTable.of(Variant.TENHOU_4P, new double[] {1, 1.2, 0.9}, -1, 2.0 / 3, new double[][] {{0.01, -0.02}});
    Phi phi = Phi.of(gru, RankUtility.TENHOU, phiTable);
    Random random = new Random(127);
    int slots = phi.memo().slots();
    RoundHistory[] histories = new RoundHistory[slots / 4];
    for (int index = 0; index < histories.length; index++) {
      histories[index] = GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(4));
    }
    GrpState[] states = states(8, 131);
    TableState table = new TableState(Variant.TENHOU_4P);
    TableState finished = GrpFixtures.finished();
    double[] values = new double[4];
    Allocations.assertSteady(histories.length * states.length + " GRU values", 5, () -> {
      for (RoundHistory history : histories) {
        for (GrpState state : states) {
          phi.values(history, state, values);
          state.copyInto(table);
          phi.values(history, table, values);
        }
        phi.values(history, finished, values);
      }
    });
  }

  /**
   * 表を重ねた GruSinkhorn は、周辺確率に次局（局末の卓の局）の温度を掛けて Φ にし、終わった局（履歴の最後の局）の補正を足す。
   * 温度が 1 の局では、補正の後も席の和を保つ。
   */
  @Test
  public void phiTablesOverlayTheGruEntries() {
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 8, 2), 137, 1.0);
    double[] temperatures = {1, 1, 1, 1, 1.4, 1.4, 0.7, 0.7, 1, 2};
    double[][] corrections = {{0.03, -0.01, 0.02}, {0, 0, 0}, {-0.05, 0.04, 0.01}, {0.02, 0.02, -0.06}};
    PhiTable phiTable = PhiTable.of(Variant.TENHOU_4P, temperatures, -1, 2.0 / 3, corrections);
    Phi phi = Phi.of(gru, RankUtility.TENHOU, phiTable);
    GruWorkspace workspace = GruWorkspace.of(gru);
    Random random = new Random(139);
    double[] marginals = new double[16];
    double[] expected = new double[4];
    double[] values = new double[4];
    for (int trial = 0; trial < 300; trial++) {
      RoundHistory history = GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(8));
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      gru.marginals(history, state, workspace, marginals);
      phiTable.temper(state.round(), marginals);
      for (int seat = 0; seat < 4; seat++) {
        expected[seat] = 0;
        for (int rank = 0; rank < 4; rank++) {
          expected[seat] += marginals[seat * 4 + rank] * RankUtility.TENHOU.value(Variant.TENHOU_4P, rank);
        }
      }
      phiTable.calibrate(history.round(history.size() - 1), expected);
      phi.values(history, state, values);
      assertBits(values, expected, state);
      if (temperatures[Math.min(state.round(), temperatures.length - 1)] == 1) {
        Assert.assertEquals(values[0] + values[1] + values[2] + values[3], 0.0, 1e-12, state.toString());
      }
    }
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Phi.of(gru, RankUtility.TENHOU, PhiTable.of(Variant.TENHOU_3P, new double[0], 0, 1, new double[0][])));
  }

  /** 点数の順位は局末の点の順位の効用をそのまま返し、周辺確率は 0 か 1。モデルを持たず、履歴も読まない。 */
  @Test
  public void scoreRankUsesTheScoreRanks() {
    Phi phi = Phi.scoreRank(Variant.TENHOU_4P, RankUtility.TENHOU);
    Assert.assertEquals(phi.kind(), Phi.Kind.SCORE_RANK);
    Assert.assertTrue(phi.kind().scoreOnly());
    Assert.assertFalse(phi.needsHistory());
    Assert.assertNull(phi.model());
    Assert.assertThrows(IllegalStateException.class, phi::networkWorkspace);
    Random random = new Random(149);
    double[] values = new double[4];
    double[] phiOut = new double[4];
    double[] marginals = new double[16];
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      phi.values(state, values);
      phi.evaluate(null, state, phiOut, marginals);
      for (int seat = 0; seat < 4; seat++) {
        Assert.assertEquals(
            values[seat], RankUtility.TENHOU.value(Variant.TENHOU_4P, state.rank(seat)), 1e-12, state.toString());
        for (int rank = 0; rank < 4; rank++) {
          Assert.assertEquals(marginals[seat * 4 + rank], rank == state.rank(seat) ? 1.0 : 0.0);
        }
      }
      assertBits(phiOut, values, state);
    }
  }

  /** 点数のなめらかな関数の周辺確率は分布で、Φ は周辺確率と効用の積和。evaluate と values はビットで同じ。 */
  @Test
  public void scoreSmoothValuesFollowTheMarginals() {
    Phi phi = Phi.scoreSmooth(Variant.TENHOU_4P, RankUtility.TENHOU);
    Assert.assertEquals(phi.kind(), Phi.Kind.SCORE_SMOOTH);
    Assert.assertTrue(phi.kind().scoreOnly());
    Assert.assertNull(phi.model());
    Random random = new Random(151);
    double[] values = new double[4];
    double[] phiOut = new double[4];
    double[] marginals = new double[16];
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      phi.values(state, values);
      phi.evaluate(null, state, phiOut, marginals);
      assertBits(phiOut, values, state);
      for (int seat = 0; seat < 4; seat++) {
        double sum = 0;
        double expected = 0;
        for (int rank = 0; rank < 4; rank++) {
          sum += marginals[seat * 4 + rank];
          expected += marginals[seat * 4 + rank] * RankUtility.TENHOU.value(Variant.TENHOU_4P, rank);
        }
        Assert.assertEquals(sum, 1, 1e-9);
        Assert.assertEquals(values[seat], expected, 1e-12);
      }
    }
  }

  /** evaluate の Φ は、ネットワーク・表を重ねた GruSinkhorn のどちらでも values とビットで同じ。 */
  @Test
  public void evaluateMatchesTheValueBits() {
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 16, true), 157, 1.0);
    Phi networkPhi = Phi.of(network, RankUtility.TENHOU);
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(Variant.TENHOU_4P, 8, 2), 163, 1.0);
    double[] temperatures = {1, 1, 1, 1, 1.4, 1.4, 0.7, 0.7, 1, 2};
    double[][] corrections = {{0.03, -0.01, 0.02}, {0, 0, 0}, {-0.05, 0.04, 0.01}, {0.02, 0.02, -0.06}};
    PhiTable phiTable = PhiTable.of(Variant.TENHOU_4P, temperatures, -1, 2.0 / 3, corrections);
    Phi gruPhi = Phi.of(gru, RankUtility.TENHOU, phiTable);
    GruWorkspace workspace = GruWorkspace.of(gru);
    Random random = new Random(167);
    double[] values = new double[4];
    double[] phiOut = new double[4];
    double[] marginals = new double[16];
    double[] expectedMarginals = new double[16];
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(Variant.TENHOU_4P, random);
      networkPhi.values(state, values);
      networkPhi.evaluate(null, state, phiOut, marginals);
      assertBits(phiOut, values, state);
      for (int seat = 0; seat < 4; seat++) {
        double expected = 0;
        for (int rank = 0; rank < 4; rank++) {
          expected += marginals[seat * 4 + rank] * RankUtility.TENHOU.value(Variant.TENHOU_4P, rank);
        }
        Assert.assertEquals(values[seat], expected, 1e-9);
      }

      RoundHistory history = GrpFixtures.history(Variant.TENHOU_4P, random, 1 + random.nextInt(8));
      gruPhi.values(history, state, values);
      gruPhi.evaluate(history, state, phiOut, marginals);
      assertBits(phiOut, values, state);
      gru.marginals(history, state, workspace, expectedMarginals);
      phiTable.temper(state.round(), expectedMarginals);
      assertBits(marginals, expectedMarginals, state);
    }
    Assert.assertEquals(networkPhi.stats().requests(), 200);
    Assert.assertEquals(gruPhi.stats().requests(), 200);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> networkPhi.evaluate(null, GrpFixtures.state(Variant.TENHOU_4P, random), phiOut, new double[4]));
  }

  /** 数えは全てのスレッドを足し合わせ、since は差を返す。 */
  @Test
  public void statsSumOverThreads() throws Exception {
    Phi phi = Phi.of(GrpFixtures.network(GrpFixtures.shape(Variant.TENHOU_4P, 8, false), 173, 1.0), RankUtility.TENHOU);
    GrpState[] states = states(50, 179);
    double[] values = new double[4];
    for (GrpState state : states) {
      phi.values(state, values);
    }
    Phi.Stats before = phi.stats();
    Assert.assertEquals(before.requests(), 50);
    Assert.assertEquals(before.hits() + before.misses() + before.bypasses(), 50);
    int threads = 4;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Future<?>> results = new ArrayList<>();
      for (int thread = 0; thread < threads; thread++) {
        results.add(pool.submit(() -> {
          double[] out = new double[4];
          for (int pass = 0; pass < 2; pass++) {
            for (GrpState state : states) {
              phi.values(state, out);
            }
          }
          phi.values(GrpFixtures.finished(), out);
        }));
      }
      for (Future<?> result : results) {
        result.get();
      }
    } finally {
      pool.shutdownNow();
    }
    Phi.Stats delta = phi.stats().since(before);
    Assert.assertEquals(delta.requests(), threads * 101L);
    Assert.assertEquals(delta.hits() + delta.misses() + delta.bypasses(), threads * 100L);
    Assert.assertTrue(delta.misses() >= threads * 50L, delta.toString());
    Assert.assertTrue(delta.hits() > 0, delta.toString());
    Assert.assertTrue(delta.nanos() > 0);
  }

  /**
   * スロットより多くの卓を引くので、メモの外れ（上書き）と当たりの両方を通る。半荘が続く卓は TableState にもコピーして、TableState を
   * 受け取る方でも引く。
   */
  private static void valuesOfAll(Phi phi, GrpState[] states, TableState table, TableState finished, double[] values) {
    for (GrpState state : states) {
      phi.values(state, values);
      state.copyInto(table);
      phi.values(table, values);
      phi.values(state, values);
      phi.values(finished, values);
    }
  }

  private static GrpState[] states(int count, long seed) {
    Random random = new Random(seed);
    GrpState[] states = new GrpState[count];
    for (int index = 0; index < count; index++) {
      states[index] = GrpFixtures.state(Variant.TENHOU_4P, random);
    }
    return states;
  }

  private static double[][] gruValuesInShuffledOrder(
      Phi phi, List<RoundHistory> histories, List<GrpState> states, long seed) {
    List<Integer> order = new ArrayList<>();
    for (int index = 0; index < states.size(); index++) {
      order.add(index);
    }
    Collections.shuffle(order, new Random(seed));
    double[][] values = new double[states.size()][4];
    for (int pass = 0; pass < 2; pass++) {
      for (int index : order) {
        phi.values(histories.get(index), states.get(index), values[index]);
      }
    }
    return values;
  }

  private static double[][] valuesInShuffledOrder(Phi phi, List<GrpState> states, long seed) {
    List<Integer> order = new ArrayList<>();
    for (int index = 0; index < states.size(); index++) {
      order.add(index);
    }
    Collections.shuffle(order, new Random(seed));
    double[][] values = new double[states.size()][4];
    for (int pass = 0; pass < 2; pass++) {
      for (int index : order) {
        phi.values(states.get(index), values[index]);
      }
    }
    return values;
  }

  private static TableState finish(TableState start, RoundOutcome outcome) {
    Variant variant = start.variant();
    int[] delta = new int[variant.seats()];
    Settlement.apply(variant, start, outcome, delta);
    TableState end = new TableState(variant);
    Progression.next(start, outcome, delta, end);
    Assert.assertTrue(end.gameOver(), end.toString());
    return end;
  }

  private static void assertUntouched(PhiMemo memo) {
    Assert.assertEquals(memo.hits() + memo.misses() + memo.bypasses(), 0);
  }

  private static void assertBits(double[] actual, double[] expected, GrpState state) {
    for (int seat = 0; seat < expected.length; seat++) {
      Assert.assertEquals(
          Double.doubleToRawLongBits(actual[seat]), Double.doubleToRawLongBits(expected[seat]), state + " seat " + seat);
    }
  }
}
