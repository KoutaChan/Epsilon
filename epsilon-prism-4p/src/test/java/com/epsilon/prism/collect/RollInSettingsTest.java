package com.epsilon.prism.collect;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.collect.CollectorTest.Collected;
import com.epsilon.prism.collect.CollectorTest.HashedScorePolicy;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 根のクラスごとの中立ロールイン（labels.rollin と labels.copies）と、門前か副露かで分けた主経路のラベル行の率（noise.rates.open）が、設定
 * どおりに収集に効くことを確かめる。四人打ちの既定の率は、数で書いた既定値と全ての表で一致する。
 */
public class RollInSettingsTest {
  private static final Logger log = LoggerFactory.getLogger(RollInSettingsTest.class);

  /** 一様分布を混ぜず、探索もしない行動方策（ロールインの乱数が当たれば必ず根を作る）。 */
  private static final Behaviour GREEDY = new Behaviour(0, 0, CrnScheme.SINGLE_RACE);

  /** 一様分布を ε_b = 0.05 だけ混ぜる行動方策。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.SINGLE_RACE);

  /** 二項分布の許容幅を、標準偏差の何倍にするか。 */
  private static final double SIGMAS = 5;

  /**
   * 根のクラスごとの決まりを 1 つずつ変えた上書き。CALL は、主経路の全ての判断をラベル行にして（β の分母を数えるため）、β 0.3、一様な q
   * （ε_q 1）、ρ_br 0.5、上限なしにする。THREAT は、打った手のほかから d を引き、後の打牌の 3 クラスの ρ_br を 0、ほかを 1 にする。
   * DISCARD_MID は β 0.05、ρ_br 1、上限 1 にする。
   */
  private static final String[] PER_ROOT = {
    "noise.rates.table.CALL=1.0",
    "labels.rollin.CALL=0.3",
    "labels.copies.CALL.deviationFloor=1.0",
    "labels.copies.CALL.copyLabelRate=0.5",
    "labels.copies.CALL.maxCopyLabels=100",
    "labels.rollin.THREAT=0.1",
    "labels.copies.THREAT.deviation=OTHER",
    "labels.copies.THREAT.copyLabelRate=1.0",
    "labels.copies.THREAT.laterRates.DISCARD_EARLY=0",
    "labels.copies.THREAT.laterRates.DISCARD_MID=0",
    "labels.copies.THREAT.laterRates.DISCARD_LATE=0",
    "labels.copies.THREAT.maxCopyLabels=100",
    "labels.rollin.DISCARD_MID=0.05",
    "labels.copies.DISCARD_MID.copyLabelRate=1.0",
    "labels.copies.DISCARD_MID.maxCopyLabels=1"
  };

  /** 打牌の 3 クラス。 */
  private static final Set<DecisionClass> DISCARDS =
      EnumSet.of(DecisionClass.DISCARD_EARLY, DecisionClass.DISCARD_MID, DecisionClass.DISCARD_LATE);

  /** 副露している席の率を変えるクラス。 */
  private static final Set<DecisionClass> OPENED =
      EnumSet.of(DecisionClass.DISCARD_EARLY, DecisionClass.DISCARD_MID, DecisionClass.DISCARD_LATE, DecisionClass.THREAT);

  /** 四人打ちの判断のクラス（北抜きのクラスは四人打ちの設定の表に無く、判断にも現れない）。 */
  private static final Set<DecisionClass> CLASSES = ClassTables.classes(Variant.TENHOU_4P);

  private static Sampling sampling(String... assignments) {
    return Sampling.of(TestSettings.of(assignments));
  }

  /**
   * 番号 {@code first} から {@code first + games − 1} までの半荘を、一様分布を混ぜない行動方策と既定の帳簿で集める（進行のワーカー 2 本、
   * 同時に 4 半荘）。
   */
  private static Collected collect(Sampling sampling, long iterationSeed, long first, int games) {
    return CollectorTest.collectGames(
        sampling, CollectorTest.LEDGER, iterationSeed, first, games, GREEDY, new HashedScorePolicy(), 2, 4);
  }

  /** 2 つの抽出率で、四人打ちのクラスの全ての表がビットまで一致することを確かめる。 */
  static void assertSameRates(Sampling actual, Sampling expected) {
    assertBits(actual.holdoutFraction(), expected.holdoutFraction(), "holdout");
    Assert.assertEquals(actual.maxCandidates(), expected.maxCandidates());
    assertBits(actual.epsilonC(), expected.epsilonC(), "epsilonC");
    for (DecisionClass c : CLASSES) {
      Assert.assertEquals(actual.measuresAll(c), expected.measuresAll(c), c.name());
      for (Openness hand : Openness.values())
        assertBits(actual.labelRate(c, hand), expected.labelRate(c, hand), c + " " + hand);
      RollInRule a = actual.rollIn(c), e = expected.rollIn(c);
      assertBits(a.rate(), e.rate(), c + " rate");
      assertBits(a.epsilonQ(), e.epsilonQ(), c + " epsilonQ");
      Assert.assertEquals(a.target(), e.target(), c.name());
      Assert.assertEquals(a.maxLabels(), e.maxLabels(), c.name());
      for (DecisionClass later : CLASSES)
        assertBits(a.laterRate(later), e.laterRate(later), c + " later " + later);
    }
  }

  private static void assertBits(double actual, double expected, String label) {
    Assert.assertEquals(Double.doubleToLongBits(actual), Double.doubleToLongBits(expected), label + ": " + actual + " " + expected);
  }

  /** k が、n 回・率 p の二項分布の平均から {@link #SIGMAS} 標準偏差（に端の 1 を足した幅）の内にあることを確かめる。 */
  static void assertBinomial(long k, long n, double p, String label) {
    assertWithin(k, n * p, n * p * (1 - p), label + " (" + k + " of " + n + " at " + p + ")");
  }

  /** 数え上げた k が、平均 {@code mean} から {@link #SIGMAS} 標準偏差（分散は {@code variance}。端の 1 を足した幅）の内にあることを確かめる。 */
  static void assertWithin(long k, double mean, double variance, String label) {
    Assert.assertTrue(Math.abs(k - mean) <= SIGMAS * Math.sqrt(variance) + 1, label + ": mean " + mean);
  }

  private static long sum(Map<DecisionClass, Long> table) {
    long total = 0;
    if (table != null) for (long count : table.values()) total += count;
    return total;
  }

  private static long count(Map<DecisionClass, Long> table, DecisionClass c) {
    return table == null ? 0 : table.getOrDefault(c, 0L);
  }

  /** 根の行を、半荘と判断 ID のキーで引く表。 */
  private static Map<String, Row> roots(List<Row> rows) {
    Map<String, Row> roots = new HashMap<>();
    for (Row row : rows) if (row.source() == Row.Source.ROOT) roots.put(row.gameId() + ":" + row.decisionId(), row);
    return roots;
  }

  /** 写しの中のラベル行から、その根の行を引く。 */
  private static Row rootOf(Map<String, Row> roots, Row row) {
    Row root = roots.get(row.gameId() + ":" + row.rootDecisionId());
    Assert.assertNotNull(root, "root of " + row.gameId() + ":" + row.rootDecisionId());
    return root;
  }

  /** 四人打ちの既定の率は、まとめて決めた率（{@link TestSampling#grouped}）で書いた既定値と全ての表で一致する。 */
  @Test
  public void defaultsKeepTheirRates() {
    assertSameRates(sampling(), TestSampling.grouped(0.05, 0.013, 0.039, 0.0312, 0.104, 1.0, 6, true, 0.1, 0.002, 0.03, 0.1, 0.3, 2));
  }

  /**
   * 根のクラスごとの決まりが、そのクラスの根と写しにだけ効くことを確かめる（{@link #PER_ROOT}）。
   *
   * <ul>
   *   <li>CALL: 根の数は、主経路の CALL の判断（率 1 で全てラベル行）を回数とした β 0.3 の二項分布の幅に収まる。q は一様なので、傾向スコアの
   *       q は 1/|C|。d = played となる根の数は Σ 1/|C| の幅に収まる。写しの中のラベル行は、逸脱した席の判断を回数とした ρ_br 0.5 の
   *       二項分布の幅に収まる。
   *   <li>THREAT: 打った手のほかから d を引くので、全ての根が写しを作り（d = played が無い）、根の候補集合に打った手が無い。写しの中に後の
   *       打牌の 3 クラスの行は無い。
   *   <li>DISCARD_MID: 写しごとに 1 行まで。
   *   <li>ほかの根のクラスは、labels のセクションの決まり（上限 2）のまま。全ての根の傾向スコアは β·q(d)、ラベル行の傾向スコアは NaN。
   * </ul>
   */
  @Test(timeOut = 1_800_000)
  public void eachRootClassFollowsItsOwnRule() {
    Sampling sampling = sampling(PER_ROOT);
    Assert.assertEquals(sampling.rollIn(DecisionClass.THREAT).target(), DeviationTarget.OTHER);
    Assert.assertEquals(sampling.rollIn(DecisionClass.CALL).target(), DeviationTarget.ANY);
    Collected collected = collect(sampling, 31_007L, 400, 8);
    CollectionMetrics.Snapshot counts = collected.result().counts();
    CollectionMetrics.Breakdown breakdown = counts.breakdown();
    log.info("eachRootClassFollowsItsOwnRule: {} {}", collected.result().summary(), breakdown);
    assertBreakdownAddsUp(counts);
    Assert.assertEquals(counts.floorRoots(), 0);

    int call = DecisionClass.CALL.ordinal();
    long callLabels = counts.labelRows()[call];
    Assert.assertTrue(callLabels >= 200, "call decisions " + callLabels);
    assertBinomial(counts.roots()[call], callLabels, 0.3, "CALL roots");

    Map<String, Row> roots = roots(collected.rows());
    double sameMean = 0, sameVariance = 0;
    Map<String, Integer> perCopy = new HashMap<>();
    for (Row row : collected.rows()) {
      switch (row.source()) {
        case MAIN -> Assert.assertTrue(Float.isNaN(row.propensity()));
        case ROLLIN -> {
          Assert.assertTrue(Float.isNaN(row.propensity()));
          Row root = rootOf(roots, row);
          RollInRule rule = sampling.rollIn(root.decisionClass());
          String key = row.gameId() + ":" + row.rootDecisionId();
          Assert.assertTrue(perCopy.merge(key, 1, Integer::sum) <= rule.maxLabels(), key);
          if (root.decisionClass() == DecisionClass.THREAT)
            Assert.assertFalse(DISCARDS.contains(row.decisionClass()), "a discard row in a THREAT copy: " + key);
        }
        case ROOT -> {
          RollInRule rule = sampling.rollIn(row.decisionClass());
          int supportSize = Integer.bitCount(row.support());
          Assert.assertTrue((row.support() >>> row.deviation() & 1) != 0, "deviation outside the support");
          Assert.assertEquals(row.propensity(), (float) (rule.rate() * row.deviationProbability()));
          Assert.assertTrue(row.deviationProbability() >= rule.epsilonQ() / supportSize - 1e-6);
          switch (row.decisionClass()) {
            case CALL -> {
              Assert.assertEquals(row.deviationProbability(), (float) (1.0 / supportSize));
              double p = 1.0 / supportSize;
              sameMean += p;
              sameVariance += p * (1 - p);
            }
            case THREAT -> {
              Assert.assertNotEquals(row.deviation(), row.played());
              Assert.assertEquals(row.support() >>> row.played() & 1, 0, "the played action in an OTHER support");
            }
            default -> {}
          }
        }
      }
    }
    assertWithin(count(breakdown.sameDeviations(), DecisionClass.CALL), sameMean, sameVariance, "CALL same deviations");
    long callCopyLabels = sum(breakdown.copyLabels().get(DecisionClass.CALL));
    long callDeviators = count(breakdown.deviatorDecisions(), DecisionClass.CALL);
    Assert.assertTrue(callDeviators >= 200, "CALL deviator decisions " + callDeviators);
    assertBinomial(callCopyLabels, callDeviators, 0.5, "CALL copy labels");

    int threat = DecisionClass.THREAT.ordinal();
    Assert.assertTrue(counts.roots()[threat] > 0, "no THREAT roots");
    Assert.assertEquals(count(breakdown.sameDeviations(), DecisionClass.THREAT), 0);
    Assert.assertEquals(count(breakdown.copies(), DecisionClass.THREAT), counts.roots()[threat]);
    Map<DecisionClass, Long> threatLabels = breakdown.copyLabels().get(DecisionClass.THREAT);
    Assert.assertTrue(sum(threatLabels) > 0, "no labels in THREAT copies");
    for (DecisionClass c : DISCARDS) Assert.assertEquals(count(threatLabels, c), 0, c.name());

    long midCopies = count(breakdown.copies(), DecisionClass.DISCARD_MID);
    long midLabels = sum(breakdown.copyLabels().get(DecisionClass.DISCARD_MID));
    Assert.assertTrue(midLabels > 0 && midLabels <= midCopies, midLabels + " labels in " + midCopies + " copies");
  }

  /**
   * 内訳の和が配列の数と合うことを確かめる。門前と副露を合わせるとラベル行と判断の数に、根のクラスを合わせると写しの中のラベル行の数になる。
   * 写しの数と d = played の数の和は、根の数になる。
   */
  private static void assertBreakdownAddsUp(CollectionMetrics.Snapshot counts) {
    CollectionMetrics.Breakdown breakdown = counts.breakdown();
    long copies = 0, same = 0;
    for (DecisionClass c : DecisionClass.values()) {
      int i = c.ordinal();
      long labels = 0, decisions = 0, copyLabels = 0;
      for (Openness hand : Openness.values()) {
        labels += count(breakdown.labels().get(hand), c);
        decisions += count(breakdown.decisions().get(hand), c);
        Assert.assertTrue(count(breakdown.labels().get(hand), c) <= count(breakdown.decisions().get(hand), c), c + " " + hand);
      }
      for (DecisionClass root : DecisionClass.values()) copyLabels += count(breakdown.copyLabels().get(root), c);
      Assert.assertEquals(labels, counts.labelRows()[i], c.name());
      Assert.assertEquals(decisions, counts.mainDecisions()[i], c.name());
      Assert.assertEquals(copyLabels, counts.copyLabelRows()[i], c.name());
      Assert.assertEquals(count(breakdown.copies(), c) + count(breakdown.sameDeviations(), c), counts.roots()[i], c.name());
      Assert.assertTrue(
          sum(breakdown.copyLabels().get(c)) <= count(breakdown.deviatorDecisions(), c), "copy labels of root " + c);
      if (count(breakdown.copies(), c) == 0) Assert.assertEquals(count(breakdown.deviatorDecisions(), c), 0, c.name());
      copies += count(breakdown.copies(), c);
      same += count(breakdown.sameDeviations(), c);
    }
    Assert.assertEquals(copies, counts.copies());
    Assert.assertEquals(same, counts.sameDeviations());
  }

  /**
   * 根のクラス CALL の写しの決まりだけを変えると、根が CALL の写しの中のラベル行だけが変わることを確かめる。主経路のラベル行、根、ほかの
   * 根のクラスの写しの中のラベル行はビットまで一致する（写しの中のラベル行は、写しの打ち方を変えない）。
   */
  @Test(timeOut = 1_800_000)
  public void aRootRuleChangesOnlyTheRowsOfItsCopies() {
    Collected first = collect(sampling(PER_ROOT), 41_011L, 500, 4);
    Collected second = collect(sampling(TestSettings.concat(PER_ROOT, "labels.copies.CALL.copyLabelRate=0.1")), 41_011L, 500, 4);
    List<Row> kept = new ArrayList<>(), keptSecond = new ArrayList<>();
    long callFirst = 0, callSecond = 0;
    Map<String, Row> firstRoots = roots(first.rows());
    for (Row row : first.rows())
      if (row.source() == Row.Source.ROLLIN && rootOf(firstRoots, row).decisionClass() == DecisionClass.CALL) callFirst++;
      else kept.add(row);
    Map<String, Row> secondRoots = roots(second.rows());
    for (Row row : second.rows())
      if (row.source() == Row.Source.ROLLIN && rootOf(secondRoots, row).decisionClass() == DecisionClass.CALL) callSecond++;
      else keptSecond.add(row);
    Assert.assertTrue(callFirst > callSecond, callFirst + " then " + callSecond);
    Assert.assertTrue(kept.stream().anyMatch(row -> row.source() == Row.Source.ROLLIN), "no other copy rows");
    Assert.assertEquals(CollectorTest.encoded(keptSecond), CollectorTest.encoded(kept));
    Assert.assertEquals(
        second.result().counts().breakdown().deviatorDecisions(), first.result().counts().breakdown().deviatorDecisions());
  }

  /**
   * 副露している席のラベル行の率（noise.rates.open）を確かめる。打牌の 3 クラスと THREAT の率を 1・0.5・0 にした 3 つの収集（ロールイン
   * なし）を比べる。主経路の判断の数と、門前の席のラベル行は 3 つで一致する。副露している席のラベル行は、率 0.5 では率 1 の数を回数とした
   * 二項分布の幅に収まり、率 1 の行の部分集合になる。率 0 では無い（ほかのクラスは率の表のまま）。ラベル行の傾向スコアは NaN。
   */
  @Test(timeOut = 1_800_000)
  public void openRatesApplyOnlyToOpenSeats() {
    List<String> noRollIn = new ArrayList<>();
    for (DecisionClass c : CLASSES) noRollIn.add("labels.rollin." + c + "=0");
    List<Collected> collected = new ArrayList<>();
    for (String rate : List.of("1", "0.5", "0")) {
      List<String> assignments = new ArrayList<>(noRollIn);
      for (DecisionClass c : OPENED) assignments.add("noise.rates.open." + c + "=" + rate);
      Sampling sampling = sampling(assignments.toArray(String[]::new));
      for (DecisionClass c : OPENED)
        assertBits(sampling.labelRate(c, Openness.OPEN), Double.parseDouble(rate), c + " open");
      collected.add(collect(sampling, 51_013L, 600, 8));
    }
    CollectionMetrics.Breakdown all = collected.get(0).result().counts().breakdown();
    log.info("openRatesApplyOnlyToOpenSeats: {}", all);
    long open = 0;
    for (DecisionClass c : OPENED) open += count(all.labels().get(Openness.OPEN), c);
    Assert.assertTrue(open >= 100, "open labels " + open);
    for (int i = 1; i < collected.size(); i++) {
      CollectionMetrics.Breakdown other = collected.get(i).result().counts().breakdown();
      Assert.assertEquals(other.decisions(), all.decisions());
      Assert.assertEquals(other.labels().get(Openness.CLOSED), all.labels().get(Openness.CLOSED));
      for (DecisionClass c : DecisionClass.values())
        if (!OPENED.contains(c))
          Assert.assertEquals(count(other.labels().get(Openness.OPEN), c), count(all.labels().get(Openness.OPEN), c), c.name());
      Set<String> superset = new HashSet<>(CollectorTest.encoded(collected.get(i - 1).rows()));
      for (String row : CollectorTest.encoded(collected.get(i).rows()))
        Assert.assertTrue(superset.contains(row), "a row of rate " + i + " is not a row of the higher rate");
    }
    long half = 0;
    for (DecisionClass c : OPENED) half += count(collected.get(1).result().counts().breakdown().labels().get(Openness.OPEN), c);
    assertBinomial(half, open, 0.5, "open labels at 0.5");
    for (DecisionClass c : OPENED)
      Assert.assertEquals(count(collected.get(2).result().counts().breakdown().labels().get(Openness.OPEN), c), 0, c.name());
    for (Collected each : collected) {
      Assert.assertEquals(each.result().counts().rootCount(), 0);
      for (Row row : each.rows()) Assert.assertTrue(Float.isNaN(row.propensity()));
    }
  }

  /** 根のクラスごとの決まりと副露の率を変えた収集も、実行の順・同時に進める半荘の数・ワーカーの数に依らない（行・配列・内訳が一致する）。 */
  @Test(timeOut = 1_800_000)
  public void configuredRollInDoesNotDependOnScheduling() {
    Sampling sampling =
        sampling(TestSettings.concat(PER_ROOT, "noise.rates.open.DISCARD_MID=0.2", "noise.rates.open.THREAT=0.5"));
    Collected serial =
        CollectorTest.collectGames(sampling, CollectorTest.LEDGER, 61_017L, 700, 4, FLOOR, new HashedScorePolicy(), 1, 4);
    Collected parallel =
        CollectorTest.collectGames(sampling, CollectorTest.LEDGER, 61_017L, 700, 4, FLOOR, new HashedScorePolicy(), 4, 2);
    CollectionMetrics.Snapshot a = serial.result().counts(), b = parallel.result().counts();
    Assert.assertTrue(a.copies() > 0 && sum(a.labelRows()) > 0 && a.breakdown().openLabels() > 0, a.toString());
    Assert.assertEquals(CollectorTest.encoded(parallel.rows()), CollectorTest.encoded(serial.rows()));
    Assert.assertEquals(b.breakdown(), a.breakdown());
    Assert.assertEquals(Arrays.toString(b.labelRows()), Arrays.toString(a.labelRows()));
    Assert.assertEquals(Arrays.toString(b.copyLabelRows()), Arrays.toString(a.copyLabelRows()));
    Assert.assertEquals(Arrays.toString(b.roots()), Arrays.toString(a.roots()));
    Assert.assertEquals(Arrays.toString(b.mainDecisions()), Arrays.toString(a.mainDecisions()));
    Assert.assertEquals(b.copyDecisions(), a.copyDecisions());
    Assert.assertEquals(b.forkDecisions(), a.forkDecisions());
  }

  private static long sum(long[] counts) {
    long total = 0;
    for (long count : counts) total += count;
    return total;
  }
}
