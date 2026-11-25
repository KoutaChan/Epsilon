package com.epsilon.prism.learn;

import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.AgariOrder;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.rows.Row;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Constructor;
import java.util.Arrays;
import java.util.Locale;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 教師の値（{@link Teacher#values}）のテスト。アンカーを混ぜた目標は、測った手では (1 − w)·Y + w·s_アンカー で、測っていない手は NaN の
 * まま。重みが 0 なら Y そのもの。長さが違えば IllegalArgumentException を投げる。補正とアンカーの 4 つの組み合わせのどれでも、教師の値は
 * 配列を 2 つ確保するやり方（補正した配列と、混ぜた別の配列）とビット単位で一致し、行の配列は書き換えない。補正とアンカーの両方がある行では、
 * そのやり方より配列を 1 つ少なく確保する。
 */
public class TeacherTest {
  private static final Logger log = LoggerFactory.getLogger(TeacherTest.class);

  private static final int F = LedgerColumn.COUNT;

  /** 行の合法手の数。 */
  private static final int ACTIONS = 24;

  /** 行の数（アンカーの s が 1 つの区間に収まる数）。 */
  private static final int ROWS = 512;

  private static final double WEIGHT = 0.3;

  @Test
  public void anchorMixesMeasuredActionsOnly() {
    float[] values = {0.1f, Float.NaN, 0.3f};
    float[] anchor = {1f, 2f, 3f};
    float[] mixed = Teacher.anchored(values, anchor, 0.5, new float[3]);
    Assert.assertEquals(mixed[0], 0.55f, 1e-6f);
    Assert.assertTrue(Float.isNaN(mixed[1]));
    Assert.assertEquals(mixed[2], 1.65f, 1e-6f);
    Assert.assertEquals(Teacher.anchored(values, anchor, 0, new float[3])[2], 0.3f, 0f);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> Teacher.anchored(values, new float[] {1f}, 0.5, new float[3]));
  }

  /** 書き込む先が Y の配列そのものでも、別の配列に書いたときとビット単位で同じ値になる。 */
  @Test
  public void anchorMayMixIntoTheValuesItReads() {
    var random = new SplittableRandom(3);
    for (int trial = 0; trial < 1_000; trial++) {
      float[] values = new float[ACTIONS], anchor = new float[ACTIONS];
      for (int a = 0; a < ACTIONS; a++) {
        values[a] = random.nextInt(5) == 0 ? Float.NaN : (float) random.nextGaussian();
        anchor[a] = (float) random.nextGaussian();
      }
      float[] apart = Teacher.anchored(values, anchor, WEIGHT, new float[ACTIONS]);
      float[] inPlace = values.clone();
      Assert.assertSame(Teacher.anchored(inPlace, anchor, WEIGHT, inPlace), inPlace);
      assertBitwise(inPlace, apart, "trial " + trial);
    }
  }

  /**
   * 補正なし・補正だけ・アンカーだけ・両方の教師で、帳簿のある行でも無い行でも、教師の値は配列を 2 つ確保するやり方の値とビット単位で一致する。
   * 行の配列は変わらない。アンカーを混ぜた値と、帳簿のある行で補正した値は、行の配列とは別の配列で返る。
   */
  @Test
  public void teacherValuesKeepTheFormerBitsAndLeaveTheRowAlone()
      throws ReflectiveOperationException {
    Fixture fixture = new Fixture(11);
    for (Teacher teacher : fixture.teachers()) {
      for (int id = 0; id < ROWS; id++) {
        Row row = fixture.rows[id];
        float[] before = row.values().clone();
        float[] got = teacher.values(row, id);
        float[] expected = fixture.former(teacher, row, id);
        String where =
            "correction "
                + (teacher.correction() != null)
                + " anchor "
                + teacher.anchored()
                + " row "
                + id;
        assertBitwise(got, expected, where);
        assertBitwise(row.values(), before, where + " row values");
        boolean ledger = row.ledgerFeatures() == F;
        if (teacher.anchored() || (teacher.correction() != null && ledger)) {
          Assert.assertNotSame(got, row.values(), where);
        } else {
          Assert.assertSame(got, row.values(), where);
        }
      }
    }
  }

  /**
   * 補正とアンカーの両方がある教師（帳簿の補正を使う系譜の教師）では、帳簿のある行の教師の値は、配列を 2 つ確保するやり方より、行あたり
   * ちょうど配列 1 つ分（float × 合法手の数と、配列のヘッダー）少なく確保する。確保の量は ThreadMXBean で数え、JIT のウォームアップの後の
   * 3 回の最小を比べる。
   */
  @Test
  public void aCorrectedAndAnchoredRowAllocatesOneArrayLess() throws ReflectiveOperationException {
    ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    if (!threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Thread allocation counting is not available");
    }
    Fixture fixture = new Fixture(29);
    Teacher teacher = new Teacher(fixture.correction, fixture.anchor, WEIGHT, 1f);
    long[] best = {Long.MAX_VALUE, Long.MAX_VALUE};
    int sink = 0;
    for (int round = 0; round < 300; round++) {
      boolean measure = round >= 297;
      for (int path = 0; path < 2; path++) {
        long before = threads.getCurrentThreadAllocatedBytes();
        for (int id = 0; id < ROWS; id++) {
          Row row = fixture.rows[id];
          float[] values = path == 0 ? teacher.values(row, id) : fixture.former(teacher, row, id);
          sink += Float.floatToRawIntBits(values[id % ACTIONS]);
        }
        long allocated = threads.getCurrentThreadAllocatedBytes() - before;
        if (measure) best[path] = Math.min(best[path], allocated);
      }
    }
    // 帳簿の無い行では補正が行の配列を返すので、2 つのやり方の確保は同じ。差は帳簿のある行の数で割る
    int ledgerRows = fixture.ledgerRows();
    double now = (double) best[0] / ROWS, former = (double) best[1] / ROWS;
    double saved = (double) (best[1] - best[0]) / ledgerRows;
    log.info(
        String.format(
            Locale.ROOT,
            "TEACHER allocation per row: now %.1f B, former %.1f B, saved %.1f B per ledger row (%d"
                + " of %d), %d actions (sink %d)",
            now,
            former,
            saved,
            ledgerRows,
            ROWS,
            ACTIONS,
            sink));
    double array = Float.BYTES * ACTIONS;
    Assert.assertTrue(
        saved >= array && saved <= array + 32,
        "saved " + saved + " now " + now + " former " + former);
  }

  /** 星型 DAG の閉形式射影（{@link AgariOrder#project}）が、全部分集合を調べる総当たりの最小二乗解と一致することを確かめる。 */
  @Test
  public void starProjectionMatchesBruteForceQp() {
    var random = new SplittableRandom(73);
    for (int trial = 0; trial < 500; trial++) {
      int n = 2 + random.nextInt(8);
      int win = random.nextInt(n);
      boolean top = (trial & 1) == 0;
      float[] initial = new float[n];
      for (int a = 0; a < n; a++) {
        initial[a] =
            a != win && random.nextInt(4) == 0 ? Float.NaN : (float) (random.nextGaussian() * 0.5);
      }
      if (trial % 25 == 0) {
        initial[win] = Float.NaN;
      }
      float[] closed = initial.clone();
      boolean changed = AgariOrder.project(closed, win, top);
      Assert.assertFalse(AgariOrder.violates(closed, win, top), "trial " + trial);
      if (Float.isNaN(initial[win])) {
        Assert.assertFalse(changed);
        assertBitwise(closed, initial, "nan win trial " + trial);
        continue;
      }
      double[] expected = bruteForceProject(initial, win, top);
      boolean anyViolator = AgariOrder.violates(initial, win, top);
      Assert.assertEquals(changed, anyViolator, "changed trial " + trial);
      if (!anyViolator) {
        assertBitwise(closed, initial, "feasible trial " + trial);
      } else {
        for (int a = 0; a < n; a++) {
          if (Float.isNaN(initial[a])) {
            Assert.assertTrue(Float.isNaN(closed[a]));
          } else {
            Assert.assertEquals(
                closed[a], (float) expected[a], 1e-6f, "trial " + trial + " a=" + a);
          }
        }
      }
    }
  }

  /**
   * 証明辺のある行の教師: 終局和了の枝は帳簿補正 C_win = 0（生のラベルそのまま、ダブロンの結果も含めて置き換えない）になり、アンカー混合の後に
   * 証明辺へ 1 回だけ射影する。証明辺の無い行（NONE / CERT_MID）と LEGACY モードは従来とビット同一で、元の行の配列は書き換えない。
   */
  @Test
  public void certifiedTeacherAppliesZeroWinCorrectionRawLabelAndSingleProjection()
      throws ReflectiveOperationException {
    Fixture fixture = new Fixture(41);
    Teacher legacy = new Teacher(fixture.correction, fixture.anchor, WEIGHT, 1f);
    Teacher certified = legacy.withAgariOrder(AgariOrder.Mode.CERTIFIED);

    Row row = fixture.rows[1];
    float[] before = row.values().clone();
    Assert.assertThrows(IllegalStateException.class, () -> certified.values(row, 1));

    // LEGACY は証明結果が何であっても従来とビット同一
    var topCert = new AgariOrder.Certification(0, AgariOrder.Kind.CERT_TOP, 0, 0, false);
    assertBitwise(legacy.values(row, 1, topCert), legacy.values(row, 1), "legacy ignores cert");

    // CERTIFIED でも証明辺の無い行（NONE / CERT_MID）は LEGACY とビット同一
    var midCert = new AgariOrder.Certification(0, AgariOrder.Kind.CERT_MID, 1, 1, true);
    assertBitwise(
        certified.values(row, 1, AgariOrder.Certification.NONE),
        legacy.values(row, 1),
        "certified NONE");
    assertBitwise(certified.values(row, 1, midCert), legacy.values(row, 1), "certified CERT_MID");

    // CERTIFIED + CERT_TOP / CERT_BOTTOM: C_win = 0（raw[0] を保持）, Tempo 混合, 1 回射影
    for (var cert :
        new AgariOrder.Certification[] {
          topCert, new AgariOrder.Certification(0, AgariOrder.Kind.CERT_BOTTOM, 3, 3, true)
        }) {
      float[] expected = fixture.correction.values(row);
      expected[0] = row.values()[0];
      Teacher.anchored(expected, fixture.anchor.scores(1), WEIGHT, expected);
      AgariOrder.project(expected, 0, cert.top());
      float[] got = certified.values(row, 1, cert);
      assertBitwise(got, expected, cert.kind().name());
      Assert.assertFalse(AgariOrder.violates(got, 0, cert.top()));
      assertBitwise(row.values(), before, "row values unchanged");
    }

    // 補正もアンカーも無い RAW 教師でも row.values() を書き換えない
    Teacher rawCertified = Teacher.RAW.withAgariOrder(AgariOrder.Mode.CERTIFIED);
    float[] gotRaw = rawCertified.values(row, 1, topCert);
    Assert.assertFalse(AgariOrder.violates(gotRaw, 0, true));
    assertBitwise(row.values(), before, "raw certified leaves row values alone");
  }

  /** 全ての部分集合 S ⊆ {測った他手} を試し、和了と S を平均で揃えた候補から実行可能かつ二乗距離最小の解を求める。 */
  private static double[] bruteForceProject(float[] initial, int win, boolean top) {
    int n = initial.length;
    int[] others = new int[n];
    int m = 0;
    for (int a = 0; a < n; a++) {
      if (a != win && !Float.isNaN(initial[a])) {
        others[m++] = a;
      }
    }
    double bestDist = Double.POSITIVE_INFINITY;
    double[] best = null;
    for (int mask = 0; mask < (1 << m); mask++) {
      double sum = initial[win];
      int count = 1;
      for (int i = 0; i < m; i++) {
        if ((mask & (1 << i)) != 0) {
          sum += initial[others[i]];
          count++;
        }
      }
      double mu = sum / count;
      boolean feasible = true;
      for (int i = 0; i < m; i++) {
        if ((mask & (1 << i)) == 0) {
          double v = initial[others[i]];
          if (top ? v > mu + 1e-12 : v < mu - 1e-12) {
            feasible = false;
            break;
          }
        }
      }
      if (!feasible) {
        continue;
      }
      double dist = (mu - initial[win]) * (mu - initial[win]);
      for (int i = 0; i < m; i++) {
        if ((mask & (1 << i)) != 0) {
          double d = mu - initial[others[i]];
          dist += d * d;
        }
      }
      if (dist < bestDist) {
        bestDist = dist;
        best = new double[n];
        for (int a = 0; a < n; a++) {
          best[a] = initial[a];
        }
        best[win] = mu;
        for (int i = 0; i < m; i++) {
          if ((mask & (1 << i)) != 0) {
            best[others[i]] = mu;
          }
        }
      }
    }
    Assert.assertNotNull(best);
    return best;
  }

  private static void assertBitwise(float[] actual, float[] expected, String where) {
    Assert.assertEquals(actual.length, expected.length, where);
    for (int a = 0; a < expected.length; a++) {
      Assert.assertEquals(
          Float.floatToRawIntBits(actual[a]),
          Float.floatToRawIntBits(expected[a]),
          where + " action " + a);
    }
  }

  /** テスト用に作った行（3 行に 1 行は帳簿なし）と、補正の係数とアンカーの s。 */
  private static final class Fixture {
    final Row[] rows = new Row[ROWS];
    final LedgerCorrection correction;
    final AnchorScores anchor;

    Fixture(long seed) throws ReflectiveOperationException {
      var random = new SplittableRandom(seed);
      double[][][] beta = new double[2][DecisionClass.COUNT][F];
      for (double[][] fold : beta)
        for (double[] c : fold) for (int f = 0; f < F; f++) c[f] = random.nextGaussian() * 0.3;
      correction = new LedgerCorrection.ByClass(beta);
      float[] flat = new float[ROWS * ACTIONS];
      int[] ids = new int[ROWS], starts = new int[ROWS + 1];
      DecisionClass[] classes = DecisionClass.values();
      for (int id = 0; id < ROWS; id++) {
        ids[id] = id;
        starts[id + 1] = starts[id] + ACTIONS;
        boolean ledger = id % 3 != 0;
        float[] values = new float[ACTIONS];
        float[] terms = new float[ledger ? ACTIONS * F : 0];
        for (int a = 0; a < ACTIONS; a++) {
          boolean measured = a % 5 != 4;
          values[a] = measured ? (float) random.nextGaussian() : Float.NaN;
          for (int f = 0; ledger && f < F; f++) {
            terms[a * F + f] =
                measured
                    ? Float.float16ToFloat(Float.floatToFloat16((float) random.nextGaussian()))
                    : Float.NaN;
          }
          flat[id * ACTIONS + a] = (float) random.nextGaussian();
        }
        float[] prior = new float[ACTIONS];
        Arrays.fill(prior, 1f / ACTIONS);
        rows[id] =
            new Row(
                1_000L + id / 4,
                id,
                classes[id % classes.length],
                Row.Purpose.TRAIN,
                Row.Source.MAIN,
                0,
                values,
                prior,
                0,
                -1,
                Float.NaN,
                Float.NaN,
                -1L,
                terms,
                new long[ACTIONS * OutcomeCodec.WORDS],
                new long[0],
                new long[0]);
      }
      Constructor<AnchorScores> make =
          AnchorScores.class.getDeclaredConstructor(
              int.class, int[].class, float[][].class, int[][].class);
      make.setAccessible(true);
      anchor = make.newInstance(1, ids, new float[][] {flat}, new int[][] {starts});
    }

    int ledgerRows() {
      int count = 0;
      for (Row row : rows) if (row.ledgerFeatures() == F) count++;
      return count;
    }

    Teacher[] teachers() {
      return new Teacher[] {
        Teacher.RAW,
        new Teacher(correction, null, 0, 1f),
        new Teacher(null, anchor, WEIGHT, 1f),
        new Teacher(correction, anchor, WEIGHT, 1f)
      };
    }

    /** 配列を 2 つ確保するやり方。補正した配列（補正が無ければ行の配列）に、アンカーがあれば別の新しい配列の上で混ぜる。 */
    float[] former(Teacher teacher, Row row, int id) {
      float[] values =
          teacher.correction() == null ? row.values() : teacher.correction().values(row);
      return teacher.anchored()
          ? Teacher.anchored(
              values, anchor.scores(id), teacher.anchorWeight(), new float[values.length])
          : values;
    }
  }
}
