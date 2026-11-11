package com.epsilon.prism.learn;

import com.epsilon.prism.auxiliary.OutcomeCodec;
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
    Assert.assertThrows(IllegalArgumentException.class, () -> Teacher.anchored(values, new float[] {1f}, 0.5, new float[3]));
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
  public void teacherValuesKeepTheFormerBitsAndLeaveTheRowAlone() throws ReflectiveOperationException {
    Fixture fixture = new Fixture(11);
    for (Teacher teacher : fixture.teachers()) {
      for (int id = 0; id < ROWS; id++) {
        Row row = fixture.rows[id];
        float[] before = row.values().clone();
        float[] got = teacher.values(row, id);
        float[] expected = fixture.former(teacher, row, id);
        String where = "correction " + (teacher.correction() != null) + " anchor " + teacher.anchored() + " row " + id;
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
            "TEACHER allocation per row: now %.1f B, former %.1f B, saved %.1f B per ledger row (%d of %d),"
                + " %d actions (sink %d)",
            now,
            former,
            saved,
            ledgerRows,
            ROWS,
            ACTIONS,
            sink));
    double array = Float.BYTES * ACTIONS;
    Assert.assertTrue(saved >= array && saved <= array + 32, "saved " + saved + " now " + now + " former " + former);
  }

  private static void assertBitwise(float[] actual, float[] expected, String where) {
    Assert.assertEquals(actual.length, expected.length, where);
    for (int a = 0; a < expected.length; a++) {
      Assert.assertEquals(
          Float.floatToRawIntBits(actual[a]), Float.floatToRawIntBits(expected[a]), where + " action " + a);
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
      for (double[][] fold : beta) for (double[] c : fold) for (int f = 0; f < F; f++) c[f] = random.nextGaussian() * 0.3;
      correction = new LedgerCorrection(beta);
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
                measured ? Float.float16ToFloat(Float.floatToFloat16((float) random.nextGaussian())) : Float.NaN;
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
          AnchorScores.class.getDeclaredConstructor(int.class, int[].class, float[][].class, int[][].class);
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
      float[] values = teacher.correction() == null ? row.values() : teacher.correction().values(row);
      return teacher.anchored()
          ? Teacher.anchored(values, anchor.scores(id), teacher.anchorWeight(), new float[values.length])
          : values;
    }
  }
}
