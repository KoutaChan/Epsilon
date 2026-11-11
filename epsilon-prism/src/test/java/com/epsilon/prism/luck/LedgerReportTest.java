package com.epsilon.prism.luck;

import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.core.DecisionClass;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 学習の段が書く帳簿の報告を確かめる。ledger.json に書く形（{@link LedgerReport.Full}）は当てはめの報告の値をそのまま持ち、JSON に
 * 書いて読み戻すと同じ record になる。有限でない値は書かず、そのときの JSON も読み戻せる。段の報告に載せる要約
 * （{@link LedgerReport.Summary}）は、偏りのある列の名前を、z がしきい値を越えたときだけ載せる。
 */
public class LedgerReportTest {
  private static final int F = LedgerColumn.COUNT;

  @Test
  public void theFullReportKeepsTheReportValues() {
    var rows = LedgerFitTest.rows(6000, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 40);
    LedgerReport report = LedgerFitTest.CLASSIC.fitRows(rows).report();
    LedgerReport.Full full = report.full();
    Assert.assertEquals(full.rows(), report.rows());
    Assert.assertEquals(full.slots(), report.slots());
    Assert.assertEquals(full.skippedRows(), report.skippedRows());
    Assert.assertEquals(full.seconds(), report.seconds());
    assertSegment(full.all(), report.all(), "all");
    Assert.assertEquals(List.copyOf(full.byClass().keySet()), List.copyOf(report.byClass().keySet()));
    report.byClass().forEach((name, line) -> assertSegment(full.byClass().get(name), line, name));
    Assert.assertEquals(full.bias(), report.bias());
    Assert.assertEquals(full.bias().all().size(), F);
    Assert.assertEquals(roundTrip(full, LedgerReport.Full.class), full);
  }

  /**
   * 行が 1 つの半荘のものだけで、どの行も測った手の Y が同じ場合。行の中で中心化した Y の平方の和が 0 なので、R²・同等の分岐の本数・
   * 偶奇ごとの R²・組ごとの R² は有限でない。束が 1 つなので z も有限でない。ledger.json の形と要約はそれらの値を書かない。JSON に NaN も
   * Infinity も入らず、読み戻すと同じ record になる。
   */
  @Test
  public void nonFiniteValuesAreLeftOut() {
    var random = new SplittableRandom(42);
    List<LedgerFitTest.SyntheticRow> rows = new ArrayList<>();
    for (int i = 0; i < 40; i++) {
      float[] values = {0.5f, 0.5f, 0.5f, Float.NaN};
      float[] ledger = new float[4 * F];
      Arrays.fill(ledger, Float.NaN);
      for (int k = 0; k < 3 * F; k++) ledger[k] = (float) random.nextGaussian();
      rows.add(new LedgerFitTest.SyntheticRow(7, DecisionClass.CALL, values, ledger));
    }
    LedgerReport report = LedgerFitTest.CLASSIC.fitRows(rows).report();
    Assert.assertEquals(report.rows(), 40L);
    Assert.assertTrue(Double.isNaN(report.all().r2()));
    Assert.assertTrue(Double.isNaN(report.all().equivalentForks()));
    LedgerReport.Full full = report.full();
    Assert.assertNull(full.all().r2());
    Assert.assertNull(full.all().equivalentForks());
    Assert.assertTrue(full.all().r2ByGroup().isEmpty(), full.all().r2ByGroup().toString());
    for (LedgerReport.Fold fold : full.all().folds()) Assert.assertNull(fold.r2());
    for (LedgerReport.FeatureBias bias : full.bias().all()) {
      Assert.assertNull(bias.z(), bias.name());
      Assert.assertNotNull(bias.mean(), bias.name());
    }
    String json = RecordFile.json(full);
    Assert.assertFalse(json.contains("NaN") || json.contains("Infinity"), json);
    Assert.assertEquals(roundTrip(full, LedgerReport.Full.class), full);

    LedgerReport.Summary summary = LedgerReport.Summary.of(report, 3.0, false);
    Assert.assertNull(summary.r2());
    Assert.assertNull(summary.maxAbsZ());
    Assert.assertTrue(summary.biased().isEmpty());
    Assert.assertEquals(roundTrip(summary, LedgerReport.Summary.class), summary);
  }

  /**
   * 1 つの列（自分のツモ）にだけ平均 0.1 を足すと、要約はその列の名前だけを偏りとして載せ、z の絶対値の最大はその列の z になる。
   * しきい値を上げると名前は載らないが、最大はそのまま。足さない元の行では、z がしきい値 4.5 を越える列は無い。
   */
  @Test
  public void aBiasedColumnRaisesTheFlag() {
    var rows = LedgerFitTest.rows(6000, 0, LedgerFitTest.ones(), new double[F], 0.3, 41);
    int column = LedgerColumn.OWN_TSUMO.ordinal();
    List<LedgerFitTest.SyntheticRow> shifted = new ArrayList<>();
    for (var row : rows) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++) if (!Float.isNaN(row.values()[a])) ledger[a * F + column] += 0.1f;
      shifted.add(LedgerFitTest.withValues(row, row.values(), ledger));
    }
    LedgerReport biased = LedgerFitTest.CLASSIC.fitRows(shifted).report();
    double z = biased.bias().all().get(column).z();
    Assert.assertTrue(z > 4.5, "z " + z);

    LedgerReport.Summary flagged = LedgerReport.Summary.of(biased, 4.5, true);
    Assert.assertEquals(flagged.biased(), List.of(LedgerColumn.OWN_TSUMO.reportName()));
    same(flagged.maxAbsZ(), Math.abs(z), "max |z|");
    same(flagged.r2(), biased.all().r2(), "r2");
    Assert.assertEquals(flagged.r2ByGroup(), biased.all().r2ByGroup());
    Assert.assertTrue(flagged.streamed());
    Assert.assertEquals(roundTrip(flagged, LedgerReport.Summary.class), flagged);

    LedgerReport.Summary quiet = LedgerReport.Summary.of(biased, 1000, true);
    Assert.assertTrue(quiet.biased().isEmpty(), quiet.biased().toString());
    same(quiet.maxAbsZ(), Math.abs(z), "max |z|");

    LedgerReport.Summary unbiased = LedgerReport.Summary.of(LedgerFitTest.CLASSIC.fitRows(rows).report(), 4.5, true);
    Assert.assertTrue(unbiased.biased().isEmpty(), unbiased.biased().toString());
    Assert.assertTrue(unbiased.maxAbsZ() < 4.5, "max |z| " + unbiased.maxAbsZ());
  }

  /** 書く形の区分が、当てはめの区分の値をそのまま持つことを確かめる。どの値も有限な区分で使う。 */
  private static void assertSegment(LedgerReport.Segment segment, LedgerReport.Line line, String label) {
    Assert.assertEquals(segment.rows(), line.rows(), label);
    Assert.assertEquals(segment.slots(), line.slots(), label);
    same(segment.variance(), line.variance(), label + " variance");
    same(segment.residual(), line.residual(), label + " residual");
    same(segment.r2(), line.r2(), label + " r2");
    same(segment.equivalentForks(), line.equivalentForks(), label + " forks");
    Assert.assertEquals(segment.r2ByGroup(), line.r2ByGroup(), label);
    List<String> names = Arrays.stream(LedgerColumn.values()).map(LedgerColumn::reportName).toList();
    Assert.assertEquals(List.copyOf(segment.beta().keySet()), names, label);
    Assert.assertEquals(segment.folds().size(), 2, label);
    for (int f = 0; f < F; f++) {
      same(segment.beta().get(names.get(f)), line.beta()[f], label + " " + names.get(f));
      for (int fold = 0; fold < 2; fold++)
        same(segment.folds().get(fold).beta().get(names.get(f)), line.betaByFold()[fold][f], label + " fold " + fold);
    }
    for (int fold = 0; fold < 2; fold++) {
      same(segment.folds().get(fold).r2(), line.r2ByFold()[fold], label + " fold " + fold);
      Assert.assertEquals(segment.folds().get(fold).fallback(), line.fallback()[fold], label + " fold " + fold);
    }
  }

  /** 書いた値が null でなく、有限な値とビットまで同じことを確かめる。 */
  private static void same(Double written, double value, String label) {
    Assert.assertNotNull(written, label);
    Assert.assertEquals(Double.doubleToRawLongBits(written), Double.doubleToRawLongBits(value), label);
  }

  /** JSON に書いて読み戻す。 */
  private static <R extends Record> R roundTrip(R record, Class<R> type) {
    return RecordFile.read(type, RecordFile.json(record), "test").value();
  }
}
