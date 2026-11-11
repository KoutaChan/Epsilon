package com.epsilon.prism.luck;

import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.settings.PrismSettings;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 設定の noise セクションから作る運の精算（{@link LuckSettlement}）を確かめる。見るのは次の 3 つ。帳簿の 4 つのモードが、記録・
 * 当てはめ・補正のどこまでを行うか。計算する組、切る幅、補正に使う組、ridge、列 1 つあたりに要る行の数、偏りの警告のしきい値を設定から
 * 読むか。この jar で扱えない組み合わせを拒むか。
 */
public class LuckSettlementTest {
  private static final int F = LedgerColumn.COUNT;

  /** 補正の R² でゲインを上げない TEMPO の設定（補正しないモードではこれが要る）。 */
  private static final String NO_GAIN = "noise.tempo.noiseGain=NONE";

  private static LuckSettlement settle(String... assignments) {
    PrismSettings settings = TestSettings.of(assignments);
    return LuckSettlement.of(settings.noise(), settings.run().variant());
  }

  /** ライターの行を受け取るタップに、シャード 0 のライターが書いた順で行を足したもの。 */
  private static LedgerFit.WriterTap tapped(LuckSettlement luck, List<? extends FitRow> rows) {
    LedgerFit.WriterTap tap = luck.tap(1, index -> true);
    Assert.assertNotNull(tap);
    for (FitRow row : rows) tap.add(0, row);
    return tap;
  }

  /**
   * OFF は帳簿を計算せず（0 列）、和も作らず、学習の段では何もしない。RECORD は古典の 5 組を既定の幅で計算して 17 列を書くが、和は
   * 作らない。AUDIT は和から当てはめて報告するが、補正はしないので、掛けた R² は 0。APPLY は報告と補正を返し、掛けた R² は全体の R²。
   * 報告と補正は、古典の 5 組を既定の値で当てはめたものとビットで同じ。和に足した行の数が選んだ行の数と違えば例外を投げる。
   */
  @Test
  public void eachModeRecordsFitsAndAppliesItsOwnPart() {
    var rows = LedgerFitTest.rows(6000, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 50);
    LedgerFit.Result expected = LedgerFitTest.CLASSIC.fitRows(rows);

    LuckSettlement off = settle("noise.ledger.mode=OFF", NO_GAIN);
    Assert.assertNull(off.recording());
    Assert.assertEquals(off.width(), 0);
    Assert.assertNull(off.tap(2, index -> true));
    Assert.assertSame(off.learn(1, null, List.of(), 0), LedgerOutcome.NONE);
    Assert.assertEquals(off.meaning().mode(), "OFF");
    Assert.assertTrue(off.meaning().columns().isEmpty());

    LuckSettlement record = settle("noise.ledger.mode=RECORD", NO_GAIN);
    Assert.assertEquals(record.recording(), new LedgerRecording(LedgerGroup.computed(), 32, 8));
    Assert.assertEquals(record.width(), F);
    Assert.assertEquals(record.width(), LedgerCatalog.of(Variant.TENHOU_4P).width());
    Assert.assertNull(record.tap(2, index -> true));
    Assert.assertSame(record.learn(1, null, List.of(), 0), LedgerOutcome.NONE);

    LuckSettlement audit = settle("noise.ledger.mode=AUDIT", NO_GAIN);
    Assert.assertEquals(audit.width(), F);
    LedgerOutcome audited = audit.learn(1, tapped(audit, rows), List.of(), rows.size());
    Assert.assertNull(audited.correction());
    Assert.assertEquals(audited.appliedR2(), 0.0);
    Assert.assertTrue(audited.summary().streamed());
    LedgerFitTest.assertBitwiseEqual(new LedgerFit.Result(expected.correction(), audited.report()), expected);
    Assert.assertEquals(audited.classes().classes().keySet(), audited.report().byClass().keySet());

    LuckSettlement apply = settle();
    Assert.assertEquals(apply.meaning().mode(), "APPLY");
    LedgerOutcome applied = apply.learn(1, tapped(apply, rows), List.of(), rows.size());
    LedgerFitTest.assertBitwiseEqual(new LedgerFit.Result(applied.correction(), applied.report()), expected);
    double r2 = applied.report().all().r2();
    Assert.assertTrue(r2 > 0.5, "r2 " + r2);
    Assert.assertEquals(applied.appliedR2(), r2);

    Assert.expectThrows(
        IllegalStateException.class, () -> apply.learn(1, tapped(apply, rows), List.of(), rows.size() + 1));
  }

  /**
   * APPLY でも、帳簿のある行が無ければ補正は無く、掛けた R² は 0。どの行も測った手の Y が同じで R² が有限にならないときは、補正は
   * 返すが、掛けた R² は 0。
   */
  @Test
  public void theAppliedR2IsZeroWithoutAFiniteFit() {
    LuckSettlement apply = settle();
    var rows = LedgerFitTest.rows(200, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 51);
    List<LedgerFitTest.SyntheticRow> bare = new ArrayList<>();
    List<LedgerFitTest.SyntheticRow> flat = new ArrayList<>();
    for (var row : rows) {
      bare.add(LedgerFitTest.withValues(row, row.values(), null));
      float[] values = row.values().clone();
      for (int a = 0; a < values.length; a++) if (!Float.isNaN(values[a])) values[a] = 0.25f;
      flat.add(LedgerFitTest.withValues(row, values, row.ledger()));
    }
    LedgerOutcome none = apply.learn(1, tapped(apply, bare), List.of(), bare.size());
    Assert.assertEquals(none.report().rows(), 0L);
    Assert.assertEquals(none.report().skippedRows(), (long) bare.size());
    Assert.assertNull(none.correction());
    Assert.assertEquals(none.appliedR2(), 0.0);

    LedgerOutcome nan = apply.learn(1, tapped(apply, flat), List.of(), flat.size());
    Assert.assertTrue(Double.isNaN(nan.report().all().r2()));
    Assert.assertNotNull(nan.correction());
    Assert.assertEquals(nan.appliedR2(), 0.0);
  }

  /**
   * 計算する組（POINTS と POLICY）、点と方策の切る幅、補正に使う組（POLICY）、ridge 0.5、列 1 つあたり 10 行を設定から読む。収集で使う値は
   * その組と幅を持ち、行の列の数は 17 のまま。当てはめの結果は、その組・ridge・行の数で当てはめたものとビットで同じ。持ち越しのキーも同じ値を持つ。
   */
  @Test
  public void theNoiseSectionSetsTheLedgerValues() {
    LuckSettlement luck =
        settle(
            "noise.ledger.compute=[POINTS, POLICY]",
            "noise.ledger.apply=[POLICY]",
            "noise.ledger.pointClip=4",
            "noise.ledger.policyClip=0.5",
            "noise.ledger.ridge=0.5",
            "noise.ledger.minRowsPerColumn=10");
    Set<LedgerGroup> compute = EnumSet.of(LedgerGroup.POINTS, LedgerGroup.POLICY);
    Assert.assertEquals(luck.recording(), new LedgerRecording(compute, 4, 0.5));
    Assert.assertEquals(luck.width(), F);
    var rows = LedgerFitTest.rows(3000, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 52);
    LedgerOutcome outcome = luck.learn(1, tapped(luck, rows), List.of(), rows.size());
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(outcome.correction(), outcome.report()),
        new LedgerFit(Set.of(LedgerGroup.POLICY), 0.5, 10).fitRows(rows));
    LedgerMeaning meaning = luck.meaning();
    Assert.assertEquals(meaning.compute(), compute);
    Assert.assertEquals(meaning.apply(), Set.of(LedgerGroup.POLICY));
    Assert.assertEquals(meaning.pointClip(), Double.valueOf(4));
    Assert.assertEquals(meaning.policyClip(), Double.valueOf(0.5));
    Assert.assertEquals(meaning.columns(), LedgerCatalog.of(Variant.TENHOU_4P).columns());
  }

  /**
   * 偏りの警告のしきい値は、設定の biasZ で決まる。自分のツモの列に平均 0.1 を足した行では、既定の 3 ならその列の名前が警告に載り、1000 なら
   * 載らない。z の絶対値の最大はどちらでも同じ。
   */
  @Test
  public void theBiasWarningReadsBiasZ() {
    var rows = LedgerFitTest.rows(6000, 0, LedgerFitTest.ones(), new double[F], 0.3, 53);
    int column = LedgerColumn.OWN_TSUMO.ordinal();
    List<LedgerFitTest.SyntheticRow> shifted = new ArrayList<>();
    for (var row : rows) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++) if (!Float.isNaN(row.values()[a])) ledger[a * F + column] += 0.1f;
      shifted.add(LedgerFitTest.withValues(row, row.values(), ledger));
    }
    LuckSettlement strict = settle("noise.ledger.mode=AUDIT", NO_GAIN);
    LuckSettlement loose = settle("noise.ledger.mode=AUDIT", NO_GAIN, "noise.ledger.biasZ=1000");
    LedgerReport.Summary flagged = strict.learn(1, tapped(strict, shifted), List.of(), shifted.size()).summary();
    LedgerReport.Summary quiet = loose.learn(1, tapped(loose, shifted), List.of(), shifted.size()).summary();
    Assert.assertEquals(flagged.biased(), List.of(LedgerColumn.OWN_TSUMO.reportName()));
    Assert.assertTrue(flagged.maxAbsZ() > 3, "max |z| " + flagged.maxAbsZ());
    Assert.assertTrue(quiet.biased().isEmpty(), quiet.biased().toString());
    Assert.assertEquals(quiet.maxAbsZ(), flagged.maxAbsZ());
  }

  /**
   * 次の設定は拒む: 古典の 5 組の外の組、計算しない組での補正、期待値の裏ドラと裏ドラの組の組み合わせ（期待値の裏ドラはそれだけでも
   * この jar では使えない）、クラス以外の区分。古典の組の部分集合と、切る幅・ridge・行の数・しきい値の値は受け付ける。
   */
  @Test
  public void theSettingsRejectWhatTheJarCannotSettle() {
    rejects("noise.ledger.compute", "not available", "noise.ledger.compute=[IMPROVE, POINTS, DORA, URA, POLICY, SLOPE]");
    rejects("noise.ledger.apply", "subset", "noise.ledger.compute=[POINTS]");
    rejects("noise.ledger.compute", "must not compute URA", "noise.ura.mode=EXPECTED");
    rejects(
        "noise.ura",
        "not available",
        "noise.ura.mode=EXPECTED",
        "noise.ledger.compute=[IMPROVE, POINTS, DORA, POLICY]",
        "noise.ledger.apply=[IMPROVE, POINTS, DORA, POLICY]");
    rejects("noise.ledger", "not available", "noise.ledger.cells=CLASS_STAGE");
    PrismSettings accepted =
        TestSettings.of(
            "noise.ledger.mode=AUDIT",
            NO_GAIN,
            "noise.ledger.compute=[DORA, URA]",
            "noise.ledger.apply=[URA]",
            "noise.ledger.pointClip=64",
            "noise.ledger.policyClip=2",
            "noise.ledger.ridge=0.01",
            "noise.ledger.minRowsPerColumn=4",
            "noise.ledger.biasZ=5");
    Assert.assertEquals(
        LuckSettlement.of(accepted.noise(), accepted.run().variant()).recording(),
        new LedgerRecording(EnumSet.of(LedgerGroup.DORA, LedgerGroup.URA), 64, 2));
  }

  private static void rejects(String path, String because, String... assignments) {
    SettingsException failure = Assert.expectThrows(SettingsException.class, () -> TestSettings.of(assignments));
    boolean named = failure.problems().stream().anyMatch(problem -> problem.contains(path) && problem.contains(because));
    Assert.assertTrue(named, String.join(" ", assignments) + " -> " + failure.problems());
  }
}
