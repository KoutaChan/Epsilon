package com.epsilon.prism.luck;

import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.noise.LedgerCalibration;
import com.epsilon.prism.noise.LedgerCalibrationKind;
import com.epsilon.prism.noise.LedgerCalibrationSource;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.PointUnit;
import com.epsilon.prism.settings.PrismSettings;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 設定の noise セクションから作る運の精算（{@link LuckSettlement}）を確かめる。見るのは次の 4 つ。帳簿の 4 つのモードが、記録・
 * 当てはめ・補正のどこまでを行うか。計算する組、切る幅、補正に使う組、ridge、列 1 つあたりに要る行の数、偏りの警告のしきい値を設定から
 * 読むか。この jar で扱えない組み合わせを拒むか。帳簿の補正率の 3 つのモードが、教師・掛けた R²・次の反復の補正率をどう決めるか。
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
    Assert.assertEquals(
        record.recording(),
        new LedgerRecording(LedgerGroup.computed(), 32, 8, PointUnit.THOUSANDS));
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
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(byClass(expected.correction()), audited.report()), expected);
    Assert.assertEquals(audited.classes().classes().keySet(), audited.report().byClass().keySet());

    LuckSettlement apply = settle();
    Assert.assertEquals(apply.meaning().mode(), "APPLY");
    LedgerOutcome applied = apply.learn(1, tapped(apply, rows), List.of(), rows.size());
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(byClass(applied.correction()), applied.report()), expected);
    double r2 = applied.report().all().r2();
    Assert.assertTrue(r2 > 0.5, "r2 " + r2);
    Assert.assertEquals(applied.appliedR2(), r2);

    Assert.expectThrows(
        IllegalStateException.class,
        () -> apply.learn(1, tapped(apply, rows), List.of(), rows.size() + 1));
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
    Assert.assertEquals(
        luck.recording(), new LedgerRecording(compute, 4, 0.5, PointUnit.THOUSANDS));
    Assert.assertEquals(luck.width(), F);
    var rows = LedgerFitTest.rows(3000, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 52);
    LedgerOutcome outcome = luck.learn(1, tapped(luck, rows), List.of(), rows.size());
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(byClass(outcome.correction()), outcome.report()),
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
      for (int a = 0; a < row.actions(); a++)
        if (!Float.isNaN(row.values()[a])) ledger[a * F + column] += 0.1f;
      shifted.add(LedgerFitTest.withValues(row, row.values(), ledger));
    }
    LuckSettlement strict = settle("noise.ledger.mode=AUDIT", NO_GAIN);
    LuckSettlement loose = settle("noise.ledger.mode=AUDIT", NO_GAIN, "noise.ledger.biasZ=1000");
    LedgerReport.Summary flagged =
        strict.learn(1, tapped(strict, shifted), List.of(), shifted.size()).summary();
    LedgerReport.Summary quiet =
        loose.learn(1, tapped(loose, shifted), List.of(), shifted.size()).summary();
    Assert.assertEquals(flagged.biased(), List.of(LedgerColumn.OWN_TSUMO.reportName()));
    Assert.assertTrue(flagged.maxAbsZ() > 3, "max |z| " + flagged.maxAbsZ());
    Assert.assertTrue(quiet.biased().isEmpty(), quiet.biased().toString());
    Assert.assertEquals(quiet.maxAbsZ(), flagged.maxAbsZ());
  }

  /**
   * 次の設定は拒む: 古典の 5 組の外の組、計算しない組での補正、期待値の裏ドラと裏ドラの組の組み合わせ（期待値の裏ドラはそれだけでも
   * この jar では使えない）、前のコホートで当てはめた係数。古典の組の部分集合と、切る幅・ridge・行の数・しきい値の値は受け付ける。
   */
  @Test
  public void theSettingsRejectWhatTheJarCannotSettle() {
    rejects(
        "noise.ledger.compute",
        "not available",
        "noise.ledger.compute=[IMPROVE, POINTS, DORA, URA, POLICY, SLOPE]");
    rejects("noise.ledger.apply", "subset", "noise.ledger.compute=[POINTS]");
    rejects("noise.ledger.compute", "must not compute URA", "noise.ura.mode=EXPECTED");
    rejects(
        "noise.ura",
        "not available",
        "noise.ura.mode=EXPECTED",
        "noise.ledger.compute=[IMPROVE, POINTS, DORA, POLICY]",
        "noise.ledger.apply=[IMPROVE, POINTS, DORA, POLICY]");
    rejects("noise.ledger", "not available", "noise.ledger.coefficients=PREVIOUS_COHORT");
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
        new LedgerRecording(
            EnumSet.of(LedgerGroup.DORA, LedgerGroup.URA), 64, 2, PointUnit.THOUSANDS));
  }

  /**
   * 補正率の節が無い設定と OFF の設定は、同じ運の精算になる。行を貯めない受け取り先を返し、結果（係数・報告・補正・掛けた R²）はビットまで
   * 同じで、補正率の結果は無い。持ち越しのキーも同じで、補正率の形は持たない。
   */
  @Test
  public void calibrationOffIsTheLedgerAsBefore() {
    var rows = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 3000, 0, 61);
    LuckSettlement absent = settle();
    LuckSettlement off = settle("noise.ledger.calibration.mode=OFF");
    LedgerOutcome before = absent.learn(1, tapped(absent, rows), List.of(), rows.size());
    LedgerOutcome after = off.learn(1, tapped(off, rows), List.of(), rows.size(), null);
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(byClass(after.correction()), after.report()),
        new LedgerFit.Result(byClass(before.correction()), before.report()));
    Assert.assertEquals(after.appliedR2(), before.appliedR2());
    Assert.assertEquals(timeless(after.summary()), timeless(before.summary()));
    Assert.assertNull(after.calibration());
    Assert.assertNull(after.next());
    Assert.assertNull(after.summary().calibration());
    for (var row : rows)
      Assert.assertEquals(
          byClass(after.correction()).values(row), byClass(before.correction()).values(row));
    Assert.assertEquals(off.meaning(), absent.meaning());
    Assert.assertNull(off.meaning().calibration());
    LedgerFit.WriterTap tap = off.tap(1, index -> true);
    tap.fit();
    Assert.expectThrows(IllegalStateException.class, tap::retained);
  }

  /**
   * SHADOW は教師と掛けた R² を帳簿の補正のまま変えず、持ち越した補正率を掛けた補正 Y − h(o)·β·L を測り、今のコホートの行で次の反復の
   * 補正率を当てはめる。前のコホートで当てはめた補正率を、局面で帳簿の効き方が違う行に掛けると、J は帳簿の補正より下がる。持ち越しが無いか、
   * 前のコホートのものでないか、補正に使う組が違えば、補正率を 1 にして理由を書く。
   */
  @Test
  public void shadowMeasuresTheCarriedScaleButKeepsTheTeacher() {
    var before = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 0, 62);
    var rows = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 1_000_000, 63);
    LuckSettlement off = settle();
    LuckSettlement shadow = settle("noise.ledger.calibration.mode=SHADOW");
    LedgerOutcome first = shadow.learn(1, tapped(shadow, before), List.of(), before.size(), null);
    LedgerCalibrator carried = first.next();
    Assert.assertNotNull(carried);
    Assert.assertEquals(carried.cohort(), 1);
    Assert.assertEquals(carried.variant(), Variant.TENHOU_4P);
    Assert.assertEquals(carried.coefficients().size(), 108);
    Assert.assertEquals(carried.shrink(), 0.08);
    Assert.assertTrue(first.calibration().report().fallback().startsWith("no calibrator"));

    LedgerOutcome plain = off.learn(2, tapped(off, rows), List.of(), rows.size());
    LedgerOutcome measured = shadow.learn(2, tapped(shadow, rows), List.of(), rows.size(), carried);
    LedgerFitTest.assertBitwiseEqual(
        new LedgerFit.Result(byClass(measured.correction()), measured.report()),
        new LedgerFit.Result(byClass(plain.correction()), plain.report()));
    Assert.assertEquals(measured.appliedR2(), plain.appliedR2());
    for (var row : rows)
      Assert.assertEquals(
          byClass(measured.correction()).values(row), byClass(plain.correction()).values(row));
    LedgerReport.Calibration report = measured.calibration().report();
    Assert.assertEquals(report.mode(), "SHADOW");
    Assert.assertEquals(report.applied(), "current");
    Assert.assertNull(report.fallback());
    Assert.assertEquals(report.origin().cohort(), 1);
    Assert.assertEquals(report.origin().source(), LedgerCalibrationSource.PREVIOUS_COHORT);
    Assert.assertEquals(report.rows(), rows.size());
    Assert.assertTrue(report.j().shadow() < report.j().current(), report.j().toString());
    Assert.assertTrue(report.j().current() < report.j().raw(), report.j().toString());
    Assert.assertTrue(report.gates().isEmpty());
    Assert.assertEquals(measured.next().cohort(), 2);
    Assert.assertEquals(measured.summary().calibration(), report.summary());
    Assert.assertEquals(measured.summary().calibration().origin(), Integer.valueOf(1));
    Assert.assertEquals(measured.full(null).calibration(), report);

    for (Object[] stale :
        new Object[][] {
          {null, "no calibrator was carried from cohort 2"},
          {first.next(), "the carried calibrator was fit on cohort 1, not 2"}
        }) {
      LedgerOutcome wrong =
          shadow.learn(
              3, tapped(shadow, rows), List.of(), rows.size(), (LedgerCalibrator) stale[0]);
      LedgerReport.Calibration fallback = wrong.calibration().report();
      Assert.assertEquals(fallback.fallback(), stale[1]);
      Assert.assertNull(fallback.origin());
      Assert.assertEquals(fallback.j().shadow(), fallback.j().current());
    }
    LuckSettlement grouped =
        settle("noise.ledger.calibration.mode=SHADOW", "noise.ledger.apply=[POINTS, POLICY]");
    LuckSettlement clipped =
        settle("noise.ledger.calibration.mode=SHADOW", "noise.ledger.pointClip=16");
    for (Object[] meaning :
        new Object[][] {
          {grouped, shadow.meaning(), "the carried calibrator scales the groups"},
          {
            clipped,
            shadow.meaning(),
            "the carried calibrator was fit on another ledger correction: the ledger pointClip"
                + " changed from 32.0 to 16.0"
          },
          {shadow, null, "the ledger settings that fit the carried calibrator are unknown"}
        }) {
      LuckSettlement other = (LuckSettlement) meaning[0];
      LedgerOutcome changed =
          other.learn(
              2, tapped(other, rows), List.of(), rows.size(), carried, (LedgerMeaning) meaning[1]);
      String fallback = changed.calibration().report().fallback();
      Assert.assertTrue(fallback.startsWith((String) meaning[2]), fallback);
      Assert.assertNull(changed.calibration().report().origin(), fallback);
    }
    Assert.assertNull(
        shadow
            .learn(2, tapped(shadow, rows), List.of(), rows.size(), carried, shadow.meaning())
            .calibration()
            .report()
            .fallback());
  }

  /**
   * 帳簿を当てはめて監査するだけのモード（AUDIT）で補正率を求めると、教師は生の Y なので、掛けた腕は RAW、掛けた R² は 0。補正率は
   * 測って、次の反復の補正率も当てはめる。
   */
  @Test
  public void auditingTheLedgerTeachesTheRawArm() {
    var before = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 0, 68);
    var rows = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 1_000_000, 69);
    LuckSettlement audit =
        settle("noise.ledger.mode=AUDIT", "noise.ledger.calibration.mode=SHADOW", NO_GAIN);
    LedgerCalibrator carried =
        audit.learn(1, tapped(audit, before), List.of(), before.size(), null).next();
    LedgerOutcome measured = audit.learn(2, tapped(audit, rows), List.of(), rows.size(), carried);
    LedgerReport.Calibration report = measured.calibration().report();
    Assert.assertEquals(report.applied(), "raw");
    Assert.assertEquals(measured.summary().calibration().applied(), "raw");
    Assert.assertNull(measured.correction());
    Assert.assertEquals(measured.appliedR2(), 0.0);
    Assert.assertEquals(report.origin().cohort(), 1);
    Assert.assertEquals(measured.next().cohort(), 2);
  }

  /**
   * APPLY は、持ち越した補正率を掛けた補正 Y − h(o)·β·L を教師にし、掛けた R² をその補正のスロットの R² にする。β は帳簿の補正と同じ
   * もの。持ち越しが無い反復は、教師も掛けた R² も帳簿の補正のまま。
   */
  @Test
  public void applyTeachesTheScaledCorrection() {
    var before = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 0, 64);
    var rows = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 1_000_000, 65);
    LuckSettlement off = settle();
    LuckSettlement apply = settle("noise.ledger.calibration.mode=APPLY");
    LedgerCalibrator carried =
        apply.learn(1, tapped(apply, before), List.of(), before.size(), null).next();
    LedgerOutcome plain = off.learn(2, tapped(off, rows), List.of(), rows.size());
    LedgerOutcome scaled = apply.learn(2, tapped(apply, rows), List.of(), rows.size(), carried);
    LedgerReport.Calibration report = scaled.calibration().report();
    Assert.assertEquals(report.mode(), "APPLY");
    Assert.assertEquals(report.applied(), "shadow");
    Assert.assertEquals(scaled.appliedR2(), Math.max(0, report.slotR2().shadow()));
    Assert.assertNotEquals(scaled.appliedR2(), plain.appliedR2());
    Assert.assertSame(byClass(scaled.correction()).scale(), carried);
    int changed = 0;
    for (var row : rows) {
      double[] beta =
          byClass(plain.correction()).coefficients(row.gameId(), row.decisionClass().ordinal(), 0);
      double h = carried.scale(row.standing());
      float[] teacher = byClass(scaled.correction()).values(row);
      for (int a = 0; a < 3; a++) {
        double expected = row.values()[a];
        for (int f = 0; f < LedgerColumn.COUNT; f++) expected -= h * beta[f] * row.ledger(a, f);
        Assert.assertEquals(teacher[a], (float) expected, 1e-6f);
        if (teacher[a] != byClass(plain.correction()).values(row)[a]) changed++;
      }
      Assert.assertTrue(Float.isNaN(teacher[3]));
    }
    Assert.assertTrue(changed > rows.size(), "changed " + changed);

    LedgerOutcome first = apply.learn(1, tapped(apply, rows), List.of(), rows.size(), null);
    LedgerOutcome reference = off.learn(1, tapped(off, rows), List.of(), rows.size());
    Assert.assertEquals(first.calibration().report().applied(), "current");
    Assert.assertEquals(first.appliedR2(), reference.appliedR2());
    Assert.assertNull(byClass(first.correction()).scale());
    for (var row : rows)
      Assert.assertEquals(
          byClass(first.correction()).values(row), byClass(reference.correction()).values(row));
  }

  /**
   * 補正率の設定の既定値は、形 COMMON_SCALE、学習元は前のコホート、格子 9 × 3、二乗の罰の係数 1e-4、縮める罰 ρ 0.08。どの値も
   * 設定で変えられ、ρ は 0 でもよい。AUDIT では SHADOW まで選べ、APPLY は帳簿の APPLY でだけ選べる。範囲の外の値（負の ρ、0 以下の
   * 二乗の罰の係数、2 点未満か上限を越える格子）を拒み、帳簿を当てはめないモードには補正率の節が無い。
   */
  @Test
  public void calibrationSettingsHaveTheDesignDefaults() {
    PrismSettings shadow = TestSettings.of("noise.ledger.calibration.mode=SHADOW");
    var fitting = (LedgerSettings.Fitting) shadow.noise().ledger();
    Assert.assertEquals(
        fitting.calibration(),
        new LedgerCalibration.Shadow(
            LedgerCalibrationKind.COMMON_SCALE,
            LedgerCalibrationSource.PREVIOUS_COHORT,
            9,
            3,
            1e-4,
            0.08));
    Assert.assertFalse(fitting.calibration().applies());
    PrismSettings tuned =
        TestSettings.of(
            "noise.ledger.calibration.mode=APPLY",
            "noise.ledger.calibration.source=PREVIOUS_COHORT",
            "noise.ledger.calibration.gapGrid=5",
            "noise.ledger.calibration.roundGrid=2",
            "noise.ledger.calibration.regularization=0.003",
            "noise.ledger.calibration.shrink=0");
    Assert.assertEquals(
        ((LedgerSettings.Fitting) tuned.noise().ledger()).calibration(),
        new LedgerCalibration.Apply(
            LedgerCalibrationKind.COMMON_SCALE,
            LedgerCalibrationSource.PREVIOUS_COHORT,
            5,
            2,
            0.003,
            0));
    rejects(
        "noise.ledger.calibration.shrink",
        ">= 0",
        "noise.ledger.calibration.mode=SHADOW",
        "noise.ledger.calibration.shrink=-0.01");
    rejects(
        "noise.ledger.calibration.regularization",
        "> 0",
        "noise.ledger.calibration.mode=SHADOW",
        "noise.ledger.calibration.regularization=0");
    rejects(
        "noise.ledger.calibration.source",
        "PREVIOUS_COHORT",
        "noise.ledger.calibration.mode=SHADOW",
        "noise.ledger.calibration.source=THIS_COHORT");
    var none = (LedgerSettings.Fitting) TestSettings.of().noise().ledger();
    Assert.assertEquals(none.calibration(), new LedgerCalibration.Off());
    PrismSettings audited =
        TestSettings.of("noise.ledger.mode=AUDIT", NO_GAIN, "noise.ledger.calibration.mode=SHADOW");
    Assert.assertTrue(((LedgerSettings.Fitting) audited.noise().ledger()).calibration().scales());
    rejects(
        "noise.ledger.calibration",
        "needs ledger mode APPLY",
        "noise.ledger.mode=AUDIT",
        NO_GAIN,
        "noise.ledger.calibration.mode=APPLY");
    rejects(
        "noise.ledger.calibration.gapGrid",
        "[2, 17]",
        "noise.ledger.calibration.mode=SHADOW",
        "noise.ledger.calibration.gapGrid=1");
    rejects(
        "noise.ledger.calibration.roundGrid",
        "[2, 9]",
        "noise.ledger.calibration.mode=APPLY",
        "noise.ledger.calibration.roundGrid=10");
    rejects(
        "noise.ledger.calibration",
        "",
        "noise.ledger.mode=RECORD",
        NO_GAIN,
        "noise.ledger.calibration.mode=SHADOW");
  }

  /**
   * 教師を決めるまで（settle）と補正率の 2 回目の走査（calibrate）に分けても、続けて回した learn と同じ結果になる。教師に掛ける補正は
   * 走査の前に決まり（APPLY では持ち越した補正率を掛けたもの）、走査は 1 度だけ回って同じ結果を返す。補正率を求めない段では、走査は無く、
   * 帳簿の補正をそのまま返す。
   */
  @Test
  public void settlingBeforeTheSecondPassKeepsTheOutcome() {
    var before = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 0, 66);
    var rows = LedgerCalibratorTest.scaledRows(Variant.TENHOU_4P, 20_000, 1_000_000, 67);
    LuckSettlement apply = settle("noise.ledger.calibration.mode=APPLY");
    LedgerCalibrator carried =
        apply.learn(1, tapped(apply, before), List.of(), before.size(), null).next();
    LuckSettlement.Settled settled =
        apply.settle(2, tapped(apply, rows), List.of(), rows.size(), carried);
    var teacher = byClass(settled.correction());
    Assert.assertSame(teacher.scale(), carried);
    LedgerOutcome calibrated = settled.calibrate();
    Assert.assertSame(settled.calibrate(), calibrated);
    Assert.assertSame(calibrated.correction(), teacher);
    LedgerOutcome learned = apply.learn(2, tapped(apply, rows), List.of(), rows.size(), carried);
    Assert.assertEquals(calibrated.appliedR2(), learned.appliedR2());
    Assert.assertEquals(calibrated.calibration().report(), learned.calibration().report());
    Assert.assertEquals(calibrated.next(), learned.next());
    for (var row : rows)
      Assert.assertEquals(teacher.values(row), byClass(learned.correction()).values(row));

    LuckSettlement off = settle();
    LuckSettlement.Settled plain = off.settle(2, tapped(off, rows), List.of(), rows.size(), null);
    Assert.assertSame(plain.calibrate().correction(), plain.correction());
    Assert.assertNull(plain.calibrate().calibration());
  }

  /** 秒数を 0 にした要約。 */
  private static LedgerReport.Summary timeless(LedgerReport.Summary summary) {
    return new LedgerReport.Summary(
        summary.rows(),
        summary.slots(),
        summary.skippedRows(),
        summary.r2(),
        summary.equivalentForks(),
        summary.r2ByGroup(),
        summary.maxAbsZ(),
        summary.biased(),
        0,
        summary.streamed(),
        summary.calibration());
  }

  private static void rejects(String path, String because, String... assignments) {
    SettingsException failure =
        Assert.expectThrows(SettingsException.class, () -> TestSettings.of(assignments));
    boolean named =
        failure.problems().stream()
            .anyMatch(problem -> problem.contains(path) && problem.contains(because));
    Assert.assertTrue(named, String.join(" ", assignments) + " -> " + failure.problems());
  }

  private static LedgerCorrection.ByClass byClass(LedgerCorrection correction) {
    return (LedgerCorrection.ByClass) correction;
  }
}
