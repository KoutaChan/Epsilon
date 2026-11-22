package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.Settings;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LossWeightSettings;
import com.epsilon.prism.noise.RootCoefficients;
import com.epsilon.prism.noise.SlopeRefresh;
import com.epsilon.prism.rows.Row;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 同じ帳簿を持つ候補に異なる係数を掛けたときの教師と、世代をまたぐ選択を確かめる。 */
public class RootCoefficientsTest {
  private static RootHoldout.Difference difference(double mean) {
    return RootHoldout.Difference.of(new LedgerReport.Paired(2, 2, mean, 2, 0, 0));
  }

  @Test
  public void requiresAStrictForwardWinOverTheAppliedCommonScaleAndRawTargets() {
    var common =
        new com.epsilon.prism.noise.LedgerCalibration.Apply(
            com.epsilon.prism.noise.LedgerCalibrationKind.COMMON_SCALE,
            com.epsilon.prism.noise.LedgerCalibrationSource.PREVIOUS_COHORT,
            9,
            3,
            0.0001,
            0.08);
    var model = train(Variant.TENHOU_4P, true);
    double[][] losses = {{1, 2, 3}, {2, 2, 3}, {2, 3, 1}, {2, 1, 3}};
    var expected =
        List.of(
            LedgerReport.Arm.ROOT,
            LedgerReport.Arm.COMMON_SCALE,
            LedgerReport.Arm.RAW,
            LedgerReport.Arm.COMMON_SCALE);
    for (int i = 0; i < losses.length; i++) {
      double r = losses[i][0], c = losses[i][1], raw = losses[i][2];
      for (var incumbent : List.of(LedgerReport.Arm.CLASS, LedgerReport.Arm.COMMON_SCALE)) {
        var reference = incumbent == LedgerReport.Arm.CLASS ? null : common;
        var scores =
            new RootHoldout.Scores(
                4,
                4,
                1,
                raw,
                r,
                incumbent,
                reference,
                c,
                difference(r - raw),
                difference(r - c),
                difference(c - raw));
        var selected = new RootHoldout.Selection(scores.best(true), scores);
        var frozen = new RootModel(model.weights(), model.rates(), 4, model.choice(), selected);
        var winner = expected.get(i) == LedgerReport.Arm.COMMON_SCALE ? incumbent : expected.get(i);
        Assert.assertEquals(RootHoldout.decide(frozen, reference, true).arm(), winner);
        Assert.assertEquals(
            RootHoldout.decide(frozen, common, false).arm(), LedgerReport.Arm.CLASS);
      }
    }
    Assert.assertEquals(RootHoldout.decide(null, null, false).arm(), LedgerReport.Arm.CLASS);
    Assert.assertEquals(
        RootHoldout.decide(null, common, true).arm(), LedgerReport.Arm.COMMON_SCALE);
    Assert.assertEquals(
        RootHoldout.decide(train(Variant.TENHOU_4P, false), common, true).arm(),
        LedgerReport.Arm.COMMON_SCALE);
  }

  @Test
  public void recoversAnOrthonormalBasisForRepeatedAndZeroEigenvalues() {
    double[] upper = {1.5, 0, 0, 0, 1, 1.5, 0, 0, 0.5, 1, 1.5, 0, 1, 0.5, 1, 1.5};
    double[] vectors = upper.clone();
    double[] eigen = SymmetricEigen.decompose(vectors, 4);
    double[] expected = {0, 1, 1, 4};
    for (int col = 0; col < 4; col++) {
      Assert.assertEquals(eigen[col], expected[col], 1e-14);
      for (int row = 0; row < 4; row++) {
        double product = 0, inner = 0;
        for (int k = 0; k < 4; k++) {
          product += upper[Math.max(k, row) * 4 + Math.min(k, row)] * vectors[col * 4 + k];
          inner += vectors[row * 4 + k] * vectors[col * 4 + k];
        }
        Assert.assertEquals(product, eigen[col] * vectors[col * 4 + row], 1e-14);
        Assert.assertEquals(inner, row == col ? 1d : 0d, 1e-14);
      }
    }
  }

  private record Sample(Variant variant, int sign, boolean signal) implements FitRow {
    @Override
    public long gameId() {
      return sign;
    }

    @Override
    public DecisionClass decisionClass() {
      return DecisionClass.CALL;
    }

    @Override
    public int cell() {
      return 0;
    }

    @Override
    public LedgerCell.Standing standing() {
      return new LedgerCell.Standing(0, 0, 0);
    }

    @Override
    public float[] values() {
      return new float[] {0.75f + (signal ? sign * 0.25f : 0), 0.25f - (signal ? sign * 0.25f : 0)};
    }

    @Override
    public int ledgerFeatures() {
      return LedgerColumn.COUNT;
    }

    @Override
    public float ledger(int action, int col) {
      return col == 0 ? sign : 0;
    }

    @Override
    public int played() {
      return 0;
    }

    @Override
    public Row.Source source() {
      return Row.Source.MAIN;
    }

    @Override
    public RootFeatures root(RootFeatures.Reader reader, int[] actions, int count, boolean hands) {
      var common = new double[new RootFeatures.Layout(variant, false, true).width];
      common[0] = 1;
      var types = Vocabulary.of(Action.Type.class, variant);
      return new RootFeatures(
          common,
          new int[] {types.index(Action.Type.DAHAI), 0, 0, types.index(Action.Type.PASS), 0, 0},
          true,
          standing(),
          types);
    }
  }

  private static RootModel train(Variant variant, boolean signal) {
    return trainState(variant, signal).active();
  }

  private static RootTraining.State trainState(Variant variant, boolean signal) {
    var settings =
        Settings.section(
                RootCoefficients.class,
                "root",
                List.of(
                    Layer.assignments(
                        "test",
                        List.of(
                            "root.mode=SHADOW",
                            "root.trainingCohorts=1",
                            "root.curveCohorts=1",
                            "root.selectionCohorts=1"))))
            .value();
    var groups = Set.of(LedgerGroup.IMPROVE);
    var meaning =
        new LedgerMeaning(
            List.of(LedgerColumn.OWN_IMPROVE_1),
            "APPLY",
            groups,
            32d,
            8d,
            1000,
            SlopeRefresh.DEPOSIT,
            groups,
            LedgerCells.CLASS,
            LossWeightSettings.Weighting.UNIFORM,
            "REALIZED",
            null);
    RootTraining.State state = null;
    for (int cohort = 1; cohort <= 4; cohort++) {
      var fit = new RootTraining(cohort, variant, meaning, settings, state, null, null);
      Assert.assertNull(fit.active());
      fit.add(new Sample(variant, 1, signal), null);
      fit.add(new Sample(variant, -1, signal), null);
      state =
          RecordFile.read(
                  RootTraining.State.class, RecordFile.json(fit.finish(Map.of())), "ledger state")
              .value();
    }
    return state;
  }

  @Test
  public void subtractsEachCandidatesNoiseBeforeCenteringAndPreservesBothExpectations() {
    for (var variant : Variant.values()) {
      var model = train(variant, true);
      var predictor = new RootModel.Predictor(model.weights(), model.rates());
      double[] mean = new double[2];
      double[] shared = new double[predictor.columns.length];
      for (int sign : new int[] {-1, 1}) {
        var row = new Sample(variant, sign, true);
        var root = row.root(null, null, 2, true);
        double h = predictor.prepare(root, row.standing(), shared);
        for (int a = 0; a < 2; a++) {
          double y = row.values()[a] - predictor.amount(root, a, row, a, shared, h, null);
          mean[a] += y / 2;
          Assert.assertEquals(y, a == 0 ? 0.75 : 0.25, 1e-8);
        }
      }
      Assert.assertEquals(mean[0], 0.75, 1e-12);
      Assert.assertEquals(mean[1], 0.25, 1e-12);
    }
  }

  @Test
  public void choosesNoCorrectionWhenTheLedgerCarriesNoSignal() {
    var model = train(Variant.TENHOU_3P, false);
    Assert.assertNull(model.weights().ridge());
    Assert.assertTrue(model.weights().theta().stream().allMatch(x -> x == 0));
  }

  @Test
  public void poolsIndependentSingleGameBandsAroundTheirCombinedMean() {
    var first = new LedgerCalibrator.Group("band");
    var second = new LedgerCalibrator.Group("band");
    var together = new LedgerCalibrator.Group("band");
    first.add(1, 3, 3, 1);
    together.add(1, 3, 3, 1);
    second.add(2, 3, 3, 2);
    together.add(2, 3, 3, 2);
    second.add(2, 2, 2, 2);
    together.add(2, 2, 2, 2);
    var a = first.band(true);
    var b = second.band(true);
    Assert.assertNull(a.seShadowMinusRaw());
    var pooled = a.paired().plus(b.paired());
    Assert.assertEquals(pooled.se(), together.band(true).seShadowMinusRaw(), 1e-14);
    Assert.assertTrue(pooled.se() > 0);
  }

  @Test
  public void usesTheLowerObservedCurveAndChoosesRawOnTiedOrWorseHoldoutLoss() {
    var folds =
        List.of(
            new RootRates.Fold(
                3, Map.of("0", new RootRates.Curve(List.of(1, 3), List.of(0.2, 0.8)))),
            new RootRates.Fold(
                4, Map.of("0", new RootRates.Curve(List.of(2, 4), List.of(0.7, 0.5)))));
    var rates =
        new RootRates(
            RootCoefficients.Context.GLOBAL, RootCoefficients.Direction.INCREASING, folds);
    for (int gap = 0; gap <= 4; gap++)
      Assert.assertEquals(
          rates.scale(new LedgerCell.Standing(0, gap, 0), 4),
          new double[] {0, 0, 0.5, 0.6, 0}[gap],
          1e-15);
    var fit = train(Variant.TENHOU_4P, true);
    for (double loss : new double[] {0, 1}) {
      var raw =
          RootTraining.choose(
              fit.weights(), List.of(new RootTraining.Candidate(fit.rates(), loss, null)), 4);
      Assert.assertEquals(raw.choice(), RootModel.Choice.ZERO_RATE);
      Assert.assertEquals(raw.rates().scale(new LedgerCell.Standing(0, 0, 0), 4), 0d);
    }
  }

  @Test
  public void carriesTheFrozenModelAndPendingMomentsAcrossAnEmptyCohort() {
    var v = Variant.TENHOU_3P;
    var state = trainState(v, true);
    var fit = new RootTraining(5, v, state.meaning(), state.settings(), state, null, null);
    fit.add(new Sample(v, 1, true), null);
    var accumulated = fit.finish(Map.of());
    var empty =
        new RootTraining(6, v, state.meaning(), state.settings(), accumulated, null, null)
            .finish(Map.of());
    Assert.assertEquals(empty.moments(), accumulated.moments());
    Assert.assertEquals(empty.active(), state.active());
    var resumed = new RootTraining(7, v, state.meaning(), state.settings(), empty, null, null);
    Assert.assertNull(resumed.restart);
    resumed.add(new Sample(v, 1, true), null);
    Assert.assertEquals(resumed.finish(Map.of()).weights().validated(), 7);
    var gap = new RootTraining(8, v, state.meaning(), state.settings(), empty, null, null);
    Assert.assertNull(gap.active());
    Assert.assertTrue(gap.restart.contains("expected cohort 7"));
  }
}
