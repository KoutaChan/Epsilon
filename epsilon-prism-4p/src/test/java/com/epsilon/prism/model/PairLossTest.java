package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.Row;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PairLossTest {
  private static final Shape SHAPE = new Shape(4, 4);
  private static final Row STAR_ROW =
      new Row(
          1,
          1,
          DecisionClass.CALL,
          Row.Purpose.TRAIN,
          Row.Source.MAIN,
          3,
          new float[] {0.2f, Float.NaN, -0.12f, 0.31f},
          new float[4],
          0b1101,
          -1,
          Float.NaN,
          Float.NaN,
          -1,
          new float[0],
          new long[4 * OutcomeCodec.WORDS],
          new long[0],
          new long[0]);
  private static final List<ValueLabel> LABELS =
      List.of(
          new ValueLabel(STAR_ROW.values(), 0.01f, 1.5f),
          new ValueLabel(new float[] {0.12f, -0.31f, 0.05f}, 0.02f, 0.7f),
          new ValueLabel(new float[] {Float.NaN, 0.1f}, 0.01f, 2f),
          new ValueLabel(new float[] {Float.NaN}, 0.02f, 1f));
  private static final float[] SCORES = {
    -0.021f, 1000f, 0.012f, 0.012f, -5f, 0f, 4f, 0f,
    0.3f, 0.1f, 0f, 0f, 0f, 0f, 0f, 0f
  };

  private record Result(float value, float[] gradient) {}

  private static Result result(PairLoss pair) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NDArray scores = manager.create(SCORES, SHAPE);
      scores.setRequiresGradient(true);
      Outputs outputs =
          new Outputs(
              scores,
              scores.onesLike(),
              new int[] {4, 3, 2, 1},
              new boolean[(int) SHAPE.size()],
              null);
      LossInputs inputs = LossInputs.of(manager, outputs, LABELS, null, null);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray loss =
            new PrismLoss(AuxiliaryLoss.standard()).total(outputs, inputs, 1f, 2, 1f, pair, true);
        collector.backward(loss);
        return new Result(loss.getFloat(), scores.getGradient().toFloatArray());
      }
    }
  }

  @Test
  public void baseMatchesThePrismGradient() {
    Result prism = result(null);
    Result base =
        result(new PairLoss(PairLoss.Weighting.BASE, PairLoss.ClassWeight.INVERSE_LAMBDA));
    Assert.assertEquals(base.value(), prism.value());
    for (int a = 0; a < SCORES.length; a++) {
      Assert.assertEquals(base.gradient()[a], prism.gradient()[a], 2e-5f, "action " + a);
    }
  }

  @Test
  public void uniformClassWeightsRemoveOnlyTheInverseTemperature() {
    Result prism = result(null);
    Result uniform = result(PairLoss.Mode.PRISM.objective(PairLoss.ClassWeight.UNIFORM));
    Assert.assertEquals(uniform.value(), prism.value());
    for (int row = 0; row < LABELS.size(); row++) {
      for (int a = 0; a < SHAPE.get(1); a++) {
        int position = row * (int) SHAPE.get(1) + a;
        Assert.assertEquals(
            uniform.gradient()[position],
            prism.gradient()[position] * LABELS.get(row).lambda(),
            1e-6f,
            "action " + position);
      }
    }
  }

  @Test
  public void rowAndStarFollowTheirOrderedPairWeights() {
    Result prism = result(null);
    for (PairLoss.Weighting mode : List.of(PairLoss.Weighting.ROW, PairLoss.Weighting.STAR)) {
      for (PairLoss.ClassWeight weight : PairLoss.ClassWeight.values()) {
        Result result = result(new PairLoss(mode, weight));
        Assert.assertEquals(result.value(), prism.value());
        for (int row = 0; row < LABELS.size(); row++) {
          double[] expected = gradient(row, mode, weight, 2);
          for (int a = 0; a < expected.length; a++) {
            Assert.assertEquals(
                result.gradient()[row * 4 + a],
                expected[a],
                3e-5 * Math.max(1, Math.abs(expected[a])),
                "row " + row + " action " + a);
          }
        }
      }
    }
  }

  @Test
  public void starUsesTheFirstMaximumAmongMeasuredActions() {
    int base = 2;
    Assert.assertEquals(STAR_ROW.played(), 3);
    Assert.assertEquals(Integer.numberOfTrailingZeros(STAR_ROW.support()), 0);
    Assert.assertEquals(SCORES[base], SCORES[STAR_ROW.played()]);
    Assert.assertTrue(SCORES[1] > SCORES[base] && !STAR_ROW.measured(1));
    PairLoss loss = new PairLoss(PairLoss.Weighting.STAR, PairLoss.ClassWeight.UNIFORM);
    float[] actual = result(loss).gradient();
    double[] expected = gradient(0, loss.weighting(), loss.classWeight(), base);
    for (int a = 0; a < STAR_ROW.actions(); a++) {
      Assert.assertEquals(actual[a], expected[a], 1e-6, "action " + a);
    }
  }

  private static double[] gradient(
      int row, PairLoss.Weighting mode, PairLoss.ClassWeight weight, int base) {
    ValueLabel label = LABELS.get(row);
    int count = label.measuredCount();
    double[] out = new double[4];
    if (count < 2) return out;
    double[] p = new double[4];
    double[] error = new double[4];
    double sum = 0;
    for (int a = 0; a < label.values().length; a++) {
      if (Float.isNaN(label.values()[a])) continue;
      double score = SCORES[row * 4 + a];
      p[a] = Math.exp(score / label.lambda());
      sum += p[a];
      error[a] = score - label.values()[a];
    }
    for (int a = 0; a < p.length; a++) p[a] /= sum;
    double mass = 0;
    for (int a = 0; a < label.values().length; a++) {
      if (Float.isNaN(label.values()[a])) continue;
      for (int b = 0; b < label.values().length; b++) {
        if (a == b
            || Float.isNaN(label.values()[b])
            || (mode == PairLoss.Weighting.STAR && b != base)) continue;
        double w =
            mode == PairLoss.Weighting.STAR
                ? p[a] + PrismLoss.WEIGHT_FLOOR / count
                : p[a] * p[b] + PrismLoss.WEIGHT_FLOOR / (count * count);
        mass += w;
        double g = w * (error[a] - error[b]);
        out[a] += g;
        if (mode == PairLoss.Weighting.STAR) out[b] -= g;
      }
    }
    double k = weight == PairLoss.ClassWeight.INVERSE_LAMBDA ? 1.0 / label.lambda() : 1.0;
    for (int a = 0; a < out.length; a++) out[a] *= label.weight() * k / (2 * mass);
    return out;
  }
}
