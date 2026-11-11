package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** GrpNetwork を確かめる。順列の softmax から求めた周辺確率の和、Φ の範囲と和、段階ごとの出力ヘッド、重みと作業領域の検査。 */
public class GrpNetworkTest {

  private static final double TOLERANCE = 1e-12;

  @Test
  public void marginalsOfTheJointSoftmaxSumToOneByRowAndColumn() {
    for (Variant variant : Variant.values()) {
      for (boolean phaseHeads : new boolean[] {false, true}) {
        GrpShape shape = GrpFixtures.shape(variant, 16, phaseHeads);
        GrpNetwork network = GrpFixtures.network(shape, 11, 0.8);
        GrpWorkspace workspace = GrpWorkspace.of(shape);
        RankPermutations table = RankPermutations.of(variant);
        int seats = variant.seats();
        double[] probabilities = new double[table.count()];
        double[] marginals = new double[seats * seats];
        Random random = new Random(5);
        for (int trial = 0; trial < 1_000; trial++) {
          network.permutations(GrpFixtures.state(variant, random), workspace, probabilities);
          double total = 0;
          for (double probability : probabilities) {
            Assert.assertTrue(probability >= 0 && probability <= 1, Double.toString(probability));
            total += probability;
          }
          Assert.assertEquals(total, 1, TOLERANCE);
          table.marginals(probabilities, marginals);
          for (int i = 0; i < seats; i++) {
            double row = 0;
            double column = 0;
            for (int j = 0; j < seats; j++) {
              row += marginals[i * seats + j];
              column += marginals[j * seats + i];
            }
            Assert.assertEquals(row, 1, TOLERANCE);
            Assert.assertEquals(column, 1, TOLERANCE);
          }
        }
      }
    }
  }

  @Test
  public void phiStaysWithinTheUtilitiesAndKeepsTheirSum() {
    for (Variant variant : Variant.values()) {
      GrpShape shape = GrpFixtures.shape(variant, 16, true);
      GrpNetwork network = GrpFixtures.network(shape, 13, 2.0);
      GrpWorkspace workspace = GrpWorkspace.of(shape);
      double[] phi = new double[variant.seats()];
      for (RankUtility utility : RankUtility.values()) {
        double lowest = utility.value(variant, variant.seats() - 1);
        double highest = utility.value(variant, 0);
        double sum = 0;
        for (int rank = 0; rank < variant.seats(); rank++) {
          sum += utility.value(variant, rank);
        }
        Random random = new Random(17);
        for (int trial = 0; trial < 1_000; trial++) {
          network.phi(GrpFixtures.state(variant, random), utility, workspace, phi);
          double total = 0;
          for (double value : phi) {
            Assert.assertTrue(value >= lowest - TOLERANCE && value <= highest + TOLERANCE, Double.toString(value));
            total += value;
          }
          Assert.assertEquals(total, sum, TOLERANCE);
        }
      }
    }
  }

  @Test
  public void phiIsTheExpectedUtilityInAbsoluteSeats() {
    Variant variant = Variant.TENHOU_4P;
    GrpShape shape = GrpFixtures.shape(variant, 16, false);
    GrpNetwork network = GrpFixtures.network(shape, 19, 1.0);
    GrpWorkspace workspace = GrpWorkspace.of(shape);
    RankPermutations table = RankPermutations.of(variant);
    double[] probabilities = new double[table.count()];
    double[] phi = new double[4];
    Random random = new Random(23);
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(variant, random);
      network.permutations(state, workspace, probabilities);
      network.phi(state, RankUtility.TENHOU, workspace, phi);
      for (int seat = 0; seat < 4; seat++) {
        double expected = 0;
        for (int permutation = 0; permutation < table.count(); permutation++) {
          expected += probabilities[permutation] * RankUtility.TENHOU.value(variant, table.rank(permutation, seat));
        }
        Assert.assertEquals(phi[(state.dealer() + seat) % 4], expected, TOLERANCE, state + " seat " + seat);
      }
    }
  }

  @Test
  public void logProbabilitiesMatchTheSoftmaxAndStayFiniteBelowUnderflow() {
    Variant variant = Variant.TENHOU_4P;
    GrpShape shape = GrpFixtures.shape(variant, 8, false);
    GrpNetwork network = GrpFixtures.network(shape, 47, 1.0);
    GrpWorkspace workspace = GrpWorkspace.of(shape);
    double[] probabilities = new double[24];
    Random random = new Random(53);
    for (int trial = 0; trial < 200; trial++) {
      GrpState state = GrpFixtures.state(variant, random);
      network.permutations(state, workspace, probabilities);
      for (int permutation = 0; permutation < 24; permutation++) {
        double log = network.logProbability(state, permutation, workspace);
        Assert.assertEquals(log, StrictMath.log(probabilities[permutation]), 1e-9, state + " " + permutation);
      }
    }
    float[] weights = network.weights();
    int[] starts = GrpTensor.starts(shape);
    for (int index = starts[GrpTensor.W3.ordinal()]; index < starts[GrpTensor.W3.ordinal() + 1]; index++) {
      weights[index] = 0;
    }
    weights[starts[GrpTensor.B3.ordinal()] + 1] = -800f;
    GrpNetwork far = GrpNetwork.of(shape, weights);
    GrpState state = new GrpState(variant, 4, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    far.permutations(state, workspace, probabilities);
    Assert.assertEquals(probabilities[1], 0.0, 0.0);
    double log = far.logProbability(state, 1, workspace);
    Assert.assertTrue(Double.isFinite(log) && log < -790, Double.toString(log));
  }

  @Test
  public void eachPhaseHasItsOwnHead() {
    Variant variant = Variant.TENHOU_4P;
    GrpShape shape = GrpFixtures.shape(variant, 8, true);
    float[] weights = GrpFixtures.network(shape, 29, 1.0).weights();
    int[] starts = GrpTensor.starts(shape);
    int outputs = shape.outputs();
    for (int index = starts[GrpTensor.W3.ordinal()]; index < starts[GrpTensor.W3.ordinal() + 1]; index++) {
      weights[index] = 0;
    }
    GrpNetwork network = GrpNetwork.of(shape, weights);
    GrpWorkspace workspace = GrpWorkspace.of(shape);
    double[] probabilities = new double[outputs];
    int[] rounds = {2, 5, 6, 9};
    for (GrpPhase phase : GrpPhase.values()) {
      int round = rounds[phase.ordinal()];
      Assert.assertEquals(GrpPhase.of(variant, round), phase);
      GrpState state = new GrpState(variant, round, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
      network.permutations(state, workspace, probabilities);
      int bias = starts[GrpTensor.B3.ordinal()] + phase.ordinal() * outputs;
      for (int k = 1; k < outputs; k++) {
        double ratio = probabilities[k] / probabilities[0];
        Assert.assertEquals(ratio, StrictMath.exp((double) weights[bias + k] - weights[bias]), 1e-9, phase + " " + k);
      }
    }
  }

  @Test
  public void weightsMustFitTheShapeAndBeFinite() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_3P, 4, false);
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpNetwork.of(shape, new float[3]));
    float[] weights = GrpFixtures.network(shape, 31, 1.0).weights();
    weights[7] = Float.NaN;
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpNetwork.of(shape, weights));
    weights[7] = Float.POSITIVE_INFINITY;
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpNetwork.of(shape, weights));
  }

  @Test
  public void networksKeepTheirOwnWeights() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_3P, 4, false);
    float[] weights = GrpFixtures.network(shape, 37, 1.0).weights();
    GrpNetwork network = GrpNetwork.of(shape, weights);
    float first = weights[0];
    weights[0] = first + 1;
    network.weights()[1] = 5;
    float[] kept = network.weights();
    Assert.assertEquals(kept[0], first);
    Assert.assertEquals(kept[1], weights[1]);
  }

  @Test
  public void workspacesMustMatchTheShape() {
    GrpShape shape = GrpFixtures.shape(Variant.TENHOU_4P, 8, false);
    GrpNetwork network = GrpFixtures.network(shape, 41, 1.0);
    GrpWorkspace other = GrpWorkspace.of(GrpFixtures.shape(Variant.TENHOU_4P, 9, false));
    GrpState state = new GrpState(Variant.TENHOU_4P, 0, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> network.phi(state, RankUtility.TENHOU, other, new double[4]));
    GrpWorkspace equal = GrpWorkspace.of(GrpFixtures.shape(Variant.TENHOU_4P, 8, false));
    network.phi(state, RankUtility.TENHOU, equal, new double[4]);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> network.phi(state, RankUtility.TENHOU, GrpWorkspace.of(shape), new double[3]));
  }
}
