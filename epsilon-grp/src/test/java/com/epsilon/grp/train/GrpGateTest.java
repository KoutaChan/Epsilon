package com.epsilon.grp.train;

import com.epsilon.grp.GrpField;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpTensor;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 関門の判定を確かめる。再学習は差が +2 SE 以内、入れ替えは −2 SE より良いことを求める。モデルを戻した後は、戻したモデルとの差も
 * +1 SE 以内を求める。単調性の違反率がしきい値以内か、値が有限かも見る。
 */
public class GrpGateTest {

  private static final double SE = 1e-5;

  @Test
  public void refitsMustNotBeWorseByMoreThanTwoStandardErrors() {
    Assert.assertTrue(judge(GateKind.REFIT, 1.9e-5, 0).passed());
    Assert.assertTrue(judge(GateKind.REFIT, 2e-5, 0).passed());
    Assert.assertTrue(judge(GateKind.REFIT, -5e-5, 0).passed());
    Assert.assertEquals(judge(GateKind.REFIT, 2.1e-5, 0).failures(), List.of(GateFailure.WORSE));
  }

  @Test
  public void replacementsMustBeBetterByTwoStandardErrors() {
    Assert.assertTrue(judge(GateKind.REPLACEMENT, -2.1e-5, 0).passed());
    Assert.assertEquals(judge(GateKind.REPLACEMENT, -2e-5, 0).failures(), List.of(GateFailure.NOT_BETTER));
    Assert.assertEquals(judge(GateKind.REPLACEMENT, 0, 0).failures(), List.of(GateFailure.NOT_BETTER));
  }

  @Test
  public void afterARollbackTheAnchorAllowsOneStandardError() {
    Paired used = new Paired(0, SE);
    Assert.assertTrue(GrpGate.judge(GateKind.REFIT, true, used, new Paired(0.9e-5, SE), 0, 0).passed());
    GateReport worse = GrpGate.judge(GateKind.REFIT, true, used, new Paired(1.1e-5, SE), 0, 0);
    Assert.assertEquals(worse.failures(), List.of(GateFailure.WORSE_THAN_ANCHOR));
  }

  @Test
  public void monotoneLineFollowsTheUsedModelAboveAFloor() {
    GateReport floor = GrpGate.judge(GateKind.REFIT, true, new Paired(0, SE), null, 0.001, 0.005);
    Assert.assertEquals(floor.monotoneLine(), 0.005);
    Assert.assertTrue(floor.passed());
    GateReport scaled = GrpGate.judge(GateKind.REFIT, true, new Paired(0, SE), null, 0.004, 0.0061);
    Assert.assertEquals(scaled.monotoneLine(), 0.006, 1e-15);
    Assert.assertEquals(scaled.failures(), List.of(GateFailure.NOT_MONOTONE));
    Assert.assertTrue(GrpGate.judge(GateKind.REFIT, true, new Paired(0, SE), null, 0.004, 0.0059).passed());
  }

  @Test
  public void brokenValuesFailEveryLineTheyTouch() {
    GateReport report = GrpGate.judge(GateKind.REFIT, false, new Paired(Double.NaN, Double.NaN), null, 0, Double.NaN);
    Assert.assertEquals(
        report.failures(), List.of(GateFailure.NOT_FINITE, GateFailure.WORSE, GateFailure.NOT_MONOTONE));
    GateReport all = GrpGate.judge(GateKind.REPLACEMENT, false, new Paired(1, SE), new Paired(1, SE), 0, 1);
    Assert.assertEquals(all.failures(), List.of(GateFailure.NOT_FINITE, GateFailure.NOT_BETTER,
        GateFailure.WORSE_THAN_ANCHOR, GateFailure.NOT_MONOTONE));
  }

  @Test
  public void gatesCompareTheCandidateWithTheUsedPhiOnTheGateCohort() {
    HandEnds ends = HandEnds.of(
        Variant.TENHOU_4P, RankUtility.TENHOU, TrainFixtures.uniform(4_000, 6, new SplittableRandom(31)));
    HandEndValues calibrated = shifted(ends, 0);
    HandEndValues off = shifted(ends, 0.1);
    Assert.assertTrue(GrpGate.refit(ends, calibrated, calibrated, 0, 0).passed());
    Assert.assertEquals(GrpGate.refit(ends, calibrated, off, 0, 0).failures(), List.of(GateFailure.WORSE));
    Assert.assertTrue(GrpGate.replacement(ends, off, calibrated, 0, 0).passed());
    Assert.assertEquals(
        GrpGate.replacement(ends, calibrated, calibrated, 0, 0).failures(), List.of(GateFailure.NOT_BETTER));
    GateReport anchored = GrpGate.refit(ends, off, calibrated, off, 0, 0);
    Assert.assertTrue(anchored.passed(), anchored.toString());
    Assert.assertTrue(anchored.versusAnchor().mean() < 0);
    GateReport beyondAnchor = GrpGate.refit(ends, off, off, calibrated, 0, 0);
    Assert.assertEquals(beyondAnchor.failures(), List.of(GateFailure.WORSE_THAN_ANCHOR));
  }

  @Test
  public void probesCountScoreMovesThatPushPhiTheWrongWay() {
    HandEnds ends = HandEnds.of(
        Variant.TENHOU_4P, RankUtility.TENHOU, TrainFixtures.simulated(Variant.TENHOU_4P, 400, 0, 37));
    MonotoneProbe probe = MonotoneProbe.of(ends);
    Assert.assertEquals(probe.samples(), Math.min(4_096, ends.handEnds()));
    GrpNetwork forward = firstPlaceByScore(1);
    GrpNetwork backward = firstPlaceByScore(-1);
    GrpWorkspace workspace = GrpWorkspace.of(forward.shape());
    Assert.assertEquals(probe.violationRate(forward, workspace), 0.0);
    Assert.assertEquals(probe.violationRate(backward, workspace), 1.0);
    float[] flat = forward.weights();
    int[] starts = GrpTensor.starts(forward.shape());
    Arrays.fill(flat, starts[GrpTensor.W3.ordinal()], starts[GrpTensor.W3.ordinal() + 1], 0f);
    Assert.assertEquals(probe.violationRate(GrpNetwork.of(forward.shape(), flat), workspace), 0.0);
    MonotoneProbe few = MonotoneProbe.of(ends, 10, 1);
    Assert.assertEquals(few.samples(), 10);
    Assert.assertEquals(MonotoneProbe.of(ends, 10, 1).violationRate(backward, workspace), 1.0);
  }

  /**
   * GrpNetwork 以外の Φ も、{@link MonotoneProbe#moves()} の卓で求めた値を渡せば測れる。卓は選んだ局末ごとに並ぶ。先頭が元の卓
   * （受け取る席も渡す席も −1）で、その後に点数を動かした卓が続く。点数は和を変えずに動かし、渡す席は 0 点以上に残す。ネットワークの
   * 値を渡せば、ネットワークで測った率と同じになる。
   */
  @Test
  public void probesTakeTheValuesOfAnyPhi() {
    HandEnds ends = HandEnds.of(
        Variant.TENHOU_4P, RankUtility.TENHOU, TrainFixtures.simulated(Variant.TENHOU_4P, 200, 0, 41));
    MonotoneProbe probe = MonotoneProbe.of(ends, 50, 3);
    GrpNetwork backward = firstPlaceByScore(-1);
    GrpWorkspace workspace = GrpWorkspace.of(backward.shape());
    List<MonotoneProbe.Move> moves = probe.moves();
    double[][] values = new double[moves.size()][4];
    int bases = 0;
    MonotoneProbe.Move base = null;
    for (int index = 0; index < moves.size(); index++) {
      MonotoneProbe.Move move = moves.get(index);
      backward.phi(move.state(), RankUtility.TENHOU, workspace, values[index]);
      if (move.receiver() < 0) {
        Assert.assertEquals(move.giver(), -1);
        Assert.assertEquals(move.state(), ends.state(move.handEnd()));
        base = move;
        bases++;
        continue;
      }
      Assert.assertEquals(move.handEnd(), base.handEnd());
      int moved = move.state().score(move.receiver()) - base.state().score(move.receiver());
      Assert.assertEquals(base.state().score(move.giver()) - move.state().score(move.giver()), moved);
      Assert.assertTrue(move.state().score(move.giver()) >= 0, move.toString());
    }
    Assert.assertEquals(bases, probe.samples());
    Assert.assertEquals(probe.violationRate(values), probe.violationRate(backward, workspace));
    Assert.assertThrows(IllegalArgumentException.class, () -> probe.violationRate(new double[1][4]));
  }

  /**
   * 入力が点数のフィールドだけのネットワークを作る。1 つ目の隠れ層は h = 点数のフィールド ＋ 10 で、2 つ目の隠れ層は h をそのまま
   * 通す。出力は z_π = a × h[π で 1 着になる席]。
   *
   * @param slope 傾き a。正なら点数が増えた席ほど 1 着の確率が上がり（単調）、負なら逆になる
   */
  private static GrpNetwork firstPlaceByScore(float slope) {
    GrpShape shape = new GrpShape(Variant.TENHOU_4P, EnumSet.of(GrpField.SCORE), 4, false);
    int[] starts = GrpTensor.starts(shape);
    float[] weights = new float[starts[GrpTensor.values().length]];
    for (int seat = 0; seat < 4; seat++) {
      weights[starts[GrpTensor.W1.ordinal()] + seat * 4 + seat] = 1;
      weights[starts[GrpTensor.B1.ordinal()] + seat] = 10;
      weights[starts[GrpTensor.W2.ordinal()] + seat * 4 + seat] = 1;
    }
    RankPermutations table = RankPermutations.of(Variant.TENHOU_4P);
    for (int permutation = 0; permutation < table.count(); permutation++) {
      for (int seat = 0; seat < 4; seat++) {
        if (table.rank(permutation, seat) == 0) {
          weights[starts[GrpTensor.W3.ordinal()] + seat * table.count() + permutation] = slope;
        }
      }
    }
    return GrpNetwork.of(shape, weights);
  }

  private static GateReport judge(GateKind kind, double difference, double candidateMonotone) {
    return GrpGate.judge(kind, true, new Paired(difference, SE), null, 0, candidateMonotone);
  }

  /**
   * 全ての席の Φ を同じ値にし、周辺確率を一様にした局末の値を作る。
   *
   * @param shift 全ての席の Φ。最終順位は一様なので、0 なら較正された値になる
   */
  private static HandEndValues shifted(HandEnds ends, double shift) {
    HandEndValues.Builder builder = HandEndValues.builder(ends);
    double[] uniform = new double[16];
    Arrays.fill(uniform, 0.25);
    double[] phi = new double[4];
    Arrays.fill(phi, shift);
    for (int h = 0; h < ends.handEnds(); h++) {
      builder.set(h, phi, uniform);
    }
    return builder.build();
  }
}
