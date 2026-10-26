package com.epsilon.prism.learn;

import static com.epsilon.prism.core.DecisionClass.CALL;
import static com.epsilon.prism.core.DecisionClass.DISCARD_EARLY;
import static com.epsilon.prism.core.DecisionClass.DISCARD_MID;
import static com.epsilon.prism.core.DecisionClass.KAN;
import static com.epsilon.prism.core.DecisionClass.RIICHI;
import static com.epsilon.prism.core.DecisionClass.THREAT;
import static com.epsilon.prism.rows.PrismRow.Group.A;
import static com.epsilon.prism.rows.PrismRow.Group.B;

import com.epsilon.core.Action;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.rows.ObservationPayload;
import com.epsilon.prism.rows.PrismRow;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 交差した認証の推定量（{@link Certification.Estimator}・{@link Certification#report}）を、答えを手で計算できる根の記録で確かめる。網も入力の復号も使わない。
 * Î は判断あたりの改善（Σ term / Σ 1/β）。
 *
 * <p>合法手は 0: 1m、1: 5m（手出し）、2: 5m（ツモ切り。1 の双子）、3: 2p。
 */
public class CertificationTest {
  private static final List<Action> LEGAL =
      List.of(
          Action.dahai(0),
          Action.dahai(4, false, false),
          Action.dahai(4, false, true),
          Action.dahai(10));

  /** 根の記録。d = played なら played の 1 枠（y_br = y_main）。 */
  private static PrismRow root(
      long game,
      DecisionClass decisionClass,
      PrismRow.Group group,
      int played,
      int deviation,
      float yMain,
      float yBranch,
      float q,
      float beta,
      int... support) {
    float[] values = new float[LEGAL.size()];
    Arrays.fill(values, Float.NaN);
    values[played] = yMain;
    if (deviation != played) values[deviation] = yBranch;
    float[] prior = new float[LEGAL.size()];
    Arrays.fill(prior, 1f / LEGAL.size());
    byte[] bytes = new byte[support.length];
    for (int i = 0; i < support.length; i++) bytes[i] = (byte) support[i];
    return new PrismRow(
        1,
        game,
        game * 100 + played * 10 + deviation,
        0,
        Kind.DISCARD,
        Stage.MID,
        decisionClass,
        PrismRow.Purpose.TRAIN,
        group,
        PrismRow.Source.ROOT,
        played,
        LEGAL.size(),
        values,
        prior,
        bytes,
        deviation,
        q,
        beta * q,
        -1,
        ObservationPayload.takeOwnership(new byte[] {0}));
  }

  /** 生徒の s（target の手が最大）。 */
  private static float[] prefer(int target) {
    float[] scores = {0f, 0f, 0f, 0f};
    scores[target] = 1f;
    return scores;
  }

  /** (y_br − y_main)/p を推定量と同じ精度（float の値を double にして）で。 */
  private static double term(PrismRow row) {
    return ((double) row.deviationValue() - (double) row.playedValue()) / (double) row.propensity();
  }

  /** 根 1 つが代表する判断の数 1/β = q/p（推定量と同じ式）。 */
  private static double weight(PrismRow row) {
    return (double) row.deviationProbability() / row.propensity();
  }

  /** 半荘で束ねた標準誤差: G/(G−1)·Σ(T_g − Î·n_g)² の平方根を Σn_g で割る（n_g は半荘の重みの和）。 */
  private static double clustered(double[] totals, double[] weights) {
    double n = 0, sum = 0;
    for (int g = 0; g < totals.length; g++) {
      n += weights[g];
      sum += totals[g];
    }
    double mean = sum / n, squares = 0;
    for (int g = 0; g < totals.length; g++) {
      double residual = totals[g] - mean * weights[g];
      squares += residual * residual;
    }
    int groups = totals.length;
    return Math.sqrt(groups / (groups - 1.0) * squares) / n;
  }

  @Test
  public void horvitzThompsonOnTheOtherGroupsRoots() {
    // 生徒 A を組 B の根で測る（根は b が貪欲の手を打った判断だけなので、played = π_t(o)）
    PrismRow r1 = root(1, RIICHI, B, 0, 3, 0.10f, 0.30f, 0.5f, 0.04f, 0, 3);
    // d = played（写しなし）。A は乗り換えるが d と違うので項は 0。A の手 3 は C の外
    PrismRow r2 = root(1, CALL, B, 0, 0, 0.20f, 0.20f, 0.4f, 0.04f, 0, 1);
    // A の手 2 は played（1）の双子なので乗り換えではない
    PrismRow r3 = root(2, THREAT, B, 1, 3, 0.50f, 0.70f, 0.3f, 0.04f, 1, 3);
    // A の手 2 は d（1）の双子なので当たり
    PrismRow r4 = root(2, KAN, B, 0, 1, 0.50f, 0.45f, 0.25f, 0.04f, 0, 1, 3);
    // 打牌の類（確約の類に入らない。β が違うので 1 つの根が代表する判断の数も違う）。A は乗り換えない
    PrismRow r5 = root(3, DISCARD_MID, B, 0, 3, 0.00f, 0.10f, 0.5f, 0.002f, 0, 3);
    var byA = new Certification.Estimator();
    byA.add(r1, LEGAL, prefer(3));
    byA.add(r2, LEGAL, prefer(3));
    byA.add(r3, LEGAL, prefer(2));
    byA.add(r4, LEGAL, prefer(2));
    byA.add(r5, LEGAL, prefer(0));
    Certification.Side a = byA.side("A", B);

    double t1 = term(r1), t4 = term(r4);
    double w1 = weight(r1), w2 = weight(r2), w3 = weight(r3), w4 = weight(r4), w5 = weight(r5);
    Assert.assertEquals(t1, 10.0, 1e-5);
    Assert.assertEquals(t4, -5.0, 1e-4);
    for (double w : new double[] {w1, w2, w3, w4}) Assert.assertEquals(w, 25.0, 1e-5);
    Assert.assertEquals(w5, 500.0, 1e-3);
    Certification.Estimate all = a.all();
    Assert.assertEquals(all.roots(), 5);
    Assert.assertEquals(all.decisions(), w1 + w2 + w3 + w4 + w5, 1e-9);
    Assert.assertEquals(all.games(), 3);
    Assert.assertEquals(all.switches(), 3);
    Assert.assertEquals(all.hits(), 2);
    Assert.assertEquals(all.outside(), 1);
    // 全体は判断の数で重み付けした平均
    Assert.assertEquals(all.value(), (t1 + t4) / (w1 + w2 + w3 + w4 + w5), 1e-12);
    Assert.assertEquals(
        all.error(),
        clustered(new double[] {t1, t4, 0}, new double[] {w1 + w2, w3 + w4, w5}),
        1e-12);
    Certification.Estimate commitment = a.commitment();
    Assert.assertEquals(commitment.roots(), 4);
    Assert.assertEquals(commitment.games(), 2);
    Assert.assertEquals(commitment.outside(), 1);
    double commitA = (t1 + t4) / (w1 + w2 + w3 + w4);
    Assert.assertEquals(commitment.value(), commitA, 1e-12);
    // β が 1 つの類では、判断あたりの改善 = Σ hit·(y_br − y_main)/q / 根の数
    Assert.assertEquals(commitA, ((0.30 - 0.10) / 0.5 + (0.45 - 0.50) / 0.25) / 4, 1e-6);
    double errorA = clustered(new double[] {t1, t4}, new double[] {w1 + w2, w3 + w4});
    Assert.assertEquals(commitment.error(), errorA, 1e-12);
    Assert.assertEquals(
        a.byClass().keySet(), Set.of("RIICHI", "CALL", "THREAT", "KAN", "DISCARD_MID"));
    Assert.assertEquals(a.byClass().get("CALL").value(), 0.0, 0.0, "d = played の根の項は 0");
    Assert.assertEquals(a.byClass().get("RIICHI").value(), t1 / w1, 1e-12);
    Assert.assertTrue(
        Double.isNaN(a.byClass().get("RIICHI").error()), "半荘が 1 つなら se は NaN");

    // 生徒 B を組 A の根で測る（2 つの半荘で 1 つずつ当たる）
    PrismRow r6 = root(10, CALL, A, 3, 0, 0.20f, 0.25f, 0.5f, 0.04f, 0, 3);
    PrismRow r7 = root(11, RIICHI, A, 0, 3, 0.40f, 0.38f, 0.5f, 0.04f, 0, 3);
    var byB = new Certification.Estimator();
    byB.add(r6, LEGAL, prefer(0));
    byB.add(r7, LEGAL, prefer(3));
    Certification.Side b = byB.side("B", A);
    double t6 = term(r6), t7 = term(r7), w6 = weight(r6), w7 = weight(r7);
    double commitB = (t6 + t7) / (w6 + w7);
    Assert.assertEquals(b.commitment().value(), commitB, 1e-12);
    double errorB = clustered(new double[] {t6, t7}, new double[] {w6, w7});
    Assert.assertEquals(b.commitment().error(), errorB, 1e-12);

    Certification.Report report = Certification.report(a, b, 0);
    double mean = (commitA + commitB) / 2;
    double error = Math.sqrt(errorA * errorA + errorB * errorB) / 2;
    Assert.assertEquals(report.commitmentMean(), mean, 1e-12);
    Assert.assertEquals(report.commitmentError(), error, 1e-12);
    Assert.assertEquals(report.lowerBound(), mean - 2 * error, 1e-12);
    // 確約の類の乗り換え 3 + 2 のうち C の外は 1
    Assert.assertEquals(report.outsideShare(), 0.2, 1e-12);
    Assert.assertEquals(report.passes(0.0005, 1), mean - 2 * error > -0.0005);
    Assert.assertTrue(report.passes(-(mean - 2 * error) + 1e-9, 1));
    Assert.assertFalse(report.passes(-(mean - 2 * error) - 1e-9, 1));
    Assert.assertTrue(report.passes(1e9, 0.2));
    Assert.assertFalse(report.passes(1e9, 0.1), "C の外の乗り換えが多ければ下限が良くても通さない");
  }

  /** π_t と同じ手を選ぶ生徒（T = played）は、どの根でも乗り換えず Î = 0。 */
  @Test
  public void studentEqualToTheBehaviourScoresZero() {
    var estimator = new Certification.Estimator();
    estimator.add(root(1, RIICHI, B, 0, 3, 0.10f, 0.30f, 0.5f, 0.04f, 0, 3), LEGAL, prefer(0));
    estimator.add(root(2, CALL, B, 3, 0, 0.20f, 0.90f, 0.5f, 0.04f, 0, 3), LEGAL, prefer(3));
    estimator.add(root(3, KAN, B, 1, 3, 0.50f, 0.10f, 0.5f, 0.04f, 1, 3), LEGAL, prefer(2));
    Certification.Estimate commitment = estimator.side("A", B).commitment();
    Assert.assertEquals(commitment.switches(), 0);
    Assert.assertEquals(commitment.hits(), 0);
    Assert.assertEquals(commitment.value(), 0.0, 0.0);
    Assert.assertEquals(commitment.error(), 0.0, 0.0);
  }

  /** どちらかの組に確約の根がなければ下限は NaN で、判定は不合格。ラベル行は根として受け付けない。 */
  @Test
  public void missingRootsFailTheCertification() {
    var byA = new Certification.Estimator();
    byA.add(root(1, DISCARD_EARLY, B, 0, 3, 0f, 0.1f, 0.5f, 0.002f, 0, 3), LEGAL, prefer(3));
    Certification.Report report =
        Certification.report(byA.side("A", B), new Certification.Estimator().side("B", A), 0);
    Assert.assertTrue(Double.isNaN(report.lowerBound()));
    Assert.assertFalse(report.passes(1e9, 1));
    PrismRow root = root(2, CALL, B, 0, 3, 0f, 0.1f, 0.5f, 0.04f, 0, 3);
    PrismRow label =
        new PrismRow(
            root.iteration(),
            root.gameId(),
            root.decisionId(),
            root.seat(),
            root.kind(),
            root.stage(),
            root.decisionClass(),
            root.purpose(),
            root.group(),
            PrismRow.Source.MAIN,
            root.played(),
            root.actions(),
            root.values(),
            root.prior(),
            root.support(),
            -1,
            Float.NaN,
            Float.NaN,
            -1,
            root.payload());
    Assert.assertThrows(IllegalArgumentException.class, () -> byA.add(label, LEGAL, prefer(3)));
  }
}
