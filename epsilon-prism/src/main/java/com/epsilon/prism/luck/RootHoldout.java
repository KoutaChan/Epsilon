package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.luck.LedgerReport.Arm;
import com.epsilon.prism.noise.LedgerCalibration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** 倍率の仕様と教師の腕を、同じ保留世代の行内分散で選ぶ。 */
public final class RootHoldout {
  /** 同じ半荘で測った二つの腕の損失差と標準誤差。 */
  public record Difference(
      LedgerReport.Paired paired, @Key(doc = "対応差の標準誤差", presence = Presence.ABSENT) Double se) {
    static Difference of(LedgerReport.Paired paired) {
      return new Difference(paired, paired.se());
    }

    Difference plus(Difference other) {
      return of(paired.plus(other.paired));
    }
  }

  /** 実際に比較した世代、各腕の平均損失と、腕どうしの対応差。 */
  public record Scores(
      int first,
      int through,
      int cohorts,
      double raw,
      double root,
      @Key(doc = "全保留世代に共通する比較腕", presence = Presence.ABSENT) Arm incumbent,
      @Key(doc = "全保留世代で使えた共通倍率の設定", presence = Presence.ABSENT) LedgerCalibration.Apply reference,
      @Key(doc = "比較腕の平均損失", presence = Presence.ABSENT) Double baseline,
      Difference rootMinusRaw,
      @Key(doc = "根の腕から比較腕を引いた損失差", presence = Presence.ABSENT) Difference rootMinusBaseline,
      @Key(doc = "比較腕から生を引いた損失差", presence = Presence.ABSENT) Difference baselineMinusRaw) {
    Scores plus(Scores next) {
      if (through >= next.first)
        throw new IllegalArgumentException("Ledger holdout cohorts overlap");
      var rr = rootMinusRaw.plus(next.rootMinusRaw);
      double fraction = (double) next.rootMinusRaw.paired.rows() / rr.paired.rows();
      boolean comparable =
          incumbent != null
              && incumbent == next.incumbent
              && Objects.equals(reference, next.reference);
      return new Scores(
          first,
          next.through,
          cohorts + next.cohorts,
          raw + (next.raw - raw) * fraction,
          root + (next.root - root) * fraction,
          comparable ? incumbent : null,
          comparable ? reference : null,
          comparable ? baseline + (next.baseline - baseline) * fraction : null,
          rr,
          comparable ? rootMinusBaseline.plus(next.rootMinusBaseline) : null,
          comparable ? baselineMinusRaw.plus(next.baselineMinusRaw) : null);
    }

    Arm best(boolean finiteRidge) {
      if (incumbent == null) return Arm.CLASS;
      if (!finiteRidge) return incumbent;
      if (rootMinusRaw.paired.mean() < 0 && rootMinusBaseline.paired.mean() < 0) return Arm.ROOT;
      return baselineMinusRaw.paired.mean() <= 0 ? incumbent : Arm.RAW;
    }

    Scores zeroRate() {
      var p = rootMinusRaw.paired;
      var zero =
          Difference.of(new LedgerReport.Paired(p.rows(), p.games(), 0, p.countSquares(), 0, 0));
      var c = baselineMinusRaw == null ? null : baselineMinusRaw.paired;
      var reverse =
          c == null
              ? null
              : Difference.of(
                  new LedgerReport.Paired(
                      c.rows(),
                      c.games(),
                      -c.mean(),
                      c.countSquares(),
                      -c.countResidual(),
                      c.residualSquares()));
      return new Scores(
          first,
          through,
          cohorts,
          raw,
          raw,
          incumbent,
          reference,
          baseline,
          zero,
          reverse,
          baselineMinusRaw);
    }
  }

  /** 教師を選ぶために凍結した判定。 */
  public record Selection(Arm arm, Scores losses) {}

  /** 当期に使える腕と、その腕を使う理由。 */
  public record Decision(Arm arm, String reason) {}

  static Decision decide(
      RootModel model, LedgerCalibration.Apply reference, boolean scaleAvailable) {
    Arm fallback = reference != null && scaleAvailable ? Arm.COMMON_SCALE : Arm.CLASS;
    if (model == null) return new Decision(fallback, "no frozen root model");
    if (model.choice() == RootModel.Choice.INFINITE_RIDGE)
      return new Decision(fallback, "infinite root ridge");
    Selection selected = model.selection();
    if (selected == null) return new Decision(fallback, "no complete forward arm comparison");
    if (selected.losses.incumbent != fallback
        || !Objects.equals(reference, selected.losses.reference))
      return new Decision(
          fallback, "the active ledger correction differs from the forward comparison");
    if (reference != null && !scaleAvailable)
      return new Decision(fallback, "the preceding common scale is unavailable");
    return new Decision(selected.arm, "lowest forward holdout loss");
  }

  private static final class Game {
    long count;
    double raw, baselineDelta;
    final double[] rootDelta;

    Game(int specs) {
      rootDelta = new double[specs];
    }
  }

  private final int cohort, seats;
  private final List<RootTraining.Candidate> specs;
  private final LedgerCalibration.Apply reference;
  private final LedgerCorrection baseline;
  private final Map<LedgerCell.Standing, double[]> rates = new HashMap<>();
  private final Map<Long, Game> games = new HashMap<>();
  private final Game total;

  RootHoldout(
      int cohort,
      int seats,
      List<RootTraining.Candidate> specs,
      LedgerCalibration.Apply reference,
      LedgerCorrection baseline) {
    this.cohort = cohort;
    this.seats = seats;
    this.specs = specs;
    this.reference = reference;
    this.baseline = baseline;
    total = new Game(specs.size());
  }

  void add(
      FitRow row,
      LedgerCell.Standing standing,
      float[] y,
      int[] measured,
      int n,
      double meanY,
      double a,
      double b) {
    double raw = 0, baselineDelta = 0, mean = 0;
    double[] beta =
        baseline == null
            ? null
            : baseline.coefficients(
                row.gameId(),
                row.decisionClass().ordinal(),
                baseline.byCell() == null ? 0 : row.cell());
    double scale =
        baseline == null || baseline.scale() == null ? 1 : baseline.scale().scale(standing);
    for (int j = 0; j < n; j++) {
      int index = measured[j];
      double d = y[index] - meanY;
      raw += d * d;
      if (beta != null) {
        double corrected = y[index];
        double amount = 0;
        for (int c = 0; c < beta.length; c++) amount += beta[c] * row.ledger(index, c);
        corrected -= scale * amount;
        double delta = corrected - mean;
        mean += delta / (j + 1);
        baselineDelta += delta * (corrected - mean);
      }
    }
    baselineDelta = (baselineDelta - raw) / n;
    raw /= n;
    Game game = games.computeIfAbsent(row.gameId(), unused -> new Game(specs.size()));
    game.count++;
    total.count++;
    game.raw += raw;
    total.raw += raw;
    game.baselineDelta += baselineDelta;
    total.baselineDelta += baselineDelta;
    double[] h =
        rates.computeIfAbsent(
            standing, s -> specs.stream().mapToDouble(c -> c.rates().scale(s, seats)).toArray());
    for (int i = 0; i < h.length; i++) {
      double d = h[i] * (h[i] * a - 2 * b);
      game.rootDelta[i] += d;
      total.rootDelta[i] += d;
    }
  }

  Scores scores(int spec) {
    double raw = total.raw / total.count;
    double rd = total.rootDelta[spec] / total.count, cd = total.baselineDelta / total.count;
    Arm incumbent = baseline == null ? null : (reference == null ? Arm.CLASS : Arm.COMMON_SCALE);
    return new Scores(
        cohort,
        cohort,
        1,
        raw,
        raw + rd,
        incumbent,
        incumbent == null ? null : reference,
        incumbent == null ? null : raw + cd,
        difference(spec, Arm.ROOT, Arm.RAW, rd),
        incumbent == null ? null : difference(spec, Arm.ROOT, incumbent, rd - cd),
        incumbent == null ? null : difference(spec, incumbent, Arm.RAW, cd));
  }

  private Difference difference(int spec, Arm left, Arm right, double mean) {
    double nn = 0, nr = 0, rr = 0;
    for (Game g : games.values()) {
      double delta =
          (left == Arm.ROOT ? g.rootDelta[spec] : g.baselineDelta)
              - (right == Arm.RAW ? 0 : g.baselineDelta);
      double residual = delta - mean * g.count;
      nn += (double) g.count * g.count;
      nr += g.count * residual;
      rr += residual * residual;
    }
    return Difference.of(new LedgerReport.Paired(total.count, games.size(), mean, nn, nr, rr));
  }
}
