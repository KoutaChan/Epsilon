package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.RootCoefficients;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** 凍結した候補別の補正を、行と半荘を保ったまま生の教師と比較する。 */
public final class RootAudit {
  /** 1 群の符号付き補正。半荘ごとの十分統計をこの反復と連続する反復全体について持つ。 */
  public record Signed(
      String group,
      long slots,
      long games,
      @Key(doc = "候補ごとの補正の平均", presence = Presence.ABSENT) Double mean,
      @Key(doc = "半荘ごとに束ねた平均の標準誤差", presence = Presence.ABSENT) Double se,
      @Key(doc = "この反復の半荘別の和", presence = Presence.ABSENT) LedgerReport.Paired cohort,
      @Key(doc = "同じ設定で監査した全反復の半荘別の和", presence = Presence.ABSENT) LedgerReport.Paired cumulative,
      @Key(doc = "全反復を合わせた平均の標準誤差", presence = Presence.ABSENT) Double cumulativeSe) {}

  /** 教師に適用した腕と比較の基準、根の係数の世代と選択状況。 */
  public record Status(
      LedgerReport.Arm teacher,
      LedgerReport.Arm comparison,
      @Key(doc = "学習履歴を始め直した理由", presence = Presence.ABSENT) String restart,
      @Key(doc = "係数を学び始めてからの反復数", presence = Presence.ABSENT) Integer age,
      @Key(doc = "次期に持ち越す凍結モデルの選択", presence = Presence.ABSENT) RootModel.Choice nextChoice,
      boolean pendingInfiniteRidge,
      @Key(doc = "当期に使える保留選択の腕と理由", presence = Presence.ABSENT) RootHoldout.Decision decision,
      @Key(doc = "次期の教師を選んだ保留での比較", presence = Presence.ABSENT)
          RootHoldout.Selection nextSelection) {}

  /** 根の係数の当てはめと、凍結済みの補正を同じ行で評価した結果。 */
  public record Report(
      RootCoefficients.Mode mode,
      boolean applied,
      int trainingFirst,
      long missingHands,
      @Key(doc = "評価した係数の版と倍率", presence = Presence.ABSENT) RootModel model,
      @Key(doc = "候補を単位にした決定係数", presence = Presence.ABSENT) Double r2,
      List<LedgerReport.Band> bands,
      List<Signed> signed,
      List<LedgerReport.Gate> gates,
      @Key(doc = "実際の教師と比較に使う腕", presence = Presence.ABSENT) Status status) {
    Report withGates(List<LedgerReport.Gate> observations) {
      return new Report(
          mode,
          applied,
          trainingFirst,
          missingHands,
          model,
          r2,
          bands,
          signed,
          observations,
          status);
    }
  }

  private final RootModel model;
  private final RootModel.Predictor predictor;
  private final LedgerCorrection current;
  private final Variant variant;
  private final LedgerCalibrator.Group[] bands;
  private final Map<Long, double[]> games = new HashMap<>();
  private final List<String> names = new ArrayList<>();
  private final int paths, types, cols;
  private final double[] base, parts = new double[LedgerColumn.COUNT];
  private double[] residual = new double[0];
  private double yy, ee;

  RootAudit(RootModel model, LedgerCorrection current) {
    this.model = model;
    this.current = current;
    predictor = new RootModel.Predictor(model.weights(), model.rates());
    variant = model.weights().variant();
    bands = LedgerCalibrator.bands(variant);
    base = new double[predictor.columns.length];
    names.add("all");
    paths = names.size();
    for (var s : Row.Source.values()) names.add(s.name());
    types = names.size();
    for (var a : Action.Type.values()) names.add(a.name());
    cols = names.size();
    for (var c : LedgerColumn.values()) names.add(c.reportName());
  }

  void add(FitRow row, RootFeatures root, int[] measured, int n, float[] y) {
    var standing = root.standing();
    double h = predictor.prepare(root, standing, base);
    if (residual.length < n) residual = new double[n];
    var game = games.computeIfAbsent(row.gameId(), unused -> new double[2 * names.size()]);
    var beta =
        current.coefficients(
            row.gameId(), row.decisionClass().ordinal(), current.byCell() == null ? 0 : row.cell());
    double scale = current.scale() == null ? 1 : current.scale().scale(standing);
    double my = 0, mb = 0, mc = 0, curSquare = 0;
    for (int j = 0; j < n; j++) {
      int a = measured[j];
      double b = predictor.amount(root, j, row, a, base, h, parts);
      my += y[a];
      mb += b;
      residual[j] = y[a] - b;
      double c = 0;
      for (int k = 0; k < beta.length; k++) c += beta[k] * row.ledger(a, k);
      double baseline = y[a] - scale * c;
      double d = baseline - mc;
      mc += d / (j + 1);
      curSquare += d * (baseline - mc);
      add(game, 0, b);
      add(game, paths + row.source().ordinal(), b);
      add(game, types + root.type(j).ordinal(), b);
      for (int col : predictor.columns) add(game, cols + col, parts[col]);
    }
    my /= n;
    mb /= n;
    double raw = 0, corrected = 0;
    for (int j = 0; j < n; j++) {
      double x = y[measured[j]] - my, e = residual[j] - (my - mb);
      raw += x * x;
      corrected += e * e;
    }
    yy += raw;
    ee += corrected;
    LedgerCalibrator.addBands(
        bands, variant, standing, row.gameId(), raw / n, curSquare / n, corrected / n);
  }

  private static void add(double[] g, int i, double b) {
    g[i * 2]++;
    g[i * 2 + 1] += b;
  }

  List<Signed> signed(Map<String, LedgerReport.Paired> previous) {
    List<Signed> signed = new ArrayList<>();
    for (int i = 0; i < names.size(); i++) {
      long slots = 0, ng = 0;
      double sum = 0;
      for (double[] g : games.values())
        if (g[2 * i] > 0) {
          ng++;
          slots += (long) g[2 * i];
          sum += g[2 * i + 1];
        }
      double mean = sum / slots, ss = 0, countResidual = 0, countSquares = 0;
      for (double[] g : games.values())
        if (g[2 * i] > 0) {
          double r = g[2 * i + 1] - mean * g[2 * i];
          ss += r * r;
          countResidual += g[2 * i] * r;
          countSquares += g[2 * i] * g[2 * i];
        }
      var cohort =
          slots == 0
              ? null
              : new LedgerReport.Paired(slots, ng, mean, countSquares, countResidual, ss);
      var cumulative = previous.get(names.get(i));
      if (cohort != null) cumulative = cumulative == null ? cohort : cumulative.plus(cohort);
      signed.add(
          new Signed(
              names.get(i),
              slots,
              ng,
              LedgerReport.finite(mean),
              cohort == null ? null : cohort.se(),
              cohort,
              cumulative,
              cumulative == null ? null : cumulative.se()));
    }
    return List.copyOf(signed);
  }

  Report report(
      RootCoefficients.Mode mode, int first, long missing, Status status, List<Signed> signed) {
    return new Report(
        mode,
        status.teacher() == LedgerReport.Arm.ROOT,
        first,
        missing,
        model,
        LedgerReport.finite(1 - ee / yy),
        Arrays.stream(bands).map(group -> group.band(true)).toList(),
        signed,
        List.of(),
        status);
  }
}
