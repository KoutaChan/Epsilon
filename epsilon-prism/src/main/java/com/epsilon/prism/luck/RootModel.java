package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.RootCoefficients;
import java.util.List;
import java.util.Objects;

/** 教師を作る世代より前に当てはめと選択を終えた、候補ごとの帳簿の係数。 */
public record RootModel(
    Weights weights,
    RootRates rates,
    int selectedThrough,
    @Key(doc = "保留世代で選んだ補正", presence = Presence.ABSENT) Choice choice,
    @Key(doc = "同じ前向き保留で選んだ教師の腕と損失", presence = Presence.ABSENT) RootHoldout.Selection selection) {
  /** 保留世代で選んだ補正。縮小強度が無限大または倍率が 0 なら、生の教師を使う。 */
  public enum Choice {
    FITTED,
    INFINITE_RIDGE,
    ZERO_RATE
  }

  public RootModel {
    Objects.requireNonNull(weights);
    Objects.requireNonNull(rates);
    int cohort = weights.validated();
    for (var fold : rates.folds()) {
      if (fold.cohort() <= cohort)
        throw new IllegalArgumentException("Ledger curve cohorts must increase");
      cohort = fold.cohort();
    }
    if (rates.folds().size() != weights.settings().curveCohorts()
        || selectedThrough < cohort + weights.settings().selectionCohorts())
      throw new IllegalArgumentException("Invalid ledger selection cohorts");
    if (selection != null
        && (selection.losses().first() <= cohort
            || selection.losses().through() != selectedThrough
            || selection.losses().cohorts() != weights.settings().selectionCohorts()
            || selection.arm() != selection.losses().best(weights.ridge() != null)))
      throw new IllegalArgumentException("Invalid forward root ledger arm selection");
    if (choice == null) choice = weights.ridge() == null ? Choice.INFINITE_RIDGE : Choice.FITTED;
    if ((choice == Choice.INFINITE_RIDGE) != (weights.ridge() == null)
        || choice == Choice.ZERO_RATE
            && rates.folds().stream().anyMatch(f -> !f.curves().isEmpty()))
      throw new IllegalArgumentException("Invalid root ledger choice");
  }

  /** 列ごとの線形係数。縮小強度が無いときは無限大を選び、補正は 0 になる。 */
  public record Weights(
      int version,
      Variant variant,
      RootCoefficients settings,
      List<LedgerColumn> columns,
      int first,
      int validated,
      @Key(doc = "次世代で選んだ有限の縮小強度", presence = Presence.ABSENT) Double ridge,
      List<Double> theta) {
    public static final int VERSION = 1;

    public Weights {
      columns = List.copyOf(columns);
      theta = List.copyOf(theta);
      if (version != VERSION || first < 1 || validated < first + settings.trainingCohorts()) {
        throw new IllegalArgumentException("Invalid root ledger coefficient version or cohort");
      }
      if (columns.isEmpty()
          || !columns.equals(columns.stream().distinct().sorted().toList())
          || settings.information() == RootCoefficients.Information.ROOT_HANDS
              && columns.stream().anyMatch(c -> !c.centeredGivenHands())) {
        throw new IllegalArgumentException("Invalid columns for the root ledger information set");
      }
      if (theta.size() != columns.size() * RootFeatures.Layout.of(variant, settings).width
          || theta.stream().anyMatch(x -> !Double.isFinite(x))) {
        throw new IllegalArgumentException("Invalid root ledger coefficient matrix");
      }
      if (ridge == null
          ? theta.stream().anyMatch(x -> x != 0)
          : ridge < 0 || !Double.isFinite(ridge)) {
        throw new IllegalArgumentException("Invalid root ledger shrinkage");
      }
    }
  }

  /** 根で共通の項を行につき 1 回計算し、各候補に固有の項を加える。 */
  static final class Predictor {
    final Weights weights;
    final RootRates rates;
    final int[] columns;
    final RootFeatures.Layout layout;
    final double[][] theta;

    Predictor(Weights weights, RootRates rates) {
      this.weights = weights;
      this.rates = rates;
      columns = weights.columns.stream().mapToInt(Enum::ordinal).toArray();
      layout = RootFeatures.Layout.of(weights.variant, weights.settings);
      theta = new double[columns.length][layout.width];
      int at = 0;
      for (double[] row : theta)
        for (int k = 0; k < row.length; k++) row[k] = weights.theta.get(at++);
    }

    double prepare(RootFeatures root, LedgerCell.Standing s, double[] base) {
      if (layout.hand >= 0 && !root.hands()) return 0;
      double h = rates == null ? 1 : rates.scale(s, weights.variant.seats());
      if (h == 0) return 0;
      for (int i = 0; i < columns.length; i++) {
        double[] row = theta[i], x = root.common();
        double b = 0;
        for (int j = 0; j < layout.common; j++) b += x[j] * row[j];
        if (layout.hand >= 0)
          for (int j = 0; j < layout.sharedHands; j++)
            b += x[layout.common + j] * row[layout.hand + j];
        base[i] = b;
      }
      return h;
    }

    double coefficient(RootFeatures root, int action, int col, double[] base) {
      double b = base[col];
      if (layout.types > 0) {
        double[] row = theta[col];
        b += row[layout.common + root.actionType(action)];
        if (layout.hand >= 0)
          b +=
              root.opponentsWaiting(action) * row[layout.at(RootFeatures.Field.WAITING_OPPONENTS)]
                  + root.dealInPoints(action) * row[layout.at(RootFeatures.Field.DEAL_IN)];
      }
      return b;
    }

    double amount(
        RootFeatures root,
        int index,
        FitRow row,
        int action,
        double[] base,
        double h,
        double[] contributions) {
      double sum = 0;
      for (int i = 0; i < columns.length; i++) {
        int c = columns[i];
        double term = h == 0 ? 0 : coefficient(root, index, i, base) * row.ledger(action, c);
        if (contributions != null) contributions[c] = h * term;
        sum += term;
      }
      return h * sum;
    }
  }
}
