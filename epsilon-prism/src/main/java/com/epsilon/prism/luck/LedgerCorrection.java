package com.epsilon.prism.luck;

import com.epsilon.prism.rows.Row;
import java.util.Arrays;

/** 凍結した帳簿の係数を、保存行の教師または同じ根から分岐した候補に適用する。 */
public sealed interface LedgerCorrection {
  /** 測定した候補をそれぞれ補正し、未測定の候補には NaN を残す。 */
  float[] values(Row row);

  /** 和了と比較手の補正後の教師差を返す。 */
  default double difference(Row row, int win, int other, LedgerCell.Standing standing) {
    float[] corrected = values(row);
    return (double) corrected[win] - corrected[other];
  }

  /** 分岐後の各経路の帳簿に掛ける係数。候補を指定した順に返す。 */
  double[][] atRoot(long game, int group, long[] words, int base, long[] hidden, int... actions);

  /** 分岐の根で全員の手牌を記録する必要があるか。 */
  default boolean needsHands() {
    return false;
  }

  /** 判断クラスと局面の区分で係数を共有する補正。 */
  record ByClass(double[][][] beta, LedgerCell cells, double[][][][] byCell, LedgerCalibrator scale)
      implements LedgerCorrection {

    public ByClass {
      if ((cells == null) != (byCell == null))
        throw new IllegalArgumentException(
            "Ledger cell coefficients need their cells and vice versa");
    }

    public ByClass(double[][][] beta, LedgerCell cells, double[][][][] byCell) {
      this(beta, cells, byCell, null);
    }

    public ByClass(double[][][] beta) {
      this(beta, null, null);
    }

    /** 同じ係数に補正率を掛けた補正を返す。 */
    public ByClass scaled(LedgerCalibrator scale) {
      return new ByClass(beta, cells, byCell, scale);
    }

    @Override
    public double difference(Row row, int win, int other, LedgerCell.Standing standing) {
      double[] b =
          coefficients(
              row.gameId(),
              row.decisionClass().ordinal(),
              cells == null ? 0 : cells.of(row.observation(), 0));
      double delta = 0;
      for (int f = 0; f < b.length; f++)
        delta += b[f] * ((double) row.ledger(win, f) - row.ledger(other, f));
      double h = scale == null ? 1 : scale.scale(standing);
      return (double) row.values()[win] - row.values()[other] - h * delta;
    }

    /** 指定した偶奇と判断クラスの係数。局・順位の係数があればその区分を使う。 */
    public double[] coefficients(long gameId, int group, int cell) {
      int fold = (int) (gameId & 1);
      if (byCell != null && byCell[fold][group][cell] != null) return byCell[fold][group][cell];
      return beta[fold][group];
    }

    private static final int COLUMNS = LedgerColumn.COUNT;

    /**
     * 補正した Y を返す。測った手は Y_a − β·L_a（補正率があれば Y_a − h(o)·β·L_a）にし、測らない手は NaN のままにする。帳簿のない行と
     * 帳簿の欠けた行は {@code row.values()} をそのまま返す。
     */
    public float[] values(Row row) {
      LedgerCell reader = cells != null || scale == null ? cells : LedgerCell.of(scale.variant());
      return values(FitRow.of(row, reader));
    }

    float[] values(FitRow row) {
      float[] values = row.values();
      if (row.ledgerFeatures() != COLUMNS) return values;
      double[] coefficients =
          coefficients(
              row.gameId(), row.decisionClass().ordinal(), byCell == null ? 0 : row.cell());
      double h = scale == null ? 1 : scale.scale(row.standing());
      float[] corrected = new float[values.length];
      for (int a = 0; a < values.length; a++) {
        if (Float.isNaN(values[a])) {
          corrected[a] = Float.NaN;
          continue;
        }
        double value = values[a];
        for (int f = 0; f < COLUMNS; f++) {
          float term = row.ledger(a, f);
          if (!Float.isFinite(term)) return values;
          value -= h * coefficients[f] * term;
        }
        corrected[a] = (float) value;
      }
      return corrected;
    }

    @Override
    public double[][] atRoot(
        long game, int group, long[] words, int base, long[] hidden, int... actions) {
      double[] b = coefficients(game, group, cells == null ? 0 : cells.of(words, base));
      double h =
          scale == null ? 1 : scale.scale(LedgerCell.of(scale.variant()).standing(words, base));
      double[] scaled = b.clone();
      for (int i = 0; i < scaled.length; i++) scaled[i] *= h;
      double[][] result = new double[actions.length][];
      Arrays.fill(result, scaled);
      return result;
    }
  }

  /** 根の係数を使った教師の計算で起きた失敗。 */
  final class PredictionFailure extends RuntimeException {
    private PredictionFailure(RuntimeException cause) {
      super("Root teacher prediction failed: " + cause, cause);
    }

    /** 学習器から伝わった例外に含まれる根の予測の失敗。 */
    public static PredictionFailure find(Throwable cause) {
      while (cause != null && !(cause instanceof PredictionFailure)) cause = cause.getCause();
      return (PredictionFailure) cause;
    }
  }

  /** 根から候補別の係数と共通の倍率を求める補正。 */
  final class ByRoot implements LedgerCorrection {
    private final RootModel.Predictor predictor;
    private final ThreadLocal<RootFeatures.Reader> readers;

    public ByRoot(RootModel model) {
      predictor = new RootModel.Predictor(model.weights(), model.rates());
      readers = ThreadLocal.withInitial(() -> new RootFeatures.Reader(model.weights().variant()));
    }

    @Override
    public boolean needsHands() {
      return predictor.layout.hand >= 0;
    }

    @Override
    public double[][] atRoot(
        long game, int group, long[] words, int base, long[] hidden, int... actions) {
      var features = readers.get().read(words, base, hidden, actions, actions.length, needsHands());
      double[] shared = new double[predictor.columns.length];
      double h = predictor.prepare(features, features.standing(), shared);
      double[][] result = new double[actions.length][LedgerColumn.COUNT];
      if (h != 0)
        for (int a = 0; a < actions.length; a++)
          for (int i = 0; i < predictor.columns.length; i++)
            result[a][predictor.columns[i]] = h * predictor.coefficient(features, a, i, shared);
      return result;
    }

    @Override
    public float[] values(Row row) {
      try {
        return corrected(row);
      } catch (RuntimeException failure) {
        throw new PredictionFailure(failure);
      }
    }

    private float[] corrected(Row row) {
      float[] y = row.values();
      if (row.ledgerFeatures() != LedgerColumn.COUNT) return y;
      var reader = readers.get();
      RootFeatures root =
          reader.read(row.observation(), 0, row.oracle(), null, y.length, needsHands());
      double[] shared = new double[predictor.columns.length];
      double h = predictor.prepare(root, root.standing(), shared);
      if (h == 0) return y;
      float[] corrected = y.clone();
      FitRow fit = FitRow.of(row, null);
      for (int a = 0; a < y.length; a++)
        if (!Float.isNaN(y[a])) {
          for (int c = 0; c < LedgerColumn.COUNT; c++)
            if (!Float.isFinite(row.ledger(a, c))) return y;
          corrected[a] = (float) (y[a] - predictor.amount(root, a, fit, a, shared, h, null));
        }
      return corrected;
    }
  }
}
