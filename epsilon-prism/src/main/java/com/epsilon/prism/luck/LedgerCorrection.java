package com.epsilon.prism.luck;

import com.epsilon.prism.rows.Row;

/**
 * 運の精算の補正 Ỹ = Y − β·L。
 *
 * @param beta beta[g][c][f] は、偶奇 g の行でクラス c の列 f に掛ける係数で、偶奇 1 − g の行で当てはめたもの
 * @param cells 局・順位の区分（区分ごとの係数が無ければ null）
 * @param byCell byCell[g][c][k] は、区分 k の係数（区分で当てはめなかった所は null）。区分ごとの係数が無ければ null
 */
public record LedgerCorrection(double[][][] beta, LedgerCell cells, double[][][][] byCell) {

  public LedgerCorrection {
    if ((cells == null) != (byCell == null))
      throw new IllegalArgumentException(
          "Ledger cell coefficients need their cells and vice versa");
  }

  public LedgerCorrection(double[][][] beta) {
    this(beta, null, null);
  }

  /** 指定した偶奇と判断クラスの係数。局・順位の係数があればその区分を使う。 */
  public double[] coefficients(long gameId, int group, int cell) {
    int fold = (int) (gameId & 1);
    if (byCell != null && byCell[fold][group][cell] != null) return byCell[fold][group][cell];
    return beta[fold][group];
  }

  private static final int COLUMNS = LedgerColumn.COUNT;

  /**
   * 補正した Y を返す。測った手は Y_a − β·L_a にし、測らない手は NaN のままにする。帳簿のない行と帳簿の欠けた行は
   * {@code row.values()} をそのまま返す。
   */
  public float[] values(Row row) {
    return values(FitRow.of(row, cells));
  }

  float[] values(FitRow row) {
    float[] values = row.values();
    if (row.ledgerFeatures() != COLUMNS) return values;
    double[] coefficients =
        coefficients(row.gameId(), row.decisionClass().ordinal(), byCell == null ? 0 : row.cell());
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
        value -= coefficients[f] * term;
      }
      corrected[a] = (float) value;
    }
    return corrected;
  }
}
