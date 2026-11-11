package com.epsilon.prism.luck;

import com.epsilon.prism.rows.Row;

/**
 * 運の精算の補正 Ỹ = Y − β·L。
 *
 * @param beta beta[g][c][f] は、偶奇 g の行でクラス c の列 f に掛ける係数で、偶奇 1 − g の行で当てはめたもの
 */
public record LedgerCorrection(double[][][] beta) {

  private static final int COLUMNS = LedgerColumn.COUNT;

  /**
   * 補正した Y を返す。測った手は Y_a − β·L_a にし、測らない手は NaN のままにする。帳簿のない行と帳簿の欠けた行は
   * {@code row.values()} をそのまま返す。
   */
  public float[] values(Row row) {
    return values(FitRow.of(row));
  }

  float[] values(FitRow row) {
    float[] values = row.values();
    if (row.ledgerFeatures() != COLUMNS) return values;
    double[] coefficients = beta[(int) (row.gameId() & 1)][row.decisionClass().ordinal()];
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
