package com.epsilon.prism.learn;

import com.epsilon.prism.rows.Row;

/**
 * 生徒の教師の作り方。行の教師の値は、運の精算の補正があれば Ỹ = Y − β_c·L
 * （{@link LedgerFit.Correction#values}）、錨があればその後に測った手の上で (1 − w)·Ỹ + w·s_錨（{@link #anchored}）。系譜では錨は
 * 凍結した学習の網 θ_t で w = 1 − K′、補助の頭の項のゲインは K′。refit（replay）は {@link #RAW}。
 *
 * <p>錨の s は当てはめの前に全ての行の分を出してホストに置き（{@link AnchorScores}）、符号化のスレッドが行の貯蔵の番号で引いて混ぜる。学習の
 * レーンは錨を持たない。
 *
 * @param correction 運の精算の補正（null なら生の Y）
 * @param anchor 当てはめの全ての行の錨の s（null なら錨なし）
 * @param anchorWeight 錨の重み w（[0, 1)。錨がなければ 0）
 * @param gain 補助の頭の項に掛けるゲイン K′（(0, 1]）
 */
public record Teacher(
    LedgerFit.Correction correction, AnchorScores anchor, double anchorWeight, float gain) {

  /** 生の Y・錨なし・ゲイン 1（replay の当てはめ）。 */
  public static final Teacher RAW = new Teacher(null, null, 0, 1f);

  /**
   * 組み合わせを確かめる。
   *
   * @throws IllegalArgumentException 錨の重みかゲインが範囲の外か、錨の s なしに錨の重みがあるとき
   */
  public Teacher {
    if (!(anchorWeight >= 0 && anchorWeight < 1)) {
      throw new IllegalArgumentException("anchor weight must be in [0, 1): " + anchorWeight);
    }
    if (anchorWeight > 0 && anchor == null) {
      throw new IllegalArgumentException("an anchor weight needs anchor scores");
    }
    if (!(gain > 0 && gain <= 1)) {
      throw new IllegalArgumentException("gain must be in (0, 1]: " + gain);
    }
  }

  /**
   * 行の教師の値: 補正した値（補正がなければ行の配列そのもの）に、錨があれば錨の s を混ぜたもの。
   *
   * @param row 行
   * @param id 行の貯蔵の番号（錨の s を引く）
   * @return 合法手ごとの教師（測らない手は NaN）
   * @throws IllegalArgumentException 錨の s がこの行を持たないか、手の数が行と違うとき
   */
  float[] values(Row row, int id) {
    float[] values = correction == null ? row.values() : correction.values(row);
    return anchored() ? anchored(values, anchor.scores(id), anchorWeight) : values;
  }

  boolean anchored() {
    return anchorWeight > 0;
  }

  /**
   * 錨の目標: 測った手（値が NaN でない手）では (1 − w)·Y + w·s_錨、測らない手は NaN のまま。行ごとの定数は損失に効かないので中心化しない。
   *
   * @param values 合法手ごとのラベル Y（測らない手は NaN）
   * @param anchor 合法手ごとの錨の s
   * @param weight 錨の重み w
   * @return 合法手ごとの教師
   * @throws IllegalArgumentException 錨の s の手の数が行と違うとき
   */
  static float[] anchored(float[] values, float[] anchor, double weight) {
    if (anchor.length != values.length) {
      throw new IllegalArgumentException(
          "Anchor scored " + anchor.length + " actions for a row of " + values.length);
    }
    float[] mixed = new float[values.length];
    for (int a = 0; a < values.length; a++) {
      mixed[a] =
          Float.isNaN(values[a]) ? Float.NaN : (float) ((1 - weight) * values[a] + weight * anchor[a]);
    }
    return mixed;
  }
}
