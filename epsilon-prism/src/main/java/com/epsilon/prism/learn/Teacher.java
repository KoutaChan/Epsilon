package com.epsilon.prism.learn;

import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.rows.Row;

/**
 * 生徒に渡す教師の作り方。運の精算の補正があれば、行の教師の値を Ỹ = Y − β_c·L にする（{@link LedgerCorrection#values}）。
 * アンカーがあれば、その後で測った手の値を (1 − w)·Ỹ + w·s_アンカー にする（{@link #anchored}）。系譜では、アンカーは学習する
 * ネットワークを凍結した θ_t で、w = 1 − K′、補助のヘッドの項のゲインは K′ にする。refit（replay）は {@link #RAW} を使う。
 *
 * <p>アンカーの s は、学習の前に全ての行の分を計算してホストに置く（{@link AnchorScores}）。符号化のスレッドが行のストアの番号で引いて
 * 混ぜるので、学習のレーンはアンカーを持たない。
 *
 * @param correction 運の精算の補正（null なら生の Y を使う）
 * @param anchor 学習に使う全ての行のアンカーの s（null ならアンカーなし）
 * @param anchorWeight アンカーの重み w（[0, 1)。アンカーがなければ 0）
 * @param gain 補助のヘッドの項に掛けるゲイン K′（(0, 1]）
 */
public record Teacher(
    LedgerCorrection correction, AnchorScores anchor, double anchorWeight, float gain) {

  /** 生の Y を使い、アンカーなし、ゲイン 1 の教師（replay の学習に使う）。 */
  public static final Teacher RAW = new Teacher(null, null, 0, 1f);

  /**
   * 値の組み合わせを確かめる。
   *
   * @throws IllegalArgumentException アンカーの重みかゲインが範囲の外のとき、またはアンカーの s がないのにアンカーの重みが正のとき
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
   * 行の教師の値を返す。補正した値に、アンカーがあればアンカーの s を混ぜる。補正が無いときや、帳簿が無くて補正しない行では、値は行の
   * 配列そのものになる。そのときは行の配列を書き換えず、新しい配列に混ぜる。補正で新しく作った配列なら、その上に混ぜる。
   *
   * @param row 行
   * @param id 行のストアの番号（アンカーの s を引くのに使う）
   * @return 合法手ごとの教師（測らない手は NaN）
   * @throws IllegalArgumentException アンカーの s にこの行がないとき、または手の数が行と違うとき
   */
  float[] values(Row row, int id) {
    float[] raw = row.values();
    float[] values = correction == null ? raw : correction.values(row);
    if (!anchored()) return values;
    return anchored(values, anchor.scores(id), anchorWeight, values == raw ? new float[raw.length] : values);
  }

  boolean anchored() {
    return anchorWeight > 0;
  }

  /**
   * アンカーを混ぜた目標を書く。測った手（値が NaN でない手）は (1 − w)·Y + w·s_アンカー にし、測らない手は NaN のままにする。
   * 行ごとの定数は損失に効かないので、中心化はしない。手ごとに Y を読んでから書くので、書く先は Y の配列でもよい。
   *
   * @param values 合法手ごとのラベル Y（測らない手は NaN）
   * @param anchor 合法手ごとのアンカーの s
   * @param weight アンカーの重み w
   * @param mixed 合法手ごとの教師を書く先（長さは手の数）
   * @return 書いた配列
   * @throws IllegalArgumentException アンカーの s の手の数が行と違うとき
   */
  static float[] anchored(float[] values, float[] anchor, double weight, float[] mixed) {
    if (anchor.length != values.length) {
      throw new IllegalArgumentException(
          "Anchor scored " + anchor.length + " actions for a row of " + values.length);
    }
    for (int a = 0; a < values.length; a++) {
      mixed[a] =
          Float.isNaN(values[a]) ? Float.NaN : (float) ((1 - weight) * values[a] + weight * anchor[a]);
    }
    return mixed;
  }
}
