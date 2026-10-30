package com.epsilon.prism.learn;

import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.rows.Row;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 生徒の教師の作り方。行の教師の値は、運の精算の補正があれば Ỹ = Y − β_c·L
 * （{@link LedgerFit.Correction#values}）、錨があればその後に測った手の上で (1 − w)·Ỹ + w·s_錨（{@link #anchored}）。系譜では錨は
 * 凍結した学習の網 θ_t で w = 1 − K′、補助の頭の項のゲインは K′。refit（replay）は {@link #RAW}。
 *
 * <p>錨の s は学習の各レーンが自分のデバイスに読んだ錨の網（{@link Anchor}）で、微小バッチの順伝播の直前に出す（{@link Learner}）。道は
 * 方策と同じ推論の道で選び、HIP なら収集と同じ自前の推論、そうでなければ eager の凍結した網にする。どちらも錨の配列は生徒の順伝播の前に
 * 閉じるので、デバイスのメモリは生徒の 1 微小バッチ分と計画の作業域を超えない。
 *
 * @param correction 運の精算の補正（null なら生の Y）
 * @param anchor 錨の網のディレクトリ（null なら錨なし）
 * @param anchorWeight 錨の重み w（[0, 1)。錨がなければ 0）
 * @param gain 補助の頭の項に掛けるゲイン K′（(0, 1]）
 * @param anchorInference 錨の s を出す推論の道（設定の inference）
 */
public record Teacher(
    LedgerFit.Correction correction,
    Path anchor,
    double anchorWeight,
    float gain,
    InferencePath anchorInference) {

  /** 生の Y・錨なし・ゲイン 1（replay の当てはめ）。 */
  public static final Teacher RAW = new Teacher(null, null, 0, 1f, InferencePath.EAGER);

  /**
   * 組み合わせを確かめる。
   *
   * @throws IllegalArgumentException 錨の重みかゲインが範囲の外か、錨の網なしに錨の重みがあるとき
   */
  public Teacher {
    Objects.requireNonNull(anchorInference, "anchorInference");
    if (!(anchorWeight >= 0 && anchorWeight < 1)) {
      throw new IllegalArgumentException("anchor weight must be in [0, 1): " + anchorWeight);
    }
    if (anchorWeight > 0 && anchor == null) {
      throw new IllegalArgumentException("an anchor weight needs an anchor network");
    }
    if (!(gain > 0 && gain <= 1)) {
      throw new IllegalArgumentException("gain must be in (0, 1]: " + gain);
    }
  }

  /** 行の教師の値（錨を混ぜる前。補正がなければ行の配列そのもの）。 */
  float[] values(Row row) {
    return correction == null ? row.values() : correction.values(row);
  }

  boolean anchored() {
    return anchorWeight > 0;
  }

  /**
   * 教師に錨の s を混ぜる: 測った手の上で (1 − w)·教師 + w·s にした教師を返す（λ_c・行の重みは同じ）。
   *
   * @param labels 行の順の教師
   * @param scores 錨の s（行 r の k 番目の手は {@code values[starts[r] + k]}）
   * @return 行の順の教師
   * @throws IllegalArgumentException 錨の s の行の数が教師と違うとき
   */
  List<ValueLabel> mix(List<ValueLabel> labels, Anchor.Scores scores) {
    if (scores.starts().length != labels.size()) {
      throw new IllegalArgumentException(
          "anchor scores cover " + scores.starts().length + " rows but there are " + labels.size() + " labels");
    }
    List<ValueLabel> mixed = new ArrayList<>(labels.size());
    for (int row = 0; row < labels.size(); row++) {
      ValueLabel label = labels.get(row);
      int from = scores.starts()[row];
      float[] own = Arrays.copyOfRange(scores.values(), from, from + label.values().length);
      mixed.add(new ValueLabel(anchored(label.values(), own, anchorWeight), label.lambda(), label.weight()));
    }
    return mixed;
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
