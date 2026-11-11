package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * GruSinkhorn の形。grp.json に書く。GRU を重ね、その上に射影の層を置く。幅か層の数が 1 未満なら IllegalArgumentException を
 * 投げる。
 *
 * @param hidden GRU の状態の幅。射影の層の幅も同じ
 * @param layers GRU を重ねる層の数
 */
public record GruShape(Variant variant, int hidden, int layers) {

  public GruShape {
    Objects.requireNonNull(variant, "variant");
    if (hidden < 1 || layers < 1) {
      throw new IllegalArgumentException("Hidden width and layers must be positive: " + hidden + ", " + layers);
    }
  }

  /** 1 ステップの入力の幅（RoundField の行の幅）。 */
  public int inputs() {
    return RoundField.width(variant);
  }

  /** 1 層のゲートの幅。r・z・n の 3 つ分で、状態の幅の 3 倍。 */
  public int gates() {
    return 3 * hidden;
  }

  /** 周辺確率の数 S²。並びは [絶対席 × S ＋ 順位]。 */
  public int marginals() {
    return variant.seats() * variant.seats();
  }
}
