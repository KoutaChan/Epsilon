package com.epsilon.grp.train;

import java.util.Objects;

/**
 * コホートの主経路で局末の値に使った Φ のモードと打ち切りの値。数のファイルのヘッダーに書く。どの GRP の項目を使ったかは、半荘の記録の
 * ヘッダーの id で分かる。
 *
 * @param mode モードの名前（空でない）
 * @param cutoff 数え上げの打ち切りの値（[0, 1)。数え上げないモードでは NaN）
 */
public record PhiStamp(String mode, double cutoff) {

  public PhiStamp {
    Objects.requireNonNull(mode, "mode");
    if (mode.isEmpty()) {
      throw new IllegalArgumentException("A Φ stamp needs a mode name");
    }
    if (!Double.isNaN(cutoff) && !(cutoff >= 0 && cutoff < 1)) {
      throw new IllegalArgumentException("A Φ cutoff must be NaN or in [0, 1), got " + cutoff);
    }
  }
}
