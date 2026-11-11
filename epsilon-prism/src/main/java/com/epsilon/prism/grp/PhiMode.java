package com.epsilon.prism.grp;

import java.util.Locale;
import java.util.Optional;

/** 局末の値 Φ の求め方（モード）。反復 t の Φ は、t に効く GRP の項目（phi.jsonl）に、t に効くモード（mode.jsonl）を当てて決まる。 */
public enum PhiMode {

  /** 全ての局末を、GRP の項目のネットワークで求める。 */
  GRP,

  /**
   * 次の局が南場の最後の局かそれより後になる局末は、GRP の項目の q を使って最後の局を数え上げ、ルールどおりに精算して求める。
   * それ以外の局末は GRP の項目のネットワークで求めるので、値は GRP のモードとビット単位で同じになる。
   */
  LATE;

  /** モードを名前で指すときの別名で、名前を小文字にしたもの。項目の id と取り違えないよう、この別名は項目の id には使えない。 */
  public String alias() {
    return name().toLowerCase(Locale.ROOT);
  }

  /** 別名が指すモードを返す。どのモードの別名でもなければ空。 */
  public static Optional<PhiMode> byAlias(String alias) {
    for (PhiMode mode : values()) {
      if (mode.alias().equals(alias)) {
        return Optional.of(mode);
      }
    }
    return Optional.empty();
  }
}
