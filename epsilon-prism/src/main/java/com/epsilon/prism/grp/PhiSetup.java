package com.epsilon.prism.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.variant.PrismVariant;
import java.util.Objects;

/**
 * GRP の裏の処理と、Φ を決める処理に渡す run の設定。
 *
 * @param phi Φ のセクション
 * @param variant ルール（四人打ちか三人打ちか）
 * @param seed run の乱数の種
 */
public record PhiSetup(PhiSettings phi, Variant variant, long seed) {

  public PhiSetup {
    Objects.requireNonNull(phi, "phi");
    Objects.requireNonNull(variant, "variant");
  }

  /** このルールでの順位の効用。入力の形式で決まる。 */
  public RankUtility utility() {
    return PrismVariant.of(variant).input().utility();
  }

  /** 自分の GRP の設定を返す。出どころが自分の GRP でなければ IllegalStateException を投げる。 */
  public PhiSettings.SelfGrp selfGrp() {
    if (phi instanceof PhiSettings.SelfGrp self) {
      return self;
    }
    throw new IllegalStateException("Φ source is " + phi.source() + ", not SELF_GRP");
  }

  /** 自分の GRP の run なら true。 */
  public boolean selfPlay() {
    return phi instanceof PhiSettings.SelfGrp;
  }

  /** 主経路の半荘を grp/records に記録し、全ての局について始まりの卓と結果を grp/counts に記録するなら true。 */
  public boolean records() {
    return phi instanceof PhiSettings.SelfGrp self && self.grp().monitor().records();
  }

  /** 記録した半荘を Φ で採点するなら true。 */
  public boolean scores() {
    return phi instanceof PhiSettings.SelfGrp self && self.grp().monitor().scores();
  }
}
