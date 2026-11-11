package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;
import java.util.Set;

/**
 * OutcomeNetwork の形。x → [Linear(hidden) → ReLU] × 2 → ロジット（HandOutcomes の終わり方・打点・「立直した」の項）。
 * ルールが {@code FinalRound.VARIANTS} に無いとき、フィールドが空のとき、隠れ層の幅が 1 未満のときは
 * IllegalArgumentException を投げる。
 *
 * @param fields 使うフィールド。変更できない集合にして持つ
 * @param hidden 隠れ層の幅
 */
public record OutcomeShape(Variant variant, Set<GrpField> fields, int hidden) {

  public OutcomeShape {
    if (!FinalRound.VARIANTS.contains(Objects.requireNonNull(variant, "variant"))) {
      throw new IllegalArgumentException("The final-round enumeration is not defined for " + variant);
    }
    fields = GrpLayout.of(variant, fields).fields();
    if (hidden < 1) {
      throw new IllegalArgumentException("Hidden width must be positive: " + hidden);
    }
  }

  public GrpLayout layout() {
    return GrpLayout.of(variant, fields);
  }

  /** 入力の幅を返す。 */
  public int width() {
    return layout().width();
  }

  public HandOutcomes outcomes() {
    return HandOutcomes.of(variant);
  }

  /** 出力するロジットの数を返す。 */
  public int logits() {
    return outcomes().logits();
  }
}
