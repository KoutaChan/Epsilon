package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * OutcomeNetwork の形。x → [Linear(hidden) → ReLU] × 2 → ロジット。ロジットは、HandOutcomes の終わり方・打点・「立直した」の
 * ロジットに、足すロジットのまとまり（OutcomeLink）を宣言の順で続けたもの。ルールが {@code FinalRound.VARIANTS} に無いとき、
 * フィールドが空のとき、隠れ層の幅が 1 未満のときは IllegalArgumentException を投げる。
 *
 * @param fields 使うフィールド。変更できない集合にして持つ
 * @param hidden 隠れ層の幅
 * @param links 足すロジットのまとまり。変更できない集合にして持つ
 */
public record OutcomeShape(
    Variant variant, Set<GrpField> fields, int hidden, Set<OutcomeLink> links) {

  public OutcomeShape {
    if (!FinalRound.VARIANTS.contains(Objects.requireNonNull(variant, "variant"))) {
      throw new IllegalArgumentException(
          "The final-round enumeration is not defined for " + variant);
    }
    fields = GrpLayout.of(variant, fields).fields();
    if (hidden < 1) {
      throw new IllegalArgumentException("Hidden width must be positive: " + hidden);
    }
    EnumSet<OutcomeLink> copy = EnumSet.noneOf(OutcomeLink.class);
    copy.addAll(links);
    links = Collections.unmodifiableSet(copy);
  }

  /** 足すロジットを持たない形を作る。 */
  public OutcomeShape(Variant variant, Set<GrpField> fields, int hidden) {
    this(variant, fields, hidden, Set.of());
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
    int logits = outcomes().logits();
    for (OutcomeLink link : links) {
      logits += link.width(variant);
    }
    return logits;
  }

  /**
   * 足すロジットのまとまりの開始位置を返す。
   *
   * @return ロジットの中での開始位置。形がそのまとまりを持たなければ −1
   */
  public int linkStart(OutcomeLink link) {
    if (!links.contains(link)) {
      return -1;
    }
    int start = outcomes().logits();
    for (OutcomeLink before : links) {
      if (before == link) {
        break;
      }
      start += before.width(variant);
    }
    return start;
  }
}
