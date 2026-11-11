package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Set;

/**
 * GrpNetwork の形。grp.json に書く。x → [Linear(hidden) → ReLU] × 2 → S! 個のロジット。フィールドが 1 つも無いとき、隠れ層の
 * 幅が 1 未満のときは IllegalArgumentException を投げる。
 *
 * @param fields 使うフィールド。変更できない集合にして持つ
 * @param hidden 隠れ層の幅
 * @param phaseHeads true なら出力層を段階（GrpPhase）ごとに持つ
 */
public record GrpShape(Variant variant, Set<GrpField> fields, int hidden, boolean phaseHeads) {

  public GrpShape {
    fields = GrpLayout.of(variant, fields).fields();
    if (hidden < 1) {
      throw new IllegalArgumentException("Hidden width must be positive: " + hidden);
    }
  }

  public GrpLayout layout() {
    return GrpLayout.of(variant, fields);
  }

  /** 入力の幅。 */
  public int width() {
    return layout().width();
  }

  /** 出力層の数。 */
  public int heads() {
    return phaseHeads ? GrpPhase.values().length : 1;
  }

  /** 出力の数 S!。 */
  public int outputs() {
    return RankPermutations.of(variant).count();
  }
}
