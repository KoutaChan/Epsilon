package com.epsilon.prism.grp;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * grp/mode.jsonl の 1 行（{@link PhiHistory}）。反復 t で効くのは、効き始めが t 以下の行のうち効き始めが最も遅い行（同じなら後に書いた
 * 行）。効く行が無い反復は GRP のモード。前のモードに戻すときも、新しい行を足すだけにする。
 *
 * @param mode モード
 * @param cutoff LATE の数え上げの打ち切りの値（0 以上 1 未満。GRP の行では null）
 * @param effectiveFrom 効き始める反復（1 以上）
 * @param reason 書いた理由
 */
public record ModeLine(PhiMode mode, Double cutoff, int effectiveFrom, String reason) {

  /** grp-switch が足す行で、理由の先頭に付ける文字列。 */
  static final String SWITCH = "switch:";

  public ModeLine {
    Objects.requireNonNull(mode, "mode");
    Objects.requireNonNull(reason, "reason");
    if (effectiveFrom < 1) {
      throw new IllegalArgumentException("Mode line effectiveFrom must be positive: " + effectiveFrom);
    }
    if (mode == PhiMode.LATE && (cutoff == null || !(cutoff >= 0 && cutoff < 1))) {
      throw new IllegalArgumentException("A LATE mode line needs a cutoff in [0, 1), got " + cutoff);
    }
    if (mode == PhiMode.GRP && cutoff != null) {
      throw new IllegalArgumentException("A GRP mode line takes no cutoff, got " + cutoff);
    }
  }

  /** 反復 {@code from} から LATE のモードにする行。 */
  public static ModeLine late(double cutoff, int from, String reason) {
    return new ModeLine(PhiMode.LATE, cutoff, from, reason);
  }

  /** 反復 {@code from} から GRP のモードにする行。 */
  public static ModeLine grp(int from, String reason) {
    return new ModeLine(PhiMode.GRP, null, from, reason);
  }

  /** 反復 t に効く行を返す。効き始めが t 以下の行のうち効き始めが最も遅い行で、同じなら後に書いた行。無ければ空。 */
  public static Optional<ModeLine> effectiveAt(List<ModeLine> lines, int t) {
    ModeLine effective = null;
    for (ModeLine line : lines) {
      if (line.effectiveFrom <= t && (effective == null || line.effectiveFrom >= effective.effectiveFrom)) {
        effective = line;
      }
    }
    return Optional.ofNullable(effective);
  }
}
