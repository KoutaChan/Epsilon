package com.epsilon.prism.collect;

/**
 * 進行の役割（{@link GameRunner}）。
 *
 * <ul>
 *   <li>{@link #MAIN}: 半荘の主経路。全席が b で最後まで打ち、逸脱しない。ラベル行（率 ρ_c）と中立ロールイン（率 β_c）を起こす。
 *   <li>{@link #COPY}: 主経路の判断から作った写し。逸脱した席 i が d を打ち、以後は全席 b で局末まで打って止まる。席 i の後の判断だけが率 ρ_br で
 *       ラベル行になる（写しごとに最大 maxCopyLabels）。ほかの席の行は書かない。入れ子のロールインはしない。
 *   <li>{@link #FORK}: ラベル行の候補 a の分岐（主経路か写しから）。席が a を打ち、以後は全席 b で局末まで打って止まる。行を書かず、分岐もしない。
 * </ul>
 */
public enum Role {
  MAIN,
  COPY,
  FORK
}
