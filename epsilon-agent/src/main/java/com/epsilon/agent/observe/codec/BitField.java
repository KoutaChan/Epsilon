package com.epsilon.agent.observe.codec;

import com.epsilon.mahjong.rules.Variant;

/**
 * 記録の中に詰めるビットのフィールド。フィールドは enum の定数として宣言し、{@link BitLayout} がその enum からルールごとに位置を
 * 一度だけ決める。
 *
 * <p>フィールドはルールごとの値の範囲 {@code [min, max]} を宣言する（席の数・行動の数・語彙の大きさはルールで変わる）。
 * {@link BitLayout} は 値 − min を、max − min が入る最小の幅（1〜31 ビット）で詰める。全てのビットが 0 のフィールドは min として
 * 読めるので、無しを −1 で表すフィールドは min を −1 にすれば、0 で埋めたときに無しになる。符号つきの値も同じやり方で書ける。範囲の
 * 外の値は、書くときに例外にする。同じ範囲の値を並べるフィールド（牌ごと・席ごと）は {@link #count} で繰り返しの数を宣言し、番号で
 * 読み書きする。
 */
public interface BitField {

  /** 指定したルールで書ける最大の値。 */
  int max(Variant variant);

  /** 指定したルールで書ける最小の値。全てのビットが 0 のフィールドは、この値として読める。 */
  default int min(Variant variant) {
    return 0;
  }

  /** 指定したルールで、同じ範囲の値を並べる数（1 以上）。繰り返しは番号の順に詰める。 */
  default int count(Variant variant) {
    return 1;
  }
}
