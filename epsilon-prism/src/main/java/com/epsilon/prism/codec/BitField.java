package com.epsilon.prism.codec;

/**
 * 記録の中に詰めるビットの欄。欄は enum の定数として宣言し、{@link BitLayout} がその enum から位置を一度だけ作る。
 *
 * <p>欄は値の範囲 {@code [min(), max()]} を宣言する。{@link BitLayout} は値 − min を、max − min が入る最小の幅（1〜31 ビット）で詰める。
 * 全てのビットが 0 の欄は min を読むので、なしを −1 とする欄は min を −1 にすれば 0 がなしになり、符号つきの値も同じ規則で書ける。範囲の外の
 * 値は書くときに例外にする。同じ範囲の値を並べる欄（牌ごと・席ごと）は {@link #count()} で繰り返しの数を宣言し、番号で読み書きする。
 */
public interface BitField {

  /**
   * 書ける最大の値を返す（規則の上限）。
   *
   * @return 最大の値
   */
  int max();

  /**
   * 書ける最小の値を返す。全てのビットが 0 の欄はこの値を読む。
   *
   * @return 最小の値
   */
  default int min() {
    return 0;
  }

  /**
   * 同じ範囲の値を並べる数を返す。繰り返しは番号の順に詰める。
   *
   * @return 繰り返しの数（1 以上）
   */
  default int count() {
    return 1;
  }
}
