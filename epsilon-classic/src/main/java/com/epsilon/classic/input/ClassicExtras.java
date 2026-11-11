package com.epsilon.classic.input;

import com.epsilon.mahjong.engine.SeatView;
import java.util.Objects;

/**
 * 判断の記録からは求められない入力フィールドを補う、追加の入力。判断の記録と同じ時点に、同じ席から見た値を書く。フィールドごとに long を
 * 1 つ持ち、宣言の順に並べる。classic のフィールドはどれも判断の記録から求められる（{@link ClassicTokenizer}）ので、追加のフィールドは
 * 無く、long は 0 個。入力のフィンガープリントには、追加のフィールドの並びも入れる。
 *
 * @param words 追加のフィールドの値（宣言の順）
 */
public record ClassicExtras(long[] words) {

  /** 追加のフィールド。 */
  public enum Field {}

  public static final int WORDS = Field.values().length;

  /** long の数がフィールドの数と違えば IllegalArgumentException を投げる。 */
  public ClassicExtras {
    Objects.requireNonNull(words, "words");
    if (words.length != WORDS) {
      throw new IllegalArgumentException("Classic extras hold " + WORDS + " words, not " + words.length);
    }
  }

  /** 値を全て 0 にした追加の入力を返す。 */
  public static ClassicExtras empty() {
    return new ClassicExtras(new long[WORDS]);
  }

  /**
   * 席から見た追加の入力を、判断の記録と同じ時点で書く。追加のフィールドが無いので、席の数を確かめるだけ。
   *
   * @throws IllegalArgumentException 四人打ちでないルールの局面を渡したとき
   */
  public void capture(SeatView view) {
    if (view.variant().seats() != ClassicSchema.SEATS) {
      throw new IllegalArgumentException(
          "Classic inputs need " + ClassicSchema.SEATS + " seats, not " + view.variant());
    }
  }
}
