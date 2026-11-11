package com.epsilon.prism.core;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 萬子・筒子・索子の入れ替え（6 通り）。麻雀のルールは色を入れ替えても変わらない（索子にしか無い緑一色だけは違うが、ここでは無視する）。
 * 字牌と数字は動かさず、赤 5 は赤 5 に移す（物理牌の番号は 種類 × 4 + 写しの番号で、赤 5 は写しの番号 0）。
 *
 * <p>テストで作った牌山の色を入れ替え、それを {@code startRecordedRound} に渡して同じ局を打つためのテスト用のヘルパー。番号 0 は何も
 * 入れ替えない。1・2 は巡回（萬→筒→索→萬と、その逆向き）、3〜5 は 2 色の入れ替え。
 */
public final class SuitSymmetry {

  /** 入れ替えの数（3 色の順列の数）。 */
  public static final int VIEWS = 6;

  private static final int[][] SUITS = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}, {2, 1, 0}, {0, 2, 1}};
  private static final int[][] KIND_MAPS = new int[VIEWS][Tiles.KINDS];

  static {
    int suited = Tiles.NUMBER_SUITS * Tiles.SUIT_SIZE;
    for (int view = 0; view < VIEWS; view++) {
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        KIND_MAPS[view][kind] =
            kind < suited ? SUITS[view][kind / Tiles.SUIT_SIZE] * Tiles.SUIT_SIZE + kind % Tiles.SUIT_SIZE : kind;
      }
    }
  }

  private SuitSymmetry() {}

  /** 牌の種類の色を入れ替える。負の値は牌を持たない手を表すので、そのまま返す。 */
  public static int kind(int view, int kind) {
    return kind < 0 ? kind : KIND_MAPS[view][kind];
  }

  /** 物理牌の番号の色を入れ替える。写しの番号は変えない。 */
  public static int physical(int view, int physicalId) {
    return Tiles.physicalId(kind(view, Tiles.kind(physicalId)), Tiles.copy(physicalId));
  }

  /** 牌山（位置ごとの物理牌の番号）の色を入れ替えた、新しい牌山を返す。 */
  public static int[] wall(int view, int[] wall) {
    int[] mapped = new int[wall.length];
    for (int position = 0; position < wall.length; position++) {
      mapped[position] = physical(view, wall[position]);
    }
    return mapped;
  }

  /** 牌の種類の集合（種類ごとのビット）の色を入れ替える。 */
  public static long kinds(int view, long kinds) {
    long mapped = 0L;
    for (long rest = kinds; rest != 0L; rest &= rest - 1) {
      mapped |= Tiles.bit(kind(view, Long.numberOfTrailingZeros(rest)));
    }
    return mapped;
  }

  /** {@code shadow} が {@code action} の色を入れ替えた手か。種別・赤とツモ切りの選び方・牌の種類・順子の先頭を比べる。 */
  public static boolean mapped(int view, Action action, Action shadow) {
    return action.type() == shadow.type()
        && action.selection() == shadow.selection()
        && kind(view, action.tile()) == shadow.tile()
        && kind(view, action.base()) == shadow.base();
  }

  /** 色を入れ替えた局で、判断する席 {@code slot} の合法手から、{@code action} の色を入れ替えた手を探して番号を返す。無ければ −1。 */
  public static int find(int view, Decisions shadow, int slot, Action action) {
    for (int index = 0; index < shadow.legalCount(slot); index++) {
      if (mapped(view, action, Action.of(shadow.legal(slot, index)))) {
        return index;
      }
    }
    return -1;
  }
}
