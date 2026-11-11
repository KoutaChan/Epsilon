package com.epsilon.prism.policy;

import com.epsilon.mahjong.action.Action;

/**
 * 方策の基準測度を決めるための、手のまとまり。打牌（ふつうの打牌と立直の打牌）で 1 つ、鳴き（チー・ポン）で 1 つのまとまりにし、
 * ほかの手は 1 手で 1 つのまとまりにする。
 *
 * <p>softmax の基準測度を、まとまりごとに一様にする。後悔の項と事前分布が 0 なら、まとまりごとの確率の合計は等しい（ツモ和了と打牌全体が
 * 半々、PASS と鳴き全体が半々）。手の数をそのまま基準にすると、ツモ和了・槓・九種九牌は打牌の枚数の分（14 枚なら ln 14 ≈ 2.6 nat）不利に
 * なり、鳴きは鳴き方の数の分だけ PASS より有利になる。まとまりの中の配分（どの牌を切るか、どの形で鳴くか）は変えない。
 */
public final class ActionGroups {
  public static final byte SINGLE = 0, DISCARD = 1, CALL = 2;

  private ActionGroups() {}

  public static byte of(Action.Type type) {
    return switch (type) {
      case DAHAI, RIICHI_DAHAI -> DISCARD;
      case CHI, PON -> CALL;
      default -> SINGLE;
    };
  }

  /** 合法手（行動の番号）ごとに、その手が入るまとまりを {@code group} の同じ位置に書く。 */
  public static void fill(int[] legal, byte[] group) {
    for (int a = 0; a < legal.length; a++) group[a] = of(Action.of(legal[a]).type());
  }

  /** 打牌と鳴きの手の z から、それぞれのまとまりの手の数の対数を引く。これで、まとまり全体の基準測度が 1 手分になる。 */
  static void applyBaseMeasure(byte[] group, int n, double[] z) {
    int discards = 0, calls = 0;
    for (int a = 0; a < n; a++) {
      if (group[a] == DISCARD) discards++;
      else if (group[a] == CALL) calls++;
    }
    if (discards + calls == n && (discards == 0 || calls == 0)) return;
    double discardShift = discards > 1 ? Math.log(discards) : 0;
    double callShift = calls > 1 ? Math.log(calls) : 0;
    for (int a = 0; a < n; a++) {
      if (group[a] == DISCARD) z[a] -= discardShift;
      else if (group[a] == CALL) z[a] -= callShift;
    }
  }
}
