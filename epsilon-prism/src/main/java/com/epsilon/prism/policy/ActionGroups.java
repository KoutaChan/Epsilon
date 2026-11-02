package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import java.util.List;

/**
 * 方策の基準測度の手のまとまり。打牌（DAHAI・RIICHI_DAHAI）と鳴き（CHI・PON）はそれぞれ1つのまとまり、ほかの手は1手で1つのまとまりとする。
 *
 * <p>softmax の基準測度をまとまりごとに一様にする。後悔項と事前分布が 0 なら、まとまりごとの総確率が等しい（ツモ和了と打牌全体が半々、PASS と
 * 鳴き全体が半々）。手の数をそのまま基準にすると、ツモ和了・槓・九種九牌は打牌の枚数分（14枚なら ln 14 ≈ 2.6 nat）不利になり、鳴きは鳴き方の
 * 数だけ PASS より有利になる。まとまりの中の配分（どの牌を切るか、どの形で鳴くか）は変えない。
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

  /** 合法手の並びのまとまりを group[0..n) に書く。 */
  public static void fill(List<Action> legal, byte[] group) {
    for (int a = 0, n = legal.size(); a < n; a++) group[a] = of(legal.get(a).type());
  }

  /** 行動 ID の並び（行の legal）のまとまりを group[0..n) に書く。 */
  public static void fill(int[] legalIds, byte[] group) {
    for (int a = 0; a < legalIds.length; a++) group[a] = of(Action.fromIndex(legalIds[a]).type());
  }

  /** 打牌と鳴きのまとまりの手から、まとまりの手の数の対数を引く（まとまりの基準測度を1手分にする）。 */
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
