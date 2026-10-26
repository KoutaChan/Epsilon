package com.epsilon.prism.core;

import com.epsilon.core.Action;
import java.util.List;

/**
 * 手出しとツモ切りの双子。同じ種類（DAHAI か RIICHI_DAHAI）・同じ牌種・同じ赤の選びの打牌は 1 つの候補として扱う（候補集合・逸脱・認証で同じ手と数える）。
 * 双子でない手は自分とだけ同じ。
 */
public final class Twins {
  private Twins() {}

  /** 手番の打牌（立直宣言の打牌を含む）。 */
  public static boolean discard(Action action) {
    Action.Type type = action.type();
    return type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI;
  }

  /** x と y が同じ手か双子なら true。 */
  public static boolean same(Action x, Action y) {
    if (x.equals(y)) return true;
    return discard(x)
        && x.type() == y.type()
        && x.tileType() == y.tileType()
        && x.usesAkaTileFromHand() == y.usesAkaTileFromHand();
  }

  /** legal[a] と legal[b] が同じ手か双子なら true。 */
  public static boolean same(List<Action> legal, int a, int b) {
    return a == b || same(legal.get(a), legal.get(b));
  }

  /** 双子だけでない（2 つ以上の候補がある）なら true。 */
  public static boolean branchable(List<Action> legal) {
    for (int a = 1, n = legal.size(); a < n; a++) if (!same(legal, 0, a)) return true;
    return false;
  }
}
