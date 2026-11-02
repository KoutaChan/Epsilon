package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;

/**
 * チー・ポンの直後に打てる牌（喰い替えを除く）。この決まりの唯一の持ち主で、live の鳴いた後の打牌（{@link GameEngine}）・判断の解析の遷移
 * （{@link EngineDecisionBuffer}）・牌譜の再生・対局の client・PRISM の O の捕獲が同じ関数を呼ぶ。
 *
 * <p>順はエンジンの遷移の順: 牌の種類の昇順。赤を持つ種類は赤を打つ手を先に、赤でない牌も持てば赤でない手を後に足す。
 */
public final class PostCallDiscards {
  private PostCallDiscards() {}

  /**
   * 鳴いた後の合法な打牌を書く。
   *
   * @param afterCall 鳴きを適用した後の手（呼び出しの間だけ借りる）
   * @param call 鳴いた副露（チーかポン。喰い替えの禁止はこれから決まる）
   * @param out 書く先（初めに空にする）
   * @return 書いた打牌の数
   * @throws IllegalArgumentException 副露がチー・ポンでないとき
   */
  public static int generateInto(HandView afterCall, Meld call, EngineActionBuffer out) {
    out.clear();
    long discardable =
        afterCall.concealedTileTypeMask() & ~PostCallDahaiRestriction.after(call).forbiddenMask();
    for (long remaining = discardable; remaining != 0L; remaining &= remaining - 1) {
      int tileType = Long.numberOfTrailingZeros(remaining);
      if (!afterCall.hasAkaTile(tileType)) {
        out.add(Action.dahai(tileType));
        continue;
      }
      out.add(Action.dahai(tileType, true));
      if (afterCall.hasNonAkaTile(tileType)) {
        out.add(Action.dahai(tileType, false));
      }
    }
    return out.size();
  }
}
