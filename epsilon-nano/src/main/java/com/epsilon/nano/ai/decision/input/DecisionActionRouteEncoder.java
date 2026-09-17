package com.epsilon.nano.ai.decision.input;

import com.epsilon.core.Action;
import java.util.Arrays;
import java.util.List;

/** 同じ打牌に対応するダマとリーチの候補位置を求め、方策の計算で直接参照できるようにする。 */
public final class DecisionActionRouteEncoder {

  /** エンコーダー/性能測定ワーカーが所有する行動参照経路解決一時バッファ。 */
  public static final class Scratch {
    private final int[] slotsByTypeAndIdentity = new int[Action.DISCARD_IDENTITY_COUNT * 2];
  }

  private DecisionActionRouteEncoder() {}

  /**
   * 同じ物理打牌に属するDAMA/RIICHI 枠と識別情報代表枠を書き込む。
   *
   * <p>参照先の対応は行動意味特徴ではなく、行動軸の直積マスクをネットワーク内に作らないためのメタデータである。
   *
   * @param legalActions 候補枠順の合法行動一覧
   * @param writer 同じ枠順で参照経路メタデータを書き込む入力書き込み処理
   */
  public static void encode(List<Action> legalActions, DecisionInputWriter writer) {
    encode(legalActions, writer, new Scratch());
  }

  /** 呼び出し側所有一時バッファを再利用して行動参照経路を書き込む。 */
  public static void encode(
      List<Action> legalActions, DecisionInputWriter writer, Scratch scratch) {
    int[] slotsByTypeAndIdentity = scratch.slotsByTypeAndIdentity;
    Arrays.fill(slotsByTypeAndIdentity, -1);
    for (int slot = 0; slot < legalActions.size(); slot++) {
      Action action = legalActions.get(slot);
      int identity = action.discardIdentityIndex();
      if (identity < 0) {
        continue;
      }
      int typeOffset = action.type() == Action.Type.DAHAI ? 0 : Action.DISCARD_IDENTITY_COUNT;
      slotsByTypeAndIdentity[typeOffset + identity] = slot;
    }
    for (int slot = 0; slot < legalActions.size(); slot++) {
      int identity = legalActions.get(slot).discardIdentityIndex();
      if (identity < 0) {
        continue;
      }
      int damaSlot = slotsByTypeAndIdentity[identity];
      int riichiSlot = slotsByTypeAndIdentity[Action.DISCARD_IDENTITY_COUNT + identity];
      int representativeSlot = damaSlot >= 0 ? damaSlot : riichiSlot;
      writer.actionRoute(
          slot,
          DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE,
          slot == representativeSlot ? 1 : 0);
      writer.actionRoute(
          slot,
          DecisionInputSchema.ActionRoute.DISCARD_REPRESENTATIVE_SLOT,
          representativeSlot + 1);
      writer.actionRoute(slot, DecisionInputSchema.ActionRoute.DAMA_SLOT, damaSlot + 1);
      writer.actionRoute(slot, DecisionInputSchema.ActionRoute.RIICHI_SLOT, riichiSlot + 1);
    }
  }
}
