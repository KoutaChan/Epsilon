package com.epsilon.major.ai.decision.data;

import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;

/** 合法手候補の位置と、行動を識別する ID を相互に対応付ける。 */
public final class EpsilonDecisionLegalActions {

  private static final int[] EMPTY = new int[0];

  private EpsilonDecisionLegalActions() {}

  /**
   * コンパクト格納位置数として合法な範囲か検証する。
   *
   * @param legalActionCount 入力行に存在する合法行動数
   * @return 検証済みの同じ値
   */
  public static int compactCount(int legalActionCount) {
    if (legalActionCount < 0 || legalActionCount > DecisionInputSchema.MAX_LEGAL_ACTIONS) {
      throw new IllegalArgumentException(
          "legalActionCount must be in [0, "
              + DecisionInputSchema.MAX_LEGAL_ACTIONS
              + "]: "
              + legalActionCount);
    }
    return legalActionCount;
  }

  /**
   * 先頭の合法格納位置だけを残した行動 ID 配列を返す。
   *
   * @param legalActionIdBySlot 格納位置順の行動 ID 配列
   * @param legalActionCount 有効な先頭格納位置数
   * @return 有効格納位置だけを含む配列。元配列が既に同じ長さならその参照
   */
  public static int[] compact(int[] legalActionIdBySlot, int legalActionCount) {
    int count = compactCount(legalActionCount);
    if (count == 0) {
      return EMPTY;
    }
    if (legalActionIdBySlot.length < count) {
      throw new IllegalArgumentException("legalActionIdBySlot shorter than legalActionCount");
    }
    validate(legalActionIdBySlot, 0, count);
    if (legalActionIdBySlot.length == count) {
      return legalActionIdBySlot;
    }
    int[] out = new int[count];
    System.arraycopy(legalActionIdBySlot, 0, out, 0, count);
    return out;
  }

  /**
   * コンパクト合法手配列から指定格納位置の安定行動 ID を取得する。
   *
   * @param legalActionIdBySlot コンパクト格納位置順の行動 ID
   * @param slot 取得するゼロ-に基づく格納位置
   * @return 指定格納位置の行動 ID
   */
  public static int actionIdAt(int[] legalActionIdBySlot, int slot) {
    if (slot < 0 || slot >= legalActionIdBySlot.length) {
      throw new IllegalArgumentException(
          "legal action slot out of range: slot=" + slot + " count=" + legalActionIdBySlot.length);
    }
    return legalActionIdBySlot[slot];
  }

  static void validate(int[] actionIds, int offset, int count) {
    if (count == 0) {
      return;
    }
    boolean[] seen = new boolean[Action.ACTION_SPACE_SIZE];
    int decisionKind = 0;
    for (int slot = 0; slot < count; slot++) {
      int actionId = actionIds[offset + slot];
      if (!Action.isValidIndex(actionId)) {
        throw new IllegalArgumentException("invalid legal action id: " + actionId);
      }
      if (seen[actionId]) {
        throw new IllegalArgumentException("duplicate legal action id: " + actionId);
      }
      seen[actionId] = true;

      Action.Type type = Action.fromIndex(actionId).type();
      int currentKind = type.isResponse() ? 1 : 2;
      if (decisionKind != 0 && decisionKind != currentKind) {
        throw new IllegalArgumentException(
            "response and turn actions cannot share a decision: " + type);
      }
      decisionKind = currentKind;
    }
  }
}
