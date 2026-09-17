package com.epsilon.major.ai.decision.training;

import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;

/**
 * 合法手集合から、判断機会を打牌・リーチ・応答などに分類する。
 *
 * <p>通常の打牌だけを選べる場合は DAHAI、リーチを選べる場合は RIICHI、鳴き・ロンへの応答や和了・槓などを含む場合は REACTION
 * とする。実際に選んだ行動ではなく、その時点の合法手集合で判定する。
 */
public enum EpsilonDecisionPointKind {
  /** リーチ宣言候補を含まない通常打牌決定。 */
  DAHAI,

  /** RIICHI と DAMA の選択を含む打牌決定。 */
  RIICHI,

  /** 応答、和了、途中流局、カンのいずれかを含むまれな判断機会。 */
  REACTION;

  /**
   * 一行の合法行動集合から判断機会の種別を分類する。
   *
   * @param input コンパクト合法手メタデータを持つホスト側バッチ
   * @param row 分類するバッチ行
   * @return RIICHI、REACTION、DAHAI の優先順で分類した種別
   */
  public static EpsilonDecisionPointKind of(DecisionHostBatch input, int row) {
    boolean riichi = false;
    boolean reaction = false;
    for (int slot = 0; slot < input.legalActionCount(row); slot++) {
      int actionId = input.legalActionId(row, slot);
      Action.Type type = Action.fromIndex(actionId).type();
      riichi |= type.group() == Action.Group.RIICHI;
      reaction |= isReaction(type);
    }
    return of(riichi, reaction);
  }

  /** データ本体を復元せず、学習データ片付随情報のコンパクト合法手 IDだけで分類する。 */
  static EpsilonDecisionPointKind ofLegalActions(int[] legalActionIds, int legalActionCount) {
    if (legalActionIds == null
        || legalActionCount < 1
        || legalActionCount > legalActionIds.length) {
      throw new IllegalArgumentException("invalid compact legal-action metadata");
    }
    boolean riichi = false;
    boolean reaction = false;
    for (int slot = 0; slot < legalActionCount; slot++) {
      Action.Type type = Action.fromIndex(legalActionIds[slot]).type();
      riichi |= type.group() == Action.Group.RIICHI;
      reaction |= isReaction(type);
    }
    return of(riichi, reaction);
  }

  private static EpsilonDecisionPointKind of(boolean riichi, boolean reaction) {
    if (riichi) {
      return RIICHI;
    }
    return reaction ? REACTION : DAHAI;
  }

  private static boolean isReaction(Action.Type type) {
    return switch (type.group()) {
      case MELD, KAN, TSUMO, RON, KYUSHU -> true;
      case DAHAI, RIICHI, PASS -> false;
    };
  }
}
