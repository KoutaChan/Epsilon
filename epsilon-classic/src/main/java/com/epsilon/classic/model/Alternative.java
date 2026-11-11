package com.epsilon.classic.model;

import com.epsilon.classic.input.ClassicVocabulary.ActionType;

/**
 * 方策ヘッドが採点する分岐のスコア。番号（ordinal）はスコアの列の位置で、分岐の埋め込みの行でもある。変換した重みはこの名前と
 * 並びに合わせてあるので、名前と並びは変えない。二択（ロン・ツモ・九種九牌・鳴き・槓）では選ぶ側だけを採点し、見送る側・続ける側は
 * スコアを持たない。種類の分岐（チー・ポン・大明槓、暗槓・加槓）では種類ごとに採点する。
 */
public enum Alternative {
  RON(PolicyNode.RON_GATE, ActionType.RON_AGARI),
  TSUMO(PolicyNode.TSUMO_GATE, ActionType.TSUMO_AGARI),
  KYUSHU(PolicyNode.KYUSHU_GATE, ActionType.KYUSHU_KYUHAI),
  /** 鳴く側の全体。 */
  CALL(PolicyNode.CALL_GATE, null),
  CHI(PolicyNode.MELD_TYPE, ActionType.CHI),
  PON(PolicyNode.MELD_TYPE, ActionType.PON),
  DAIMINKAN(PolicyNode.MELD_TYPE, ActionType.DAIMINKAN),
  /** 槓をする側の全体。 */
  KAN(PolicyNode.KAN_GATE, null),
  ANKAN(PolicyNode.KAN_TYPE, ActionType.ANKAN),
  KAKAN(PolicyNode.KAN_TYPE, ActionType.KAKAN);

  private final PolicyNode node;
  private final ActionType type;

  Alternative(PolicyNode node, ActionType type) {
    this.node = node;
    this.type = type;
  }

  /** このスコアで枝を選ぶノード。 */
  public PolicyNode node() {
    return node;
  }

  /** このスコアがまとめる合法手の種別。二択で選ぶ側の全体（鳴く側・槓をする側）なら null。 */
  public ActionType type() {
    return type;
  }
}
