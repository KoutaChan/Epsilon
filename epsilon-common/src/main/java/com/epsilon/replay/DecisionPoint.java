package com.epsilon.replay;

/**
 * 適用前の判断と、元イベント・状態改訂の対応。
 *
 * @param eventIndex 選択が解決するイベント位置。未解決のEOFではイベント数
 * @param stateId 借用状態の単調増加する改訂番号。内部のフリテン・槓確定も含む
 * @param causeEventIndex 判断を開始したツモ・副露・打牌・加槓のイベント位置
 * @param chosenSlot 合法手の選択位置。未観測なら-1
 * @param choiceKind 記録された行動、確定した見送り、未観測の区別
 */
public record DecisionPoint(
    int eventIndex, int stateId, int causeEventIndex, int chosenSlot, ChoiceKind choiceKind) {
  /** 未観測の応答を、記録された教師へ変換しないための区別。 */
  public enum ChoiceKind {
    RECORDED_ACTION,
    CONFIRMED_PASS,
    UNOBSERVED
  }
}
