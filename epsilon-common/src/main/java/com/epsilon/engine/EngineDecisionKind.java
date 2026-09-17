package com.epsilon.engine;

/** エンジンが行動の選択を待つ場面の種類。 */
public enum EngineDecisionKind {
  /** 自摸後の通常手番判断。 */
  TURN,
  /** 他家打牌に対するRON・PASS・副露判断。 */
  RESPONSE,
  /** 槓宣言に対する槍槓RON判断。 */
  CHANKAN,
  /** CHI・PON成立直後の強制打牌判断。 */
  POST_CALL_DAHAI
}
