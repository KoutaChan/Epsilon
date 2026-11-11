package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.VariantScoped;

/** 判断の種類。定数の名前と順は PRISM の語彙の契約なので変えない。一部のルールでだけ使う定数は末尾に足す。 */
public enum EngineDecisionKind implements VariantScoped {
  /** ツモの後の手番。槓と北抜きの後の嶺上牌のツモも含む。 */
  TURN,
  /** 打牌への応答。 */
  RESPONSE,
  /** 加槓への応答。ロンは槍槓になる。 */
  CHANKAN,
  /** チー・ポンの後の打牌。 */
  POST_CALL_DAHAI,
  /** 抜いた北への応答。できるのはロンだけで、槍槓は付かない。 */
  NUKI_RESPONSE(RuleFlag.NUKI);

  private final RuleFlag requires;

  EngineDecisionKind() {
    this(null);
  }

  EngineDecisionKind(RuleFlag requires) {
    this.requires = requires;
  }

  @Override
  public RuleFlag requires() {
    return requires;
  }
}
