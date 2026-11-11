package com.epsilon.prism.core;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;

/**
 * 判断の種別。抜いた北への応答は {@link #NUKI_RESPONSE} にする。ほかは合法手の集合から決め、複数に当てはまるときは優先の高い種別を採る
 * （和了 → 九種九牌 → 立直 → 槓 → 鳴き → 北抜き → 鳴いた後の打牌 → 打牌）。番号は行の形式に書かれるので、種別を足すときは末尾に足す。
 */
public enum Kind {
  /** ツモ和了かロン和了が合法（槍槓を含む）。 */
  WIN,
  /** 九種九牌が合法。 */
  KYUSHU,
  /** 立直宣言の打牌が合法。 */
  RIICHI,
  /** 暗槓・加槓・大明槓のいずれかが合法。 */
  KAN,
  /** 打牌への応答で、チーかポンが合法。 */
  CALL,
  /** 鳴いた直後の打牌。 */
  POSTCALL,
  /** 上のどれにも当たらない手番の打牌。 */
  DISCARD,
  /** 北抜きが合法（和了・九種九牌・立直・槓のどれも合法でない手番）。 */
  NUKI,
  /** 抜いた北への応答（ロンか見送り）。 */
  NUKI_RESPONSE;

  private static final Kind[] VALUES = values();

  public static Kind of(int ordinal) {
    return VALUES[ordinal];
  }

  /**
   * 判断の種別を決める。収集・評価・学習で同じ分類を使う。合法手が 1 つの判断にも種別は付くが、選べる手が無いので分岐はしない。
   *
   * @param decision エンジンが求める判断の種類
   * @param legal 合法手の行動の番号
   * @return 判断の種別
   */
  public static Kind classify(EngineDecisionKind decision, int[] legal) {
    if (decision == EngineDecisionKind.NUKI_RESPONSE) return NUKI_RESPONSE;
    boolean riichi = false, kan = false, call = false, kyushu = false, nuki = false;
    for (int code : legal) {
      switch (Action.of(code).type()) {
        case TSUMO_AGARI, RON_AGARI -> {
          return WIN;
        }
        case KYUSHU_KYUHAI -> kyushu = true;
        case RIICHI_DAHAI -> riichi = true;
        case ANKAN, KAKAN, DAIMINKAN -> kan = true;
        case CHI, PON -> call = true;
        case NUKI -> nuki = true;
        default -> {}
      }
    }
    if (kyushu) return KYUSHU;
    if (riichi) return RIICHI;
    if (kan) return KAN;
    if (call) return CALL;
    if (nuki) return NUKI;
    if (decision == EngineDecisionKind.POST_CALL_DAHAI) return POSTCALL;
    return DISCARD;
  }
}
