package com.epsilon.mahjong.engine;

/** 運で決まるイベントの種類。 */
public enum ChanceKind {
  /** 山からのツモ。 */
  DRAW,
  /** 嶺上牌のツモ。槓と北抜きの後の補充。 */
  RINSHAN,
  /** 槓ドラの表示牌。局の始めにめくる表示牌はイベントにしない。 */
  INDICATOR,
  /** 局末にめくる裏ドラ表示牌。 */
  URA
}
