package com.epsilon.prism.core;

/**
 * 判断の類 c。ラベル行の率 ρ_c、ロールインの率 β_c、損失の温度 λ_c、認証の集計の単位。種別（{@link Kind}）・段階（{@link Stage}）・公開の脅威の数
 * （{@link Threats#count}）だけで決まる公開情報。
 *
 * <ul>
 *   <li>打牌の類（{@link #DISCARD_EARLY}・{@link #DISCARD_MID}・{@link #DISCARD_LATE}）: 種別が DISCARD か POSTCALL で、脅威のない打牌。段階で分ける。
 *   <li>確約の類（{@link #commitment()}）: {@link #THREAT}（脅威のもとの打牌）・{@link #RIICHI}・{@link #CALL}・{@link #KAN}・{@link #END}。
 * </ul>
 *
 * <p>番号は行の形式（フレームの索引と本体）に入るので、足すときは末尾に足す。
 */
public enum DecisionClass {
  /** 脅威のない序盤の打牌（残り 50 枚以上）。 */
  DISCARD_EARLY,
  /** 脅威のない中盤の打牌（残り 25〜49 枚）。 */
  DISCARD_MID,
  /** 脅威のない終盤の打牌（残り 24 枚以下）。 */
  DISCARD_LATE,
  /** 公開の脅威（立直か副露 2 つ以上の相手がいる）のもとの打牌。 */
  THREAT,
  /** 立直宣言の打牌が合法。 */
  RIICHI,
  /** 応答で CHI か PON が合法（鳴くか見送るか）。 */
  CALL,
  /** 暗槓・加槓・大明槓のいずれかが合法。 */
  KAN,
  /** 局を終える手（ツモ和了・ロン和了・九種九牌）が合法。 */
  END;

  private static final DecisionClass[] VALUES = values();

  /** 類の数。 */
  public static final int COUNT = VALUES.length;

  public static DecisionClass of(int ordinal) {
    return VALUES[ordinal];
  }

  /** 確約の類か（打牌の 3 類のほか）。率・温度は確約の値を使い、認証は確約の類を合わせて判定する。 */
  public boolean commitment() {
    return ordinal() >= THREAT.ordinal();
  }

  /**
   * 判断の類。種別の優先は {@link Kind#classify} のまま（和了・九種 → END、立直 → RIICHI、槓 → KAN、鳴き → CALL）。打牌（DISCARD・POSTCALL）は
   * 脅威があれば THREAT、なければ段階の打牌の類。
   *
   * @param threats 判断の席から見た公開の脅威の数（{@link Threats#count}）
   */
  public static DecisionClass of(Kind kind, Stage stage, int threats) {
    return switch (kind) {
      case WIN, KYUSHU -> END;
      case RIICHI -> RIICHI;
      case KAN -> KAN;
      case CALL -> CALL;
      case DISCARD, POSTCALL ->
          threats > 0
              ? THREAT
              : switch (stage) {
                case EARLY -> DISCARD_EARLY;
                case MID -> DISCARD_MID;
                case LATE -> DISCARD_LATE;
              };
    };
  }
}
