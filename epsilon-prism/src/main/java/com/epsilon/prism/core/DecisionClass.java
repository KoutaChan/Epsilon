package com.epsilon.prism.core;

import com.epsilon.mahjong.rules.Variant;

/**
 * 判断のクラス c。ラベル行の率 ρ_c、ロールインの率 β_c、損失の温度 λ_c、運の精算の係数 β_c は、このクラスごとに持つ。クラスは種別（{@link Kind}）・
 * 段階（{@link Stage}）・公開の脅威の数（{@link Threats#count}）・局の番号だけで決まるので、公開情報から求められる。
 *
 * <ul>
 *   <li>打牌のクラス（{@link #DISCARD_EARLY}・{@link #DISCARD_MID}・{@link #DISCARD_LATE}）: 打牌か鳴いた後の打牌で、脅威がないもの。段階で 3 つに分ける。
 *   <li>確約のクラス（{@link #commitment()}）: {@link #THREAT}（脅威のもとの打牌）・{@link #RIICHI}・{@link #CALL}・{@link #KAN}・{@link #END}・
 *       {@link #FINAL_END}・{@link #NUKI}・{@link #NUKI_RESPONSE}。
 * </ul>
 *
 * <p>オーラス（飛びでなくても、この局が終われば半荘が終わりうる局。南場の最後の局と西入の局）の和了は {@link #FINAL_END} に分け、ラベル行の率を
 * ほかの確約のクラスと別に持つ（既定では全てラベル行にする）。オーラスの和了は半荘の行方を決める判断で、和了して半荘を終えた分岐の値は、推定ではなく
 * 確定した順位の効用になる。九種九牌は局をやり直すだけで半荘を終えないので、オーラスでも {@link #END} に入れる。
 *
 * <p>番号は行の形式（フレームのインデックスと本体）に書かれるので、クラスを足すときは末尾に足す。
 */
public enum DecisionClass {
  /** 脅威のない序盤の打牌（{@link Stage#EARLY}）。 */
  DISCARD_EARLY,
  /** 脅威のない中盤の打牌（{@link Stage#MID}）。 */
  DISCARD_MID,
  /** 脅威のない終盤の打牌（{@link Stage#LATE}）。 */
  DISCARD_LATE,
  /** 公開の脅威（立直か副露 2 つ以上の相手がいる）のもとの打牌。 */
  THREAT,
  /** 立直宣言の打牌が合法。 */
  RIICHI,
  /** 打牌への応答で、チーかポンが合法（鳴くか見送るか）。 */
  CALL,
  /** 暗槓・加槓・大明槓のいずれかが合法。 */
  KAN,
  /** 局を終える手（ツモ和了・ロン和了・九種九牌）が合法。オーラスの和了は {@link #FINAL_END} に入る。 */
  END,
  /** オーラスで和了（ツモ和了・ロン和了）が合法。 */
  FINAL_END,
  /** 北抜きが合法（和了・九種九牌・立直・槓のどれも合法でない手番）。 */
  NUKI,
  /** 抜いた北への応答（ロンか見送り）。オーラスでは {@link #FINAL_END} に入る。 */
  NUKI_RESPONSE;

  private static final DecisionClass[] VALUES = values();

  public static final int COUNT = VALUES.length;

  public static DecisionClass of(int ordinal) {
    return VALUES[ordinal];
  }

  /** 確約のクラスなら true。脅威のない打牌の 3 クラスのほかは、全て確約のクラス。率と温度には確約の値を使う。 */
  public boolean commitment() {
    return ordinal() >= THREAT.ordinal();
  }

  /**
   * 判断のクラスを返す。種別の優先は {@link Kind#classify} と同じ。和了は END（オーラスなら FINAL_END）、九種九牌は END、立直は RIICHI、
   * 槓は KAN、鳴きは CALL、北抜きは NUKI、抜いた北への応答は NUKI_RESPONSE（オーラスなら FINAL_END）になる。打牌と鳴いた後の打牌は、
   * 脅威があれば THREAT、なければ段階ごとの打牌のクラスになる。
   *
   * @param variant 四人打ちか三人打ちか
   * @param kind 判断の種別
   * @param stage 局の段階
   * @param threats 判断の席から見た公開の脅威の数（{@link Threats#count}）
   * @param round 局の番号（東 1 局を 0 とする通算。南場の最後の局とその後の局がオーラス）
   * @return 判断のクラス
   */
  public static DecisionClass of(Variant variant, Kind kind, Stage stage, int threats, int round) {
    return switch (kind) {
      case WIN -> round >= variant.lastRound() ? FINAL_END : END;
      case KYUSHU -> END;
      case RIICHI -> RIICHI;
      case KAN -> KAN;
      case CALL -> CALL;
      case NUKI -> NUKI;
      case NUKI_RESPONSE -> round >= variant.lastRound() ? FINAL_END : NUKI_RESPONSE;
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
