package com.epsilon.classic.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.score.RiichiState;

/** 入力のフィールドとルールベースの打ち手が共有する局のルール。判断の記録から巡目を数え、立直を宣言したときの立直の状態を決める。 */
public final class RoundRules {

  private RoundRules() {}

  /**
   * 打牌が巡目を進めるか。巡目は、ツモの後の打牌の数を席の数で割って切り捨てた値。チー・ポンの直後にその席が打つ牌（ツモの無い
   * 打牌）は数えない。
   *
   * @param o 判断の記録
   * @param event 打牌のイベントの番号
   * @return ツモの後の打牌なら true
   */
  public static boolean afterDraw(Observation o, int event) {
    int previous = event - 1;
    if (previous < 0 || o.eventKind(previous) != EventKind.MELD || o.eventAuthor(previous) != o.eventAuthor(event)) {
      return true;
    }
    Meld.Type type = o.meld(previous).type();
    return type != Meld.Type.CHI && type != Meld.Type.PON;
  }

  /**
   * 手番の打牌が、下家からポン・大明槓で鳴いた後の最初の打牌か（大明槓なら嶺上のツモの後の打牌）。立直していない席の見逃しの
   * 振聴は自分の打牌で消えるが、この打牌でだけは消えずに残る。
   *
   * @param o 手番の判断の記録（ツモの後か、チー・ポンの後の打牌）
   * @return 下家から鳴いた後、まだ打牌していなければ true
   */
  public static boolean afterCallFromShimocha(Observation o) {
    for (int event = o.events() - 1; event >= 0; event--) {
      if (o.eventAuthor(event) != 0 || o.eventKind(event) == EventKind.DISCARD) {
        return false;
      }
      if (o.eventKind(event) == EventKind.MELD && calledFromShimocha(o.meld(event))) {
        return true;
      }
    }
    return false;
  }

  /** 下家の打牌を鳴いてできた面子か（ポンか大明槓）。加槓は元のポンの鳴いた相手を持つが、鳴きではないので含めない。 */
  public static boolean calledFromShimocha(Meld meld) {
    Meld.Type type = meld.type();
    return (type == Meld.Type.PON || type == Meld.Type.DAIMINKAN) && meld.source() == Meld.RelativeSource.SHIMOCHA;
  }

  /**
   * 立直を宣言したときの立直の状態。宣言牌が第一打で、局にまだ副露（暗槓・加槓を含む）が無ければダブル立直、そうでなければ立直。
   *
   * @param firstDiscard 宣言牌がその席の第一打か
   * @param interrupted 宣言より前に、局に副露があったか
   * @return ダブル立直か立直
   */
  public static RiichiState declaration(boolean firstDiscard, boolean interrupted) {
    return firstDiscard && !interrupted ? RiichiState.DOUBLE_RIICHI : RiichiState.RIICHI;
  }
}
