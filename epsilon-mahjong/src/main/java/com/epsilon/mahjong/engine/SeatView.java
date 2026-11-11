package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;

/**
 * 1 席から見た卓。状態をコピーせずに読むので、卓を次に進める呼び出しまで有効。他家の伏せた牌、牌山、まだめくっていない表示牌、
 * 裏ドラ表示牌は読めない。席は起家を 0 とする絶対席で、牌は種類で表す。種類の集合は 34 ビットのマスク。
 */
public interface SeatView {

  Variant variant();

  /** 見る席。 */
  int seat();

  /** この判断の時点で、見る席がする判断の種類。見る席に判断が無ければ IllegalStateException を投げる。 */
  EngineDecisionKind kind();

  /** 局の通し番号（{@link Variant} の並び）。 */
  int round();

  int dealer();

  /**
   * 手番の席。直前にツモか鳴きをした席で、局の始めは親。他家の応答を求めている間は、打牌・加槓・北抜きをした席。
   */
  int current();

  int honba();

  int kyotaku();

  int score(int seat);

  /** 王牌を除いた山の残りの枚数。 */
  int liveLeft();

  EventLog events();

  /** この判断の時点の直前のイベント。他家のツモは伏せて見せる（牌は {@link TurnEvent#NONE}、赤は false）。 */
  TurnEvent last();

  /** 自分の純手牌にあるその種類の枚数。 */
  int count(int kind);

  /** 自分の純手牌にその物理牌があるなら true。 */
  boolean holds(int physicalId);

  int meldCount(int seat);

  /**
   * 面子を表す正準の番号（{@code Meld.id()}）。加槓したポンは、加槓の面子の番号になる。
   *
   * @param seat 席
   * @param index 何番目の面子か
   */
  int meld(int seat, int index);

  int nuki(int seat);

  int indicatorCount();

  /**
   * その表示牌の物理 id。
   *
   * @param index めくった順の何番目か
   */
  int indicator(int index);

  /**
   * 表示牌をめくった時点での公開イベントの数。その表示牌より前にあるイベントの数と同じ。
   *
   * @param index めくった順の何番目か
   */
  int indicatorEvent(int index);

  /** めくった表示牌のうち、その種類をドラにするものの数。 */
  int doraWeight(int kind);

  RiichiState riichi(int seat);

  boolean ippatsu(int seat);

  /** その種類の見えている枚数（河・面子・めくった表示牌・抜いた北）。自分の純手牌は数えない。 */
  int visible(int kind);

  /** その席の河にある牌の種類の集合（鳴かれた牌も含む）。 */
  long riverKinds(int seat);

  /** その席の立直が成立した後に、誰かが打った牌の種類の集合。 */
  long riichiSafe(int seat);

  /** 自分が見逃しの振聴（同巡か立直の後）なら true。河の振聴は含まない。 */
  boolean furiten();

  /** 自分の形の上の待ち（役と振聴は見ない）。手が 13 枚相当のときだけ返し、14 枚の間は 0。 */
  long waits();
}
