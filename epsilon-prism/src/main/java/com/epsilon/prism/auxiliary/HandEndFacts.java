package com.epsilon.prism.auxiliary;

import com.epsilon.core.GameState;
import com.epsilon.engine.RoundSettlement;
import java.util.Objects;

/**
 * 経路の局末の事実（席は絶対席）。{@link HandEndAnalyzer} が局末に 1 回作り、その経路の全ての枠（判断の席と判断の時点の出来事の数が違う）を
 * {@link OutcomeCodec} が判断の席から見た 3 語にする。
 *
 * <p>聴牌への到達（{@link #reachedTenpai}・{@link #reachedYakuTenpai}）は、和了した ∨ 局末の手が聴牌 ∨ 最後に聴牌だった出来事の番号が判断の
 * 時点の出来事の数以上。和了した席は局末の聴牌のマスクに入れるので、マスクと番号の 2 つで決まる。
 *
 * @param settlement エンジンの精算（局の結果・局末の点・次局）
 * @param dealer 親の席（和了の点を基本点から作るのに使う。精算からは出せない）
 * @param tenpaiMask 局末の形の聴牌の席のマスク（ビットは席。和了した席を含む。荒牌流局ではエンジンの聴牌の席と同じ）
 * @param yakuTenpaiMask 局末のロンの役がある聴牌の席のマスク（和了した席を含む。{@link TenpaiProbe} の決まり）
 * @param reach 局末の時点の聴牌への到達の写し（書き換えない）
 */
public record HandEndFacts(
    RoundSettlement settlement, int dealer, int tenpaiMask, int yakuTenpaiMask, TenpaiReach reach) {

  private static final int ALL_SEATS = (1 << GameState.NUM_PLAYERS) - 1;

  /**
   * 局末の事実を作る。
   *
   * @throws IllegalArgumentException 親の席かマスクが範囲の外のとき
   */
  public HandEndFacts {
    Objects.requireNonNull(settlement, "settlement");
    Objects.requireNonNull(reach, "reach");
    if (dealer < 0
        || dealer >= GameState.NUM_PLAYERS
        || (tenpaiMask & ~ALL_SEATS) != 0
        || (yakuTenpaiMask & ~ALL_SEATS) != 0) {
      throw new IllegalArgumentException(
          "hand end out of range: dealer="
              + dealer
              + " tenpai="
              + tenpaiMask
              + " yaku="
              + yakuTenpaiMask);
    }
  }

  /**
   * 席が局末に形の聴牌（和了を含む）なら true を返す。
   *
   * @param seat 席
   * @return 聴牌なら true
   */
  public boolean tenpai(int seat) {
    return (tenpaiMask & 1 << Objects.checkIndex(seat, GameState.NUM_PLAYERS)) != 0;
  }

  /**
   * 席が局末にロンの役がある聴牌（和了を含む）なら true を返す。
   *
   * @param seat 席
   * @return 役のある聴牌なら true
   */
  public boolean yakuTenpai(int seat) {
    return (yakuTenpaiMask & 1 << Objects.checkIndex(seat, GameState.NUM_PLAYERS)) != 0;
  }

  /**
   * 判断の後、局の終わりまでに席が聴牌に届いたなら true を返す。
   *
   * @param seat 判断の席
   * @param events 判断の時点の出来事の数（記録 O の頭）
   * @return 届いたなら true
   */
  public boolean reachedTenpai(int seat, int events) {
    return tenpai(seat) || reach.lastTenpaiEvent(seat) >= requireEvents(events);
  }

  /**
   * 判断の後、局の終わりまでに席がロンの役がある聴牌に届いたなら true を返す。
   *
   * @param seat 判断の席
   * @param events 判断の時点の出来事の数（記録 O の頭）
   * @return 届いたなら true
   */
  public boolean reachedYakuTenpai(int seat, int events) {
    return yakuTenpai(seat) || reach.lastYakuTenpaiEvent(seat) >= requireEvents(events);
  }

  private static int requireEvents(int events) {
    if (events < 0) {
      throw new IllegalArgumentException("negative event count: " + events);
    }
    return events;
  }
}
