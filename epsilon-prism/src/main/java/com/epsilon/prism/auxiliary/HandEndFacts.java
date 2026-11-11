package com.epsilon.prism.auxiliary;

import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * 経路の局末の事実。席は絶対席で書く。{@link HandEndAnalyzer} が局末に 1 回作り、{@link OutcomeCodec} がその経路の判断ごとに、判断した
 * 席から見た 3 ワードにする。判断ごとに、判断した席と、判断の時点までのイベントの数が違う。エンジンの局末の情報は次の局を始めると
 * 書き換わるので、要る値を写して持つ。
 *
 * <p>判断の後に聴牌に届いたか（{@link #reachedTenpai}・{@link #reachedYakuTenpai}）は、局末の手が聴牌（和了を含む）のとき、または最後に
 * 聴牌だったイベントの番号が判断の時点までのイベントの数以上のときに true になる。和了した席は局末の聴牌のマスクに入れてあるので、マスクと
 * 番号の 2 つで決まる。
 *
 * @param variant ルール
 * @param outcome 局の結果の写し（終わり方、和了者、放銃者、包、基本点）
 * @param yaku 和了ごとの役のビット。並びは局の結果の和了の順で、流局は長さ 0
 * @param dealer 親の席
 * @param next 精算の後の卓の写し。次の局の始まりか終局で、終局の点は残りの供託をトップに渡した後の点
 * @param tenpaiMask 局末に聴牌の席のマスク。ビットは席で、和了した席を含む。荒牌ではエンジンが聴牌とした席と同じ
 * @param yakuTenpaiMask 局末にロンの役がある聴牌の席のマスク。和了した席を含み、判定は {@link TenpaiProbe} に従う
 * @param reach 局末の時点の聴牌への到達の写し（書き換えない）
 */
public record HandEndFacts(
    Variant variant,
    RoundOutcome outcome,
    long[] yaku,
    int dealer,
    TableState next,
    int tenpaiMask,
    int yakuTenpaiMask,
    TenpaiReach reach) {

  /**
   * 局末の事実を作る。
   *
   * @throws IllegalArgumentException ルールが食い違うか、親の席・マスク・役の数が範囲の外のとき
   */
  public HandEndFacts {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(reach, "reach");
    if (outcome.variant() != variant || next.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: " + variant + ", " + outcome + ", " + next);
    }
    int seats = variant.seatMask();
    if (dealer < 0
        || dealer >= variant.seats()
        || (tenpaiMask & ~seats) != 0
        || (yakuTenpaiMask & ~seats) != 0
        || yaku.length != outcome.winnerCount()) {
      throw new IllegalArgumentException(
          "Hand end out of range: dealer=" + dealer + " tenpai=" + tenpaiMask + " yaku=" + yakuTenpaiMask
              + " winners=" + yaku.length + "/" + outcome.winnerCount());
    }
  }

  /**
   * 席が局末に聴牌なら true を返す。和了した席も true。
   *
   * @param seat 席
   * @return 聴牌なら true
   */
  public boolean tenpai(int seat) {
    return (tenpaiMask & 1 << Objects.checkIndex(seat, variant.seats())) != 0;
  }

  /**
   * 席が局末にロンの役がある聴牌なら true を返す。和了した席も true。
   *
   * @param seat 席
   * @return 役のある聴牌なら true
   */
  public boolean yakuTenpai(int seat) {
    return (yakuTenpaiMask & 1 << Objects.checkIndex(seat, variant.seats())) != 0;
  }

  /**
   * 判断の後、局の終わりまでに席が聴牌に届いたなら true を返す。
   *
   * @param seat 判断した席
   * @param events 判断の時点までのイベントの数（記録 O のヘッダーにある値）
   * @return 届いたなら true
   */
  public boolean reachedTenpai(int seat, int events) {
    return tenpai(seat) || reach.lastTenpaiEvent(seat) >= requireEvents(events);
  }

  /**
   * 判断の後、局の終わりまでに席がロンの役がある聴牌に届いたなら true を返す。
   *
   * @param seat 判断した席
   * @param events 判断の時点までのイベントの数（記録 O のヘッダーにある値）
   * @return 届いたなら true
   */
  public boolean reachedYakuTenpai(int seat, int events) {
    return yakuTenpai(seat) || reach.lastYakuTenpaiEvent(seat) >= requireEvents(events);
  }

  private static int requireEvents(int events) {
    if (events < 0) {
      throw new IllegalArgumentException("Negative event count: " + events);
    }
    return events;
  }
}
