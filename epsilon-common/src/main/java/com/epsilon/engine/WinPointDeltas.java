package com.epsilon.engine;

import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.calculate.scoring.YakuBits;
import com.epsilon.core.GameState;
import java.util.Objects;

/**
 * 和了が 1 件のときの 4 席の点の移動を、基本型だけから求める。局面を持たない呼び出し元（公開の情報から追い越しや待ちの価値を見積もる
 * 符号化など）のための口。
 *
 * <p>和了者は本場と供託を全て受け取る。支払いは {@link ScorePayments} の額に、本場をロンなら放銃者が 300 点、ツモなら支払う席ごとに
 * 100 点足す。実際の精算（{@code WinSettlementCalculator}）と同じ式で、包は扱わない。供託は場から和了者へ移るので、4 席の和は供託の点に
 * なる。
 */
public final class WinPointDeltas {

  /** 基本点の最大（全ての役満が重なった和了）。 */
  public static final int MAX_BASE_POINTS = ScoreMath.basePoints(0, 0, YakuBits.yakumanCount(-1L));

  private WinPointDeltas() {}

  /**
   * 4 席の点の移動を書く。割り当てはしない。
   *
   * @param basePoints 基本点（1〜{@link #MAX_BASE_POINTS}）
   * @param winner 和了者の席
   * @param from 放銃者の席。ツモなら和了者の席（{@link WinClaim#from()} と同じ決まり）
   * @param dealer 親の席
   * @param honba 本場
   * @param kyotaku 供託の本数
   * @param deltas 席の順に点の移動を書く先（受け取りは正、支払いは負）。先頭の 4 要素だけを書く
   * @throws IllegalArgumentException 基本点が範囲外か、本場・供託が負のとき
   * @throws IndexOutOfBoundsException 席が範囲外か、書く先が 4 要素に満たないとき
   */
  public static void settleInto(
      int basePoints, int winner, int from, int dealer, int honba, int kyotaku, int[] deltas) {
    Objects.checkIndex(winner, GameState.NUM_PLAYERS);
    Objects.checkIndex(from, GameState.NUM_PLAYERS);
    Objects.checkIndex(dealer, GameState.NUM_PLAYERS);
    Objects.checkFromIndexSize(0, GameState.NUM_PLAYERS, deltas.length);
    if (basePoints < 1 || basePoints > MAX_BASE_POINTS || honba < 0 || kyotaku < 0) {
      throw new IllegalArgumentException(
          "settlement input out of range: base=" + basePoints + " honba=" + honba + " kyotaku=" + kyotaku);
    }
    boolean dealerWins = winner == dealer;
    int gain = kyotaku * HanchanProgression.RIICHI_COST;
    if (from == winner) {
      int fromChild = ScorePayments.tsumoFromChild(basePoints, dealerWins) + honba * 100;
      int fromDealer = ScorePayments.tsumoFromDealer(basePoints) + honba * 100;
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        if (seat == winner) continue;
        int paid = seat == dealer ? fromDealer : fromChild;
        deltas[seat] = -paid;
        gain += paid;
      }
    } else {
      int paid = ScorePayments.ronPoints(basePoints, dealerWins) + honba * 300;
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) deltas[seat] = 0;
      deltas[from] = -paid;
      gain += paid;
    }
    deltas[winner] = gain;
  }
}
