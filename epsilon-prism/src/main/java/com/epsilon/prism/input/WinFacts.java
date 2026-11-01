package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.prism.input.InputSchema.WinExact;
import com.epsilon.prism.observe.Observation;
import java.util.Arrays;

/**
 * RON・TSUMO の門の道（{@link WinExact}）。O が写したエンジンの即時の和了の判定（点の移動・包・裏の資格・和了の後の順位）から作る。
 *
 * <p>和了の後の点は O の点の移動に本場を足したもの（O のロンの点の移動は本場を含まない。本場の 300 点は {@link WinPointDeltas} の差から取り、包で
 * 放銃者のほかにも払う席があればその席、なければ放銃者が払う。ツモの点の移動は本場を含む）。終局はその点（供託を受け取る前）で
 * {@link HanchanProgression#shouldFinishAfterSingleWin} が決め、順位は O の和了の後の順位（供託を受け取った後）を使う。得点は判断の時点からの
 * 自分の点の変化（本場・供託を含み、裏ドラを含まない）。連荘は親の和了で半荘が続くとき。
 */
final class WinFacts {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final WinExact[] RANKS = {
    WinExact.RANK_1, WinExact.RANK_2, WinExact.RANK_3, WinExact.RANK_4
  };

  private final float[] values = new float[WinExact.values().length];
  private final int[] after = new int[SEATS];
  private final int[] absolute = new int[SEATS];
  private final int[] deltas = new int[SEATS];

  /**
   * 和了 1 つの門の道を作る。
   *
   * @param o 判断の記録
   * @param win O の和了の番号
   */
  void load(Observation o, int win) {
    Arrays.fill(values, 0f);
    for (int seat = 0; seat < SEATS; seat++) {
      after[seat] = o.score(seat) + o.winPayment(win, seat);
    }
    if (o.winSettlement(win) == SettlementAssumption.SOLE_RON && o.honba() > 0) {
      addRonHonba(o, win);
    }
    int observer = o.observer();
    for (int seat = 0; seat < SEATS; seat++) {
      absolute[(observer + seat) % SEATS] = after[seat];
    }
    boolean ends =
        HanchanProgression.shouldFinishAfterSingleWin(
            o.kyoku(),
            o.absoluteSeat(o.dealer()),
            observer,
            absolute[0],
            absolute[1],
            absolute[2],
            absolute[3]);
    int rank = o.winRank(win, 0);
    int gain = after[0] + o.kyotaku() * HanchanProgression.RIICHI_COST - o.score(0);
    set(WinExact.ENDS, ends);
    values[WinExact.UTILITY.ordinal()] = ends ? InputSchema.UTILITY.utilityForRank(rank) : 0f;
    set(RANKS[rank], true);
    values[WinExact.POINTS.ordinal()] = gain;
    set(WinExact.RENCHAN, o.dealer() == 0 && !ends);
    set(WinExact.PAO, o.winPao(win));
    set(WinExact.URA_ELIGIBLE, o.winUraEligible(win));
  }

  /** 作った欄の値（正規化の前）。 */
  float value(WinExact field) {
    return values[field.ordinal()];
  }

  /** 和了の後、供託を受け取る前の相対席の点（終局を決める点）。 */
  int scoreAfter(int seat) {
    return after[seat];
  }

  /** ロンの本場を、和了者と払う席（包で放銃者のほかにも払う席があればその席、なければ放銃者）に足す。 */
  private void addRonHonba(Observation o, int win) {
    WinPointDeltas.settleInto(1, 0, 1, 2, o.honba(), 0, deltas);
    int gain = deltas[0];
    WinPointDeltas.settleInto(1, 0, 1, 2, 0, 0, deltas);
    int honba = gain - deltas[0];
    int discarder = o.lastEventSeat();
    int payer = discarder;
    for (int seat = 1; seat < SEATS; seat++) {
      if (seat != discarder && o.winPayment(win, seat) < 0) {
        payer = seat;
      }
    }
    after[0] += honba;
    after[payer] -= honba;
  }

  private void set(WinExact field, boolean value) {
    values[field.ordinal()] = value ? 1f : 0f;
  }
}
