package com.epsilon.prism.auxiliary;

import com.epsilon.core.GameState;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.WinClaim;
import java.util.List;

/**
 * 経路の局末の事実（{@link HandEndFacts}）を作る。枠のある経路の局末（精算の後、次局を始める前。4 人の手は局末のまま）に 1 回呼ぶ。進行
 * ワーカーに 1 つ持ち、作業域は中に持つ。
 *
 * <p>4 席の局末の聴牌と役のある聴牌を {@link TenpaiProbe} の決まりで確かめる。和了した席は確かめずに両方に入れる（偶然の役だけの和了も、
 * 役のある和了に届いたものとして数える）。局末の手は和了した席のほかは 13 枚で、九種九牌を宣言した席だけが 14 枚。
 */
public final class HandEndAnalyzer {

  private final TenpaiProbe probe = new TenpaiProbe();

  /**
   * 局末の事実を作る。
   *
   * @param state 局末の局面
   * @param settlement その局の精算
   * @param reach その経路の聴牌への到達（写して残すので、呼んだ後に次局のために消してよい）
   * @return 局末の事実
   */
  public HandEndFacts analyze(GameState state, RoundSettlement settlement, TenpaiReach reach) {
    int winners = winners(settlement.result());
    int tenpai = winners;
    int yakuTenpai = winners;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      if ((winners & 1 << seat) != 0 || !probe.tenpai(state.hand(seat))) {
        continue;
      }
      tenpai |= 1 << seat;
      if (probe.yakuTenpai(state, seat)) {
        yakuTenpai |= 1 << seat;
      }
    }
    return new HandEndFacts(settlement, state.getOya(), tenpai, yakuTenpai, new TenpaiReach(reach));
  }

  /** 和了した席のマスク（流局は 0）。 */
  private static int winners(RoundResult result) {
    if (!(result instanceof RoundResult.Winning winning)) {
      return 0;
    }
    List<? extends WinClaim> claims = winning.claims();
    int mask = 0;
    for (int index = 0, size = claims.size(); index < size; index++) {
      mask |= 1 << claims.get(index).winner();
    }
    return mask;
  }
}
