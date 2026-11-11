package com.epsilon.prism.auxiliary;

import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * 経路の局末の事実（{@link HandEndFacts}）を作る。記録する判断を含む経路が局末に着いたら、エンジンが局末にいる間（全員の手が局末のまま）に
 * 1 回呼ぶ。
 *
 * <p>局末に聴牌とする席は、和了した席にほかの聴牌の席を足したもの。荒牌ではエンジンが聴牌とした席を、ほかの終わり方では
 * {@link TenpaiProbe} が聴牌と判定した席を足す。九種九牌を宣言した 14 枚の手は、1 枚捨てて聴牌になれるなら聴牌とする。荒牌でエンジンの
 * 聴牌に合わせるのは、流局の聴牌・不聴を精算と食い違わせないため。和了していない聴牌の席は、ロンの役がある聴牌かを {@link TenpaiProbe} で
 * 確かめる。和了した席は確かめずに、聴牌とロンの役がある聴牌の両方に入れる。偶然の役だけで和了した席も、役のある和了に届いたものとして
 * 数える。
 */
public final class HandEndAnalyzer {

  private HandEndAnalyzer() {}

  /**
   * 局末の事実を作る。
   *
   * @param full 局末の局面
   * @param end エンジンが出した局末の結果
   * @param reach その経路の聴牌への到達。写して残すので、呼んだ後は次の局のために消してよい
   * @return 局末の事実
   */
  public static HandEndFacts analyze(FullView full, RoundEnd end, TenpaiReach reach) {
    Variant variant = full.variant();
    RoundOutcome outcome = end.outcome();
    int winners = 0;
    long[] yaku = new long[outcome.winnerCount()];
    for (int index = 0; index < yaku.length; index++) {
      winners |= 1 << outcome.winner(index);
      yaku[index] = end.score(index).yaku();
    }
    TenpaiProbe probe = TenpaiProbe.of(variant);
    int tenpai = winners;
    if (outcome.kind() == EndKind.EXHAUSTIVE) {
      tenpai |= outcome.tenpaiMask();
    } else {
      for (int seat = 0; seat < variant.seats(); seat++) {
        if ((winners & 1 << seat) == 0 && probe.tenpai(full, seat)) {
          tenpai |= 1 << seat;
        }
      }
    }
    int yakuTenpai = winners;
    for (int rest = tenpai & ~winners; rest != 0; rest &= rest - 1) {
      int seat = Integer.numberOfTrailingZeros(rest);
      if (probe.yakuTenpai(full, seat)) {
        yakuTenpai |= 1 << seat;
      }
    }
    RoundOutcome result = new RoundOutcome(variant);
    result.copyFrom(outcome);
    TableState next = new TableState(variant);
    next.copyFrom(end.next());
    return new HandEndFacts(
        variant, result, yaku, full.seat(0).dealer(), next, tenpai, yakuTenpai, new TenpaiReach(reach));
  }
}
