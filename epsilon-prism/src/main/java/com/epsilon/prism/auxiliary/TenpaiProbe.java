package com.epsilon.prism.auxiliary;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.calculate.shape.HandShapeView;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/**
 * 席の手が聴牌か、ロンの役がある聴牌かを確かめる（{@link TenpaiReach}・{@link HandEndAnalyzer}）。持ち主のスレッドだけで使い、割り当てを
 * しない。
 *
 * <p>聴牌は形の聴牌（最小向聴 ≤ 0。エンジンの荒牌流局の聴牌と同じ決まり）。役は公開の局況（自風・場風）だけで確かめ、ドラ・赤・振聴・偶然の
 * 役（一発・海底・河底・槍槓・嶺上）を見ない。この局で立直を宣言した席（宣言牌でロンされて立直が成り立たなかった席を含む）は立直が役なので、
 * 聴牌ならそのまま役のある聴牌とする。14 枚の手は、1 枚捨ててそうなれるなら聴牌（役のある聴牌）とする。
 */
final class TenpaiProbe {

  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandScoreEvaluator scores = new HandScoreEvaluator();
  private final HandShapeState discarded = new HandShapeState();

  /**
   * 手が形の聴牌なら true を返す。
   *
   * @param hand 13 枚か 14 枚の手
   * @return 最小向聴が 0 以下なら true
   */
  boolean tenpai(HandShapeView hand) {
    return shapes.calculateMinimum(hand) <= 0;
  }

  /**
   * 席の手がロンの役がある聴牌なら true を返す。待ちごとに役を確かめ、見つかった時点で止める。
   *
   * @param state 局面
   * @param seat 席
   * @return 役のある聴牌なら true
   * @throws IllegalStateException 手が 13 枚でも 14 枚でもないとき
   */
  boolean yakuTenpai(GameState state, int seat) {
    Hand hand = state.hand(seat);
    if (state.river(seat).riichiDeclarationIndex() >= 0) {
      return tenpai(hand);
    }
    WinConditions conditions =
        WinConditions.publicDecision(state.getJikaze(seat), state.getBakaze(), RiichiState.NONE);
    return switch (hand.effectiveTileCount()) {
      case 13 -> shapes.calculateMinimum(hand) == 0 && hasRonYaku(hand, hand, conditions);
      case 14 -> afterDiscard(hand, conditions);
      default ->
          throw new IllegalStateException(
              "seat " + seat + " holds " + hand.effectiveTileCount() + " effective tiles");
    };
  }

  /** 14 枚の手から 1 枚捨てて、ロンの役がある聴牌になれるなら true。 */
  private boolean afterDiscard(HandView hand, WinConditions conditions) {
    discarded.load(hand);
    for (long tiles = hand.concealedTileTypeMask(); tiles != 0L; tiles &= tiles - 1) {
      int tile = Long.numberOfTrailingZeros(tiles);
      discarded.remove(tile);
      boolean found =
          shapes.calculateMinimum(discarded) == 0 && hasRonYaku(discarded, hand, conditions);
      discarded.add(tile);
      if (found) {
        return true;
      }
    }
    return false;
  }

  /**
   * 13 枚の聴牌の形の待ちのどれかで、ロンの役があるなら true。手と副露で 4 枚とも持っている牌の待ち（空聴）は 5 枚目がなく和了れないので
   * 見ない（採点は 5 枚目を例外で拒む。エンジンのロンの判定も同じ牌を除く）。
   */
  private boolean hasRonYaku(HandShapeView shape, HandView melds, WinConditions conditions) {
    for (long waits = shapes.agariTileTypeMask(shape); waits != 0L; waits &= waits - 1) {
      int wait = Long.numberOfTrailingZeros(waits);
      if (held(shape, melds, wait) < Tile.TILES_PER_TYPE
          && scores.probeAfterAdding(shape, melds, wait, HandScoreEvaluator.RON, conditions) != 0) {
        return true;
      }
    }
    return false;
  }

  /** 手（shape）と副露（melds の面子）が持つ牌 tile の物理の枚数。 */
  private static int held(HandShapeView shape, HandView melds, int tile) {
    int held = shape.count(tile);
    for (int index = 0, count = melds.meldCount(); index < count; index++) {
      Meld meld = melds.meld(index);
      for (int at = 0, size = meld.size(); at < size; at++) {
        if (meld.tileAt(at) == tile) {
          held++;
        }
      }
    }
    return held;
  }
}
