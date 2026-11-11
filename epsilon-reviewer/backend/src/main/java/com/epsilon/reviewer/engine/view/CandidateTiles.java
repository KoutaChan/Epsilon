package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 候補の行に書く、鳴きと槓に使う牌（protocol/result.ts の CALL_TYPES の手）。ほかの手の牌は行動 ID だけで決まる。チーでは、鳴いた
 * 牌は捨て牌が赤なら赤にし、手牌から出す牌は赤 5 を使うチーなら赤にする。ポン・槓では赤の 1 枚を先頭に置く。
 */
public final class CandidateTiles {
  private CandidateTiles() {}

  /**
   * @param view 判断する席から見える卓
   * @return 鳴き・槓に使う牌。鳴き・槓でない手は null
   */
  public static String[] of(Action action, SeatView view) {
    return switch (action.type()) {
      case CHI -> chi(action, view);
      case PON, DAIMINKAN, ANKAN, KAKAN -> set(action, view);
      default -> null;
    };
  }

  private static String[] chi(Action action, SeatView view) {
    boolean calledRed = discardedRed(view);
    String[] tiles = new String[3];
    for (int i = 0; i < tiles.length; i++) {
      int kind = action.base() + i;
      boolean red = kind == action.tile() ? calledRed : action.selection().red();
      tiles[i] = TileCode.of(kind, red && Tiles.contains(Tiles.FIVES, kind));
    }
    return tiles;
  }

  private static String[] set(Action action, SeatView view) {
    int kind = action.tile();
    boolean red =
        switch (action.type()) {
          case PON -> action.selection().red() || discardedRed(view);
          case KAKAN -> holdsRed(view, kind) || ponRed(view, kind);
          default ->
              holdsRed(view, kind)
                  || view.last().kind() == TurnEvent.Kind.DISCARD
                      && view.last().tile() == kind
                      && view.last().red();
        };
    String[] tiles = new String[action.type() == Action.Type.PON ? 3 : 4];
    for (int i = 0; i < tiles.length; i++) tiles[i] = TileCode.of(kind, red && i == 0);
    return tiles;
  }

  private static boolean discardedRed(SeatView view) {
    return view.last().kind() == TurnEvent.Kind.DISCARD && view.last().red();
  }

  /** 純手牌に、その種類の赤 5 があるか。 */
  private static boolean holdsRed(SeatView view, int kind) {
    int red = Tiles.physicalId(kind, Tiles.RED_COPY);
    return view.variant().tiles().red(red) && view.holds(red);
  }

  /** 加槓する種類のポンに赤が含まれるか。 */
  private static boolean ponRed(SeatView view, int kind) {
    int seat = view.seat();
    for (int i = 0; i < view.meldCount(seat); i++) {
      var meld = com.epsilon.mahjong.action.Meld.of(view.meld(seat, i));
      if (meld.type() == com.epsilon.mahjong.action.Meld.Type.PON && meld.base() == kind)
        return meld.red();
    }
    throw new IllegalArgumentException("Seat " + seat + " has no pon of kind " + kind);
  }
}
