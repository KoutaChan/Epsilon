package com.epsilon.reviewer.engine.view;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;

/**
 * 候補の行に書く鳴きと槓の牌（protocol/result.ts の CALL_TYPES）。他の手の牌は行動 ID で決まる。チーの鳴いた牌は捨て牌が赤なら赤、手の牌は手の赤を
 * 使う手なら赤。ポン・槓は赤の 1 枚を先に置く。
 */
public final class CandidateTiles {
  private CandidateTiles() {}

  /** 鳴き・槓でない手は null。 */
  public static String[] of(Action action, GameState state, int seat) {
    int type = action.tileType();
    return switch (action.type()) {
      case CHI -> chi(action, state, type);
      case PON, DAIMINKAN, ANKAN, KAKAN -> set(action, state, seat, type);
      default -> null;
    };
  }

  private static String[] chi(Action action, GameState state, int called) {
    boolean calledRed =
        state.turnEvent() instanceof TurnEvent.Discard discard && discard.isAkaTile();
    int[] types = action.chiTileTypes();
    String[] tiles = new String[types.length];
    for (int i = 0; i < types.length; i++) {
      int type = types[i];
      tiles[i] =
          TileCode.of(
              type,
              Tile.canBeAka(type) && (type == called ? calledRed : action.usesAkaTileFromHand()));
    }
    return tiles;
  }

  private static String[] set(Action action, GameState state, int seat, int type) {
    boolean red =
        switch (action.type()) {
          case PON ->
              action.usesAkaTileFromHand()
                  || state.turnEvent() instanceof TurnEvent.Discard discard && discard.isAkaTile();
          case KAKAN ->
              state.hand(seat).hasAkaTile(type)
                  || state.hand(seat).requirePon(type).containsAkaTile();
          default ->
              state.hand(seat).hasAkaTile(type)
                  || state.turnEvent() instanceof TurnEvent.Discard discard
                      && discard.tileType() == type
                      && discard.isAkaTile();
        };
    String[] tiles = new String[action.type() == Action.Type.PON ? 3 : 4];
    for (int i = 0; i < tiles.length; i++) tiles[i] = TileCode.of(type, red && i == 0);
    return tiles;
  }
}
