package com.epsilon.reviewer.engine.view;

import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.River;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import java.util.ArrayList;
import java.util.List;

/** 再生の借りた局面を表示の局面へ写す。推論にはこの全情報の値を渡さない。 */
public final class TableProjector {
  private TableProjector() {}

  /**
   * 手牌は牌種の順（赤の 5 が先）、手番の席のツモ牌は手から最初の同じ牌を外して分ける。ドラの赤は表示牌の物理牌、供託は場の棒、手番は今の席。
   */
  public static TableState table(GameState state) {
    List<PlayerState> players = new ArrayList<>(GameState.NUM_PLAYERS);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) players.add(player(state, seat));
    DoraState doraState = state.doraState();
    List<String> dora = new ArrayList<>(doraState.indicatorCount());
    for (int i = 0; i < doraState.indicatorCount(); i++)
      dora.add(
          TileCode.of(
              doraState.indicatorTileType(i), Tile.isAka(state.doraIndicatorPhysicalTileId(i))));
    return new TableState(
        players,
        dora,
        state.honba(),
        state.riichiSticks(),
        state.remainingWallTiles(),
        state.currentPlayer());
  }

  /**
   * 判断の席が鳴いた後の局面。席の目の再生は他家の伏せた牌を持たないので、他家の手とツモ牌だけ判断の時点の表示から取り、判断の席の手と副露・
   * 鳴かれた河の印・手番は再生から取る。
   */
  public static TableState afterCall(GameState state, TableState shown, int seat) {
    TableState called = table(state);
    List<PlayerState> players = new ArrayList<>(called.players());
    for (int player = 0; player < players.size(); player++) {
      if (player == seat) continue;
      PlayerState before = shown.players().get(player);
      players.set(player, players.get(player).withHand(before.hand(), before.draw()));
    }
    return new TableState(
        players,
        called.dora(),
        called.honba(),
        called.kyotaku(),
        called.remaining(),
        called.activeSeat());
  }

  private static PlayerState player(GameState state, int seat) {
    Hand hand = state.hand(seat);
    List<String> tiles = new ArrayList<>(14);
    for (int type = 0; type < Tile.NUM_TILE_TYPES; type++) {
      int count = hand.count(type);
      if (hand.hasAkaTile(type)) {
        tiles.add(TileCode.of(type, true));
        count--;
      }
      for (int i = 0; i < count; i++) tiles.add(TileCode.of(type, false));
    }
    String draw = null;
    if (state.turnEvent() instanceof TurnEvent.Draw event
        && event.player() == seat
        && event.physicalTileId() >= 0) {
      draw = TileCode.of(event.tileType(), event.isAkaTile());
      tiles.remove(draw);
    }
    River discards = state.river(seat);
    List<RiverTile> river = new ArrayList<>(discards.size());
    for (int i = 0; i < discards.size(); i++)
      river.add(
          new RiverTile(
              TileCode.of(discards.tileType(i), discards.aka(i)),
              discards.tsumogiri(i),
              discards.riichiDeclaration(i),
              discards.called(i)));
    List<Meld> melds = new ArrayList<>(hand.meldCount());
    for (int i = 0; i < hand.meldCount(); i++) melds.add(MeldLayout.of(hand, i, seat));
    return new PlayerState(state.score(seat), state.isRiichi(seat), tiles, draw, river, melds);
  }
}
