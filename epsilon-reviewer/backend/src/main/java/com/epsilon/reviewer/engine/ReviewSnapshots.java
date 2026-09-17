package com.epsilon.reviewer.engine;

import com.epsilon.core.*;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.reviewer.dto.*;
import java.util.ArrayList;
import java.util.List;

/** 牌譜再生の借用状態を UI 専用の値へ変換する。推論にはこの全情報 DTO を渡さない。 */
final class ReviewSnapshots {
  private static final String[] HONORS = {"E", "S", "W", "N", "P", "F", "C"};

  static TableState table(GameState state) {
    List<PlayerState> players = new ArrayList<>(4);
    for (int seat = 0; seat < 4; seat++) {
      Hand hand = state.hand(seat);
      List<String> tiles = new ArrayList<>();
      for (int type = 0; type < 34; type++) {
        int count = hand.count(type);
        if (hand.hasAkaTile(type)) {
          tiles.add(tile(type, true));
          count--;
        }
        for (int i = 0; i < count; i++) tiles.add(tile(type, false));
      }
      String draw = null;
      if (state.turnEvent() instanceof TurnEvent.Draw event
          && event.player() == seat
          && event.physicalTileId() >= 0) {
        draw = tile(event.tileType(), event.isAkaTile());
        tiles.remove(draw);
      }
      List<RiverTile> river = new ArrayList<>();
      for (int i = 0; i < state.river(seat).size(); i++) {
        var discard = state.river(seat).discard(i);
        river.add(
            new RiverTile(
                tile(discard.tileType(), discard.aka()),
                discard.tsumogiri(),
                discard.riichiDeclaration(),
                discard.called()));
      }
      List<MeldSnapshot> melds = new ArrayList<>();
      for (int i = 0; i < hand.meldCount(); i++) melds.add(meld(hand.meld(i), seat));
      players.add(
          new PlayerState(state.score(seat), state.isRiichi(seat), tiles, draw, river, melds));
    }
    List<String> dora = new ArrayList<>();
    for (int i = 0; i < state.doraState().indicatorCount(); i++)
      dora.add(
          tile(
              state.doraState().indicatorTileType(i),
              Tile.isAka(state.doraIndicatorPhysicalTileId(i))));
    return new TableState(
        players,
        dora,
        state.honba(),
        state.riichiSticks(),
        state.remainingWallTiles(),
        state.currentPlayer());
  }

  /** 適用直後に残る和了元の物理牌を使い、赤牌と複数ロンの個別精算を保持する。 */
  static RoundOutcome snapshotRoundOutcome(ReplayEvent event, GameState state) {
    return switch (event) {
      case ReplayEvent.Hora hora -> {
        TurnEvent.TileEvent source = (TurnEvent.TileEvent) state.turnEvent();
        yield new RoundOutcome(
            "hora",
            hora.actor(),
            hora.target(),
            tile(source.tileType(), source.isAkaTile()),
            copyRecordedScoreDeltas(hora.deltas()),
            hora.details());
      }
      case ReplayEvent.Ryukyoku draw ->
          new RoundOutcome(
              "ryukyoku", null, null, null, copyRecordedScoreDeltas(draw.deltas()), null);
      default -> null;
    };
  }

  private static List<Integer> copyRecordedScoreDeltas(int[] deltas) {
    if (deltas == null) return null;
    List<Integer> values = new ArrayList<>(deltas.length);
    for (int delta : deltas) values.add(delta);
    return values;
  }

  static ActionCandidate candidate(
      Action action, float probability, boolean chosen, GameState state, int seat) {
    List<String> tiles = new ArrayList<>();
    int type = action.tileType();
    switch (action.type()) {
      case DAHAI, RIICHI_DAHAI -> tiles.add(tile(type, action.usesAkaTileFromHand()));
      case CHI -> {
        boolean calledRed = state.turnEvent() instanceof TurnEvent.Discard d && d.isAkaTile();
        for (int t : action.chiTileTypes())
          tiles.add(
              tile(t, Tile.canBeAka(t) && (t == type ? calledRed : action.usesAkaTileFromHand())));
      }
      case PON, DAIMINKAN, ANKAN, KAKAN -> {
        boolean hasRed =
            switch (action.type()) {
              case PON ->
                  action.usesAkaTileFromHand()
                      || state.turnEvent() instanceof TurnEvent.Discard d && d.isAkaTile();
              case KAKAN ->
                  state.hand(seat).hasAkaTile(type)
                      || state.hand(seat).requirePon(type).containsAkaTile();
              default ->
                  state.hand(seat).hasAkaTile(type)
                      || state.turnEvent() instanceof TurnEvent.Discard d
                          && d.tileType() == type
                          && d.isAkaTile();
            };
        int count = action.type() == Action.Type.PON ? 3 : 4;
        for (int i = 0; i < count; i++) tiles.add(tile(type, hasRed && i == 0));
      }
      default -> {}
    }
    return new ActionCandidate(
        action.toIndex(), action.type().name(), tiles, probability, chosen, action.isTsumogiri());
  }

  private static MeldSnapshot meld(Meld meld, int seat) {
    int offset = meld.relativeSource().playerOffset();
    int from = offset < 0 ? -1 : (seat + offset) % 4;
    int baseCount = meld.type() == Meld.Type.KAKAN ? 3 : meld.size();
    int calledIndex = offset < 0 ? -1 : offset == 3 ? 0 : offset == 2 ? 1 : baseCount - 1;
    List<String> tiles = new ArrayList<>();
    boolean calledRemoved = false;
    boolean redPlaced = false;
    for (int i = 0; i < baseCount; i++) {
      int t = meld.tileAt(i);
      if (calledIndex >= 0 && t == meld.calledTileType() && !calledRemoved) {
        calledRemoved = true;
        continue;
      }
      boolean red =
          !redPlaced
              && t == meld.akaTileType()
              && meld.akaSource() == Meld.AkaSource.CONSUMED_HAND_TILE;
      tiles.add(tile(t, red));
      redPlaced |= red;
    }
    if (calledIndex >= 0)
      tiles.add(
          calledIndex, tile(meld.calledTileType(), meld.akaSource() == Meld.AkaSource.CALLED_TILE));
    int addedIndex = -1;
    if (meld.type() == Meld.Type.KAKAN) {
      addedIndex = tiles.size();
      tiles.add(tile(meld.baseTileType(), meld.addedTileIsAka()));
    }
    return new MeldSnapshot(meld.type().name(), tiles, from, calledIndex, addedIndex);
  }

  static String tile(int type, boolean red) {
    if (type >= 27) return HONORS[type - 27];
    return (type % 9 + 1) + String.valueOf("mps".charAt(type / 9)) + (red ? "r" : "");
  }
}
