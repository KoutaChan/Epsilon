package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.Tile;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;

/**
 * 判断の席に見える牌（自分の手とツモ牌、全ての河、全ての副露、ドラ表示）が、変更の後も牌の数を超えないことを確かめる。鳴かれた河の牌は副露の
 * 牌として数える。超えたら、その牌の最初の変更（河・ドラ・手牌の順）を名指す。
 */
final class VisibleTileCount {
  /** 赤の 5 の色（萬・筒・索）の数。 */
  private static final int SUITS = 3;

  private final int[] copies = new int[Tile.NUM_TILE_TYPES];
  private final int[] reds = new int[SUITS];

  private VisibleTileCount() {}

  /** @param table 変更を写した判断の局面 */
  static void require(TableState table, int seat, ScenarioEdits edits) {
    var visible = new VisibleTileCount();
    PlayerState own = table.players().get(seat);
    own.hand().forEach(visible::count);
    if (own.draw() != null) visible.count(own.draw());
    for (PlayerState player : table.players()) {
      for (RiverTile tile : player.river()) if (!tile.called()) visible.count(tile.tile());
      for (Meld meld : player.melds()) meld.tiles().forEach(visible::count);
    }
    table.dora().forEach(visible::count);
    for (int i = 0; i < edits.rivers().size(); i++)
      visible.requireWithin(edits.rivers().get(i).tile(), "edits.rivers[" + i + "]");
    for (int i = 0; i < edits.dora().size(); i++)
      visible.requireWithin(edits.dora().get(i).tile(), "edits.dora[" + i + "]");
    for (int i = 0; i < edits.hand().size(); i++)
      visible.requireWithin(edits.hand().get(i).tile(), "edits.hand[" + i + "]");
  }

  private void count(String tile) {
    int type = TileCode.type(tile);
    copies[type]++;
    if (TileCode.red(tile)) reds[Tile.suitOf(type)]++;
  }

  private void requireWithin(String tile, String name) {
    int type = TileCode.type(tile);
    String kind = TileCode.of(type, false);
    if (copies[type] > Tile.TILES_PER_TYPE)
      throw new ReviewerException(
          ErrorCode.INVALID_EDIT,
          name
              + ": more than "
              + Tile.TILES_PER_TYPE
              + " copies of "
              + kind
              + " would be visible.");
    if (TileCode.red(tile) && reds[Tile.suitOf(type)] > 1)
      throw new ReviewerException(
          ErrorCode.INVALID_EDIT, name + ": more than one red " + kind + " would be visible.");
  }
}
