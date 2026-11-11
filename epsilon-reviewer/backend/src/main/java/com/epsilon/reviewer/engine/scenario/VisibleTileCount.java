package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 判断する席に見える牌（自分の手牌とツモ牌、全ての河、全ての副露、ドラ表示牌）を牌の記号ごとに数え、変更で置いた牌がどれも、その
 * 牌の写しの数（ルールの牌の集合から求める）を超えないことを確かめる。鳴かれた河の牌は副露の牌として数える。超えたら、その牌を
 * 置いた最初の変更を名指しして断る。
 */
final class VisibleTileCount {
  private final Variant variant;
  private final Map<String, Integer> counts = new HashMap<>();

  private VisibleTileCount(Variant variant) {
    this.variant = variant;
  }

  /**
   * @param table 変更を反映した判断の局面
   * @param placements 牌を替えた変更（適用した順）
   */
  static void require(Variant variant, TableState table, int seat, List<Placement> placements) {
    var visible = new VisibleTileCount(variant);
    PlayerState own = table.players().get(seat);
    own.hand().forEach(visible::count);
    if (own.draw() != null) visible.count(own.draw());
    for (PlayerState player : table.players()) {
      for (RiverTile tile : player.river()) if (!tile.called()) visible.count(tile.tile());
      for (Meld meld : player.melds()) meld.tiles().forEach(visible::count);
    }
    table.dora().forEach(visible::count);
    placements.forEach(visible::requireWithin);
  }

  private void count(String tile) {
    counts.merge(tile, 1, Integer::sum);
  }

  private void requireWithin(Placement placement) {
    String tile = placement.tile();
    int copies = TileCopies.count(variant, tile);
    if (counts.getOrDefault(tile, 0) > copies)
      throw new ReviewerException(
          ErrorCode.INVALID_EDIT,
          placement.name()
              + ": more copies of "
              + tile
              + " would be visible than the "
              + copies
              + " in the set.");
  }

  /**
   * 牌を替えた変更。
   *
   * @param name 変更の名前（edits.rivers[0] など）
   * @param tile 置いた牌
   */
  record Placement(String name, String tile) {}
}
