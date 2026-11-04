package com.epsilon.reviewer.engine.view;

import static com.epsilon.core.Meld.AkaSource.CALLED_TILE;
import static com.epsilon.core.Meld.AkaSource.CONSUMED_HAND_TILE;
import static com.epsilon.core.Meld.Type.KAKAN;

import com.epsilon.core.Hand;
import java.util.ArrayList;
import java.util.List;

/**
 * 副露の表示の並び。横向きの牌は上家から 0、対面から 1、下家から最後、暗槓は無し。加槓は 3 枚の上に加えた牌を 3 番目に置く。赤は手から出した
 * 牌なら鳴いた牌でない最初の同じ牌、鳴いた牌が赤ならその牌。
 */
public final class MeldLayout {
  private MeldLayout() {}

  /** 席の index 番目の副露。core の Meld はこの包みの Meld と名前が重なるので型名を書かずに読む。 */
  public static Meld of(Hand hand, int index, int seat) {
    var meld = hand.meld(index);
    int offset = meld.relativeSource().playerOffset();
    int from = offset < 0 ? -1 : (seat + offset) % 4;
    int baseCount = meld.type() == KAKAN ? 3 : meld.size();
    int calledIndex = offset < 0 ? -1 : offset == 3 ? 0 : offset == 2 ? 1 : baseCount - 1;
    List<String> tiles = new ArrayList<>(4);
    boolean calledRemoved = false;
    boolean redPlaced = false;
    for (int i = 0; i < baseCount; i++) {
      int type = meld.tileAt(i);
      if (calledIndex >= 0 && type == meld.calledTileType() && !calledRemoved) {
        calledRemoved = true;
        continue;
      }
      boolean red =
          !redPlaced && type == meld.akaTileType() && meld.akaSource() == CONSUMED_HAND_TILE;
      tiles.add(TileCode.of(type, red));
      redPlaced |= red;
    }
    if (calledIndex >= 0)
      tiles.add(
          calledIndex, TileCode.of(meld.calledTileType(), meld.akaSource() == CALLED_TILE));
    int addedIndex = -1;
    if (meld.type() == KAKAN) {
      addedIndex = tiles.size();
      tiles.add(TileCode.of(meld.baseTileType(), meld.addedTileIsAka()));
    }
    return new Meld(meld.type().name(), tiles, from, calledIndex, addedIndex);
  }
}
