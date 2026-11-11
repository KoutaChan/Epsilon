package com.epsilon.reviewer.engine.view;

import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.action.Meld.Type;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;

/**
 * 副露を表示するときの牌の並び。横向きの牌は、上家から鳴いたら位置 0、対面からなら 1、下家からなら最後に置き、暗槓には無い。
 * 加槓で加えた牌は、元の 3 枚の後（位置 3）に置く。赤にする牌は、手から出した牌が赤なら鳴いた牌を除いた最初の 5、鳴いた牌が赤なら
 * その牌、加槓で足した牌が赤ならその牌。三人打ちで抜いた北は、1 枚ずつ副露の後に並べる。
 */
public final class MeldLayout {
  private MeldLayout() {}

  /** 抜いた北 1 枚の表示を作る。 */
  public static Meld nuki() {
    return new Meld("NUKI", List.of(TileCode.of(Tiles.PEI, false)), -1, -1, -1);
  }

  /**
   * 面子の表示を作る。mahjong の Meld はこのパッケージの Meld と名前が重なるので、var で受けて型名を書かない。
   *
   * @param id 面子の番号
   * @param owner 面子を持つ席
   * @param seats 席の数
   */
  public static Meld of(int id, int owner, int seats) {
    var meld = com.epsilon.mahjong.action.Meld.of(id);
    RelativeSource source = meld.source();
    int from = source == RelativeSource.NONE ? -1 : source.seat(owner, seats);
    int baseCount = meld.type() == Type.KAKAN ? 3 : meld.size();
    int calledIndex =
        switch (source) {
          case NONE -> -1;
          case KAMICHA -> 0;
          case TOIMEN -> 1;
          case SHIMOCHA -> baseCount - 1;
        };
    List<String> tiles = new ArrayList<>(4);
    boolean calledRemoved = false;
    boolean redPlaced = false;
    for (int i = 0; i < baseCount; i++) {
      int kind = meld.tile(i);
      if (calledIndex >= 0 && kind == meld.called() && !calledRemoved) {
        calledRemoved = true;
        continue;
      }
      boolean red =
          !redPlaced
              && meld.aka() == AkaSource.CONSUMED_HAND_TILE
              && Tiles.contains(Tiles.FIVES, kind);
      tiles.add(TileCode.of(kind, red));
      redPlaced |= red;
    }
    if (calledIndex >= 0)
      tiles.add(calledIndex, TileCode.of(meld.called(), meld.aka() == AkaSource.CALLED_TILE));
    int addedIndex = -1;
    if (meld.type() == Type.KAKAN) {
      addedIndex = tiles.size();
      tiles.add(TileCode.of(meld.base(), meld.aka() == AkaSource.ADDED_KAN_TILE));
    }
    return new Meld(meld.type().name(), tiles, from, calledIndex, addedIndex);
  }
}
