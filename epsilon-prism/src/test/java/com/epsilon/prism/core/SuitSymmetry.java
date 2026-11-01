package com.epsilon.prism.core;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import java.util.Arrays;
import java.util.List;

/**
 * 萬子・筒子・索子の入れ替え（6 通り）。麻雀の規則は色の入れ替えで変わらない（緑一色だけが索子に固有で、ここでは無視する）。字牌と数字は動かさず、
 * 赤の 5 は赤の 5 へ写す（物理牌 id = 牌種 × 4 + 写し番号で、赤は各色の 5 の写し番号 0）。
 *
 * <p>番号 0 は恒等。1・2 は巡回（萬→筒→索→萬 と その逆）、3〜5 は 2 色の入れ替え。番号の小さい順に k 通りを使うと、3 通りで巡回群になる。
 */
public final class SuitSymmetry {
  /** 入れ替えの数（3 色の順列）。 */
  public static final int VIEWS = 6;

  private static final int[][] SUITS = {
    {0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}, {2, 1, 0}, {0, 2, 1}
  };
  private static final int[][] TYPE_MAPS = new int[VIEWS][];
  private static final int[][] PHYSICAL_MAPS = new int[VIEWS][];

  static {
    for (int view = 0; view < VIEWS; view++) {
      int[] types = new int[Tile.NUM_TILE_TYPES];
      for (int type = 0; type < types.length; type++)
        types[type] = type < 27 ? SUITS[view][type / 9] * 9 + type % 9 : type;
      int[] physical = new int[types.length * Tile.TILES_PER_TYPE];
      for (int id = 0; id < physical.length; id++)
        physical[id] = types[Tile.typeOf(id)] * Tile.TILES_PER_TYPE + id % Tile.TILES_PER_TYPE;
      TYPE_MAPS[view] = types;
      PHYSICAL_MAPS[view] = physical;
    }
  }

  private SuitSymmetry() {}

  /** 入れ替え view の物理牌の写し（136 要素。{@link GameState#mapWallTiles} に渡す）。 */
  public static int[] physicalMap(int view) {
    return PHYSICAL_MAPS[view].clone();
  }

  /** 牌種 type（-1 は牌のない行動でそのまま）を入れ替え view で写した牌種。 */
  public static int mapType(int view, int type) {
    return type < 0 ? type : TYPE_MAPS[view][type];
  }

  /** shadow が action を入れ替え view で写した行動か（種別・牌種・赤と自摸切りの選び・チーの 3 牌種）。 */
  public static boolean mapped(int view, Action action, Action shadow) {
    if (action.type() != shadow.type()
        || action.tileSelection() != shadow.tileSelection()
        || mapType(view, action.tileType()) != shadow.tileType()) return false;
    if (action.type() != Action.Type.CHI) return true;
    int[] chi = action.chiTileTypes();
    for (int k = 0; k < chi.length; k++) chi[k] = mapType(view, chi[k]);
    return Arrays.equals(chi, shadow.chiTileTypes());
  }

  /** 写した合法手の一覧 shadowLegal の中で action を写した手の番号（なければ -1）。 */
  public static int find(int view, List<Action> shadowLegal, Action action) {
    for (int index = 0; index < shadowLegal.size(); index++)
      if (mapped(view, action, shadowLegal.get(index))) return index;
    return -1;
  }
}
