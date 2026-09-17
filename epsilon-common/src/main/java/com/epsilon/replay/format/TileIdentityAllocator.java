package com.epsilon.replay.format;

import com.epsilon.core.Tile;
import java.util.Arrays;

/** 牌種表記から一局内で一意な物理 ID を割り当て、手牌の所在を追跡する。 */
public final class TileIdentityAllocator {
  private final boolean[] allocated = new boolean[136];
  private final boolean[][] hands = new boolean[4][136];
  private final int[] drawn = {-1, -1, -1, -1};
  private int lastDiscardSeat = -1;
  private int lastDiscard = -1;

  public void startRound(int indicatorType, boolean red) {
    Arrays.fill(allocated, false);
    for (boolean[] hand : hands) Arrays.fill(hand, false);
    Arrays.fill(drawn, -1);
    lastDiscardSeat = -1;
    lastDiscard = -1;
    reserveIndicator(indicatorType, red);
  }

  public int reserveIndicator(int type, boolean red) {
    return allocate(type, red);
  }

  public int allocateInitialTile(int seat, int type, boolean red) {
    int id = allocate(type, red);
    hands[seat][id] = true;
    return id;
  }

  public int allocateDrawTile(int seat, int type, boolean red) {
    int id = allocateInitialTile(seat, type, red);
    drawn[seat] = id;
    lastDiscard = -1;
    lastDiscardSeat = -1;
    return id;
  }

  public int drawnTileId(int seat) {
    return drawn[seat];
  }

  /** 呼び出し元へ渡す新規配列。ツモ牌を含む、指定した牌種の手牌。 */
  public int[] handTilesOfType(int seat, int type) {
    int count = 0;
    for (int id = type * 4; id < type * 4 + 4; id++) if (hands[seat][id]) count++;
    int[] result = new int[count];
    int index = 0;
    for (int id = type * 4; id < type * 4 + 4; id++) if (hands[seat][id]) result[index++] = id;
    return result;
  }

  public int removeDiscardTile(int seat, int type, boolean red, boolean tsumogiri) {
    int id;
    if (tsumogiri) {
      id = drawn[seat];
      if (id < 0 || !matches(id, type, red) || !hands[seat][id])
        throw new IllegalArgumentException("Discarded draw does not match the preceding draw.");
      hands[seat][id] = false;
    } else id = remove(seat, type, red, false);
    drawn[seat] = -1;
    lastDiscardSeat = seat;
    lastDiscard = id;
    return id;
  }

  public int takeCalledTile(int target, int type, boolean red) {
    if (target != lastDiscardSeat || lastDiscard < 0 || !matches(lastDiscard, type, red))
      throw new IllegalArgumentException("Called tile does not match the preceding discard.");
    int id = lastDiscard;
    lastDiscardSeat = -1;
    lastDiscard = -1;
    return id;
  }

  /** 新規復号配列の所有権を受け取り、代表 ID を実際の手牌 ID へその場で置換する。 */
  public int[] consumeMeldTiles(int seat, int[] representativeIds) {
    for (int i = 0; i < representativeIds.length; i++) {
      int representative = representativeIds[i];
      representativeIds[i] =
          remove(seat, Tile.typeOf(representative), Tile.isAka(representative), true);
    }
    drawn[seat] = -1;
    return representativeIds;
  }

  private int allocate(int type, boolean red) {
    for (int id = type * 4; id < type * 4 + 4; id++) {
      if (!allocated[id] && Tile.isAka(id) == red) {
        allocated[id] = true;
        return id;
      }
    }
    throw new IllegalArgumentException(
        "Tile count exceeds four or contains duplicate red tiles: " + Tile.name(type));
  }

  private int remove(int seat, int type, boolean red, boolean allowDraw) {
    for (int id = type * 4; id < type * 4 + 4; id++) {
      if (hands[seat][id] && matches(id, type, red) && id != drawn[seat]) {
        hands[seat][id] = false;
        return id;
      }
    }
    // 暗槓・加槓では直前に引いた牌も消費できる。
    int id = drawn[seat];
    if (allowDraw && id >= 0 && hands[seat][id] && matches(id, type, red)) {
      hands[seat][id] = false;
      return id;
    }
    throw new IllegalArgumentException(
        "Consumed tile is not in the hand: seat=" + seat + " tile=" + Tile.name(type));
  }

  private static boolean matches(int id, int type, boolean red) {
    return Tile.typeOf(id) == type && Tile.isAka(id) == red;
  }
}
