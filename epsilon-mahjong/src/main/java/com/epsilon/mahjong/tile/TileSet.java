package com.epsilon.mahjong.tile;

import java.util.Arrays;

/** 卓で使う牌の集合。使う種類はどれも 4 枚で、物理 id は {@link Tiles} の番号のまま。赤 5 は各 5 の中で番号 0 の牌（天鳳と同じ）。 */
public enum TileSet {
  FULL(Tiles.ALL_KINDS, Tiles.FIVES),
  /** 2〜8 萬（物理 id 4〜31）を使わない。赤 5 は 5p・5s だけ。 */
  SANMA(Tiles.ALL_KINDS & ~Tiles.range(Tiles.M2, Tiles.M8), Tiles.bit(Tiles.P5) | Tiles.bit(Tiles.S5));

  private final long kinds;
  private final long redKinds;
  private final int[] copies;
  private final int[] physicalIds;
  private final boolean[] red;

  /** 表示牌の種類 → ドラの種類。使わない種類は −1。 */
  private final int[] dora;

  TileSet(long kinds, long redKinds) {
    if ((redKinds & ~(kinds & Tiles.FIVES)) != 0) {
      throw new IllegalArgumentException("Red kinds must be fives of the set");
    }
    this.kinds = kinds;
    this.redKinds = redKinds;
    copies = new int[Tiles.KINDS];
    physicalIds = new int[Long.bitCount(kinds) * Tiles.COPIES];
    red = new boolean[Tiles.PHYSICAL_IDS];
    int index = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (!Tiles.contains(kinds, kind)) {
        continue;
      }
      copies[kind] = Tiles.COPIES;
      for (int copy = 0; copy < Tiles.COPIES; copy++) {
        physicalIds[index++] = Tiles.physicalId(kind, copy);
      }
      red[Tiles.physicalId(kind, Tiles.RED_COPY)] = Tiles.contains(redKinds, kind);
    }
    dora = doraTable(kinds);
  }

  /** 使う種類の 34 ビットのマスク。 */
  public long kinds() {
    return kinds;
  }

  public int size() {
    return physicalIds.length;
  }

  /** その種類の枚数。使う種類は 4、ほかは 0。 */
  public int copies(int kind) {
    return copies[kind];
  }

  public boolean contains(int physicalId) {
    return physicalId >= 0 && physicalId < Tiles.PHYSICAL_IDS && copies[Tiles.kind(physicalId)] > 0;
  }

  /** シャッフルする前の並び（物理 id の昇順）で、指定した位置にある牌の物理 id。 */
  public int physicalId(int index) {
    return physicalIds[index];
  }

  public boolean red(int physicalId) {
    return red[physicalId];
  }

  /** 赤 5 がある種類のマスク。 */
  public long redKinds() {
    return redKinds;
  }

  /**
   * 表示牌が指すドラの種類。同じグループ（各色・風牌・三元牌）で使う種類を順に見て、表示牌の次の種類を返す。グループの最後の
   * 次は先頭に戻る。使わない種類の表示牌なら IllegalArgumentException を投げる。
   */
  public int doraOf(int indicatorKind) {
    int next = dora[indicatorKind];
    if (next < 0) {
      throw new IllegalArgumentException(this + " does not use indicator kind " + indicatorKind);
    }
    return next;
  }

  private static int[] doraTable(long kinds) {
    int[] table = new int[Tiles.KINDS];
    Arrays.fill(table, -1);
    for (Tiles.Group group : Tiles.Group.values()) {
      int first = -1;
      int previous = -1;
      for (int kind = group.first(); kind <= group.last(); kind++) {
        if (!Tiles.contains(kinds, kind)) {
          continue;
        }
        if (previous < 0) {
          first = kind;
        } else {
          table[previous] = kind;
        }
        previous = kind;
      }
      if (previous >= 0) {
        table[previous] = first;
      }
    }
    return table;
  }
}
