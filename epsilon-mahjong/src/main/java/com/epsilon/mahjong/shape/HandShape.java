package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 純手牌（副露・暗槓・抜いた北を除く手の牌）を種類ごとの枚数で持ち、面子（副露と暗槓）の数も持つ。形の表を引くコードを
 * ブロックごとに持ち、牌を出し入れするたびに差分で更新する。コードは枚数を 5 進数の桁に並べた値で、
 * Σ 枚数 × 5^(種類 − ブロックの先頭の種類)。
 *
 * <p>どの種類も 4 枚まで、手の枚数（純手牌 ＋ 3 × 面子）は 14 枚まで。ルールで使わない種類の牌は入れない。同時に使えるのは
 * 1 スレッドだけ。
 */
public final class HandShape {

  /** 手の枚数の上限。和了の形の枚数と同じ。 */
  public static final int MAX_SIZE = Variant.HAND + 1;

  /** 面子 1 つの牌の数。手の枚数では槓子も 3 枚と数える。 */
  public static final int MELD_SIZE = 3;

  /** 七対子の対子の数。対子はどれも違う種類で、同じ種類の 4 枚を 2 つの対子とは数えない。 */
  public static final int SEVEN_PAIRS = 7;

  /**
   * 形の表を引くときの単位になるブロック。数牌は色ごとに 1 つで、隣り合う種類で順子を作れる。字牌の 7 種類はどれも順子に
   * ならないので、1 つにまとめる。
   */
  enum Block {
    MAN(Tiles.M1, Tiles.M9),
    PIN(Tiles.P1, Tiles.P9),
    SOU(Tiles.S1, Tiles.S9),
    HONOR(Tiles.TON, Tiles.CHUN);

    /** コードの基数（1 種類の枚数は 0〜4）。 */
    static final int RADIX = Tiles.COPIES + 1;

    private static final Block[] OF_KIND = new Block[Tiles.KINDS];

    /** 種類ごとに、ブロックのコードの中でその種類の桁が持つ重み。 */
    private static final int[] WEIGHT = new int[Tiles.KINDS];

    static {
      for (Block block : values()) {
        int weight = 1;
        for (int kind = block.first; kind <= block.last; kind++) {
          OF_KIND[kind] = block;
          WEIGHT[kind] = weight;
          weight *= RADIX;
        }
      }
    }

    private final int first;
    private final int last;
    private final boolean runs;

    Block(int first, int last) {
      this.first = first;
      this.last = last;
      runs = Tiles.Group.of(first).suited();
    }

    int first() {
      return first;
    }

    int last() {
      return last;
    }

    int size() {
      return last - first + 1;
    }

    /** 隣り合う種類で順子を作れるか（数牌の色なら true）。 */
    boolean runs() {
      return runs;
    }

    /** コードが取りうる値の数。5 をブロックの種類の数だけ掛けた値。 */
    int codes() {
      return WEIGHT[last] * RADIX;
    }

    static Block of(int kind) {
      return OF_KIND[kind];
    }

    static int weight(int kind) {
      return WEIGHT[kind];
    }
  }

  private final byte[] counts = new byte[Tiles.KINDS];
  private final int[] codes = new int[Block.values().length];
  private int tiles;
  private int melds;

  /** 1 枚以上ある種類、2 枚以上ある種類、4 枚ある種類のマスク。 */
  private long kinds;
  private long pairs;
  private long full;

  /** 指定した種類の牌を 1 枚足す。その種類が既に 4 枚あるか、手の枚数が 14 枚なら IllegalStateException を投げる。 */
  public void add(int kind) {
    int count = counts[kind];
    if (count == Tiles.COPIES || size() == MAX_SIZE) {
      throw new IllegalStateException("Cannot add " + name(kind) + " to " + this);
    }
    counts[kind] = (byte) (count + 1);
    codes[Block.of(kind).ordinal()] += Block.weight(kind);
    tiles++;
    long bit = Tiles.bit(kind);
    kinds |= bit;
    if (count == 1) {
      pairs |= bit;
    }
    if (count == Tiles.COPIES - 1) {
      full |= bit;
    }
  }

  /** 指定した種類の牌を 1 枚除く。手に無ければ IllegalStateException を投げる。 */
  public void remove(int kind) {
    int count = counts[kind];
    if (count == 0) {
      throw new IllegalStateException("No " + name(kind) + " in " + this);
    }
    counts[kind] = (byte) (count - 1);
    codes[Block.of(kind).ordinal()] -= Block.weight(kind);
    tiles--;
    long bit = Tiles.bit(kind);
    if (count == 1) {
      kinds &= ~bit;
    }
    if (count == 2) {
      pairs &= ~bit;
    }
    if (count == Tiles.COPIES) {
      full &= ~bit;
    }
  }

  /**
   * 面子を 1 つ足す。面子にした牌は、先に {@link #remove} で除いておく。面子が既に 4 つあるか、手の枚数が 14 枚を超えるなら
   * IllegalStateException を投げる。
   */
  public void addMeld() {
    if (melds == Variant.MAX_MELDS || size() + MELD_SIZE > MAX_SIZE) {
      throw new IllegalStateException("Cannot add a meld to " + this);
    }
    melds++;
  }

  public void clear() {
    Arrays.fill(counts, (byte) 0);
    Arrays.fill(codes, 0);
    tiles = 0;
    melds = 0;
    kinds = 0;
    pairs = 0;
    full = 0;
  }

  public void copyFrom(HandShape other) {
    System.arraycopy(other.counts, 0, counts, 0, counts.length);
    System.arraycopy(other.codes, 0, codes, 0, codes.length);
    tiles = other.tiles;
    melds = other.melds;
    kinds = other.kinds;
    pairs = other.pairs;
    full = other.full;
  }

  public int count(int kind) {
    return counts[kind];
  }

  /** 純手牌の枚数。 */
  public int tiles() {
    return tiles;
  }

  /** 面子（副露と暗槓）の数。 */
  public int melds() {
    return melds;
  }

  /** 手の枚数（純手牌 ＋ 3 × 面子）。打牌の前は 14、後は 13。 */
  public int size() {
    return tiles + MELD_SIZE * melds;
  }

  /** 1 枚以上ある種類の集合。 */
  public long kinds() {
    return kinds;
  }

  /** 2 枚以上ある種類の集合。 */
  public long pairs() {
    return pairs;
  }

  /** 4 枚ある種類の集合（もう引けない）。 */
  long full() {
    return full;
  }

  int code(Block block) {
    return codes[block.ordinal()];
  }

  private static String name(int kind) {
    return Face.of(kind, false).mjai();
  }

  @Override
  public String toString() {
    StringBuilder text = new StringBuilder("HandShape[");
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < counts[kind]; copy++) {
        text.append(name(kind));
      }
    }
    return text.append(" melds=").append(melds).append(']').toString();
  }
}
