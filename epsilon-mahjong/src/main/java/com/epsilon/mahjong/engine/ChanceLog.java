package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 局の中で、運で決まったイベントの記録。追記するだけで、読めるのは {@link FullView} だけ。牌は物理 id で持つ。
 * 配牌と局の始めの表示牌はイベントにしない。それらを除いた、誰にも見えない牌の枚数を局の始めの枚数として持つ。
 * イベントを 1 つ足すたびに、その牌の種類で、いま誰にも見えない枚数が 1 減る。誰かに見えた物理牌（配牌・表示牌・イベントの牌）も
 * 覚えるので、誰にも見えない牌は、牌山のまだ読んでいない位置にある牌と一致する。
 *
 * <p>ある判断の時点から次の判断の時点までのイベントは、表示牌（めくった順）が先で、ツモが後に並ぶ。どう進んでも、表示牌は
 * 続くツモより先にめくるので、この順になる。容量は、ツモ（嶺上を含めて山の枚数以下）、槓ドラ（4 枚以下）、裏ドラ表示牌（表示牌の数以下）の和。
 */
public final class ChanceLog {

  /** 表示牌と裏ドラ表示牌のように、席の無いイベントに付ける席の値。 */
  public static final int NO_SEAT = -1;

  private static final ChanceKind[] KINDS = ChanceKind.values();

  private final Variant variant;
  private final TileSet set;
  private final byte[] kind;
  private final byte[] seat;

  /** 物理 id（符号なしで読む）。 */
  private final byte[] tile;

  private final byte[] start = new byte[Tiles.KINDS];
  private final byte[] unseen = new byte[Tiles.KINDS];

  /** 誰かに見えた物理牌のビット。物理 id のビットは、{@code id / 64} 番目のワードの {@code id mod 64} ビット目にある。 */
  private final long[] seen = new long[(Tiles.PHYSICAL_IDS + Long.SIZE - 1) / Long.SIZE];

  private int size;

  ChanceLog(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    int capacity = variant.liveWall() + Variant.MAX_KANS + WinContext.MAX_INDICATORS;
    kind = new byte[capacity];
    seat = new byte[capacity];
    tile = new byte[capacity];
  }

  public int size() {
    return size;
  }

  public ChanceKind kind(int index) {
    return KINDS[kind[at(index)]];
  }

  /** ツモの席。表示牌と裏ドラ表示牌は {@link #NO_SEAT}。 */
  public int seat(int index) {
    return seat[at(index)];
  }

  /** 物理 id。 */
  public int tile(int index) {
    return Byte.toUnsignedInt(tile[at(index)]);
  }

  /** 配牌と局の始めの表示牌を除いて、誰にも見えなかったその種類の枚数。 */
  public int start(int tileKind) {
    return start[tileKind];
  }

  /** その種類で、いま誰にも見えない枚数。山と王牌で、まだ引かれても、めくられてもいない牌を数える。 */
  public int unseen(int tileKind) {
    return unseen[tileKind];
  }

  /**
   * その物理牌が誰にも見えないなら true。立直の和了で局末にめくった裏ドラ表示牌は、見えた牌に数える。
   *
   * @param physicalId 物理 id
   * @throws IllegalArgumentException ルールに無い物理 id のとき
   */
  public boolean hidden(int physicalId) {
    requireTile(physicalId);
    return (seen[physicalId >>> 6] & 1L << physicalId) == 0;
  }

  /** 配った手牌と局の始めの表示牌から、誰にも見えない枚数（局の始めといま）を数え、イベントを空にする。 */
  void begin(Hand[] hands, int indicator) {
    int indicatorKind = Tiles.kind(requireTile(indicator));
    Arrays.fill(seen, 0);
    see(indicator);
    for (int index = 0; index < set.size(); index++) {
      int physicalId = set.physicalId(index);
      for (Hand hand : hands) {
        if (hand.holds(physicalId)) {
          see(physicalId);
        }
      }
    }
    for (int tileKind = 0; tileKind < Tiles.KINDS; tileKind++) {
      int hidden = set.copies(tileKind) - (tileKind == indicatorKind ? 1 : 0);
      for (Hand hand : hands) {
        hidden -= hand.count(tileKind);
      }
      if (hidden < 0) {
        throw new IllegalArgumentException(
            "The deal and the indicator hold more than " + set.copies(tileKind) + " tiles of kind " + tileKind);
      }
      unseen[tileKind] = (byte) hidden;
    }
    System.arraycopy(unseen, 0, start, 0, Tiles.KINDS);
    size = 0;
  }

  void appendDraw(int drawer, int physicalId) {
    append(ChanceKind.DRAW, Objects.checkIndex(drawer, variant.seats()), physicalId);
  }

  void appendRinshan(int drawer, int physicalId) {
    append(ChanceKind.RINSHAN, Objects.checkIndex(drawer, variant.seats()), physicalId);
  }

  void appendIndicator(int physicalId) {
    append(ChanceKind.INDICATOR, NO_SEAT, physicalId);
  }

  void appendUra(int physicalId) {
    append(ChanceKind.URA, NO_SEAT, physicalId);
  }

  void copyFrom(ChanceLog other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.kind, 0, kind, 0, kind.length);
    System.arraycopy(other.seat, 0, seat, 0, seat.length);
    System.arraycopy(other.tile, 0, tile, 0, tile.length);
    System.arraycopy(other.start, 0, start, 0, start.length);
    System.arraycopy(other.unseen, 0, unseen, 0, unseen.length);
    System.arraycopy(other.seen, 0, seen, 0, seen.length);
    size = other.size;
  }

  /**
   * イベントを 1 つ足す。容量を超えるときと、その種類で誰にも見えない牌がもう無いとき（同じ牌を 2 度数えた）は
   * IllegalStateException を投げる。
   */
  private void append(ChanceKind chanceKind, int owner, int physicalId) {
    int tileKind = Tiles.kind(requireTile(physicalId));
    if (size == kind.length) {
      throw new IllegalStateException("A round has at most " + kind.length + " chance events");
    }
    if (unseen[tileKind] == 0) {
      throw new IllegalStateException("No unseen tile of kind " + tileKind + " is left for " + physicalId);
    }
    kind[size] = (byte) chanceKind.ordinal();
    seat[size] = (byte) owner;
    tile[size] = (byte) physicalId;
    size++;
    unseen[tileKind]--;
    see(physicalId);
  }

  private void see(int physicalId) {
    seen[physicalId >>> 6] |= 1L << physicalId;
  }

  private int requireTile(int physicalId) {
    if (!set.contains(physicalId)) {
      throw new IllegalArgumentException(variant + " has no physical tile " + physicalId);
    }
    return physicalId;
  }

  private int at(int index) {
    return Objects.checkIndex(index, size);
  }
}
