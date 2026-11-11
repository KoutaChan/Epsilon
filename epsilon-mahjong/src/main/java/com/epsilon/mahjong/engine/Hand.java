package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 1 席の手。純手牌の種類ごとの枚数と、形の表を引くための 5 進のコードを持ち、牌を出し入れするたびに差分で更新する
 * （{@code HandShape}）。ほかに、持っている物理牌（同じ種類の中での番号）、面子、抜いた北の数、見逃しの振聴を持つ。
 *
 * <p>待ちは形だけで見た和了牌で、役・振聴・残りの枚数は見ない。手牌が 13 枚相当になったときに一度だけ求め、それ以外のときは 0。
 * 純手牌から牌を除く {@code take} は、面子にする牌か抜く北を除く途中で呼ぶので、待ちを 0 にする。待ちは、続けて面子か抜きを
 * 足す呼び出しが求める。
 */
final class Hand {

  private final Variant variant;
  private final TileSet set;
  private final ShapeAnalyzer analyzer;
  private final HandShape shape = new HandShape();

  /** 種類ごとに、持っている物理牌のビット。ビット c は、その種類の中で番号 c（0 始まり）の牌。 */
  private final byte[] copies = new byte[Tiles.KINDS];

  /** 面子の正準の番号（{@link Meld#id}）を、足した順に持つ。加槓したポンは加槓の番号に替える。 */
  private final short[] melds = new short[Variant.MAX_MELDS];

  private int nuki;
  private long waits;

  /**
   * 見逃しの印。形の上の和了牌を見送ったとき（役が無くてロンできない牌、槍槓、抜いた北も含む）と、立直中に和了牌の北を
   * 抜いたときに立つ。立直していない席が自分で打牌すると消えるが、下家から鳴いた後の打牌では消えない。立直中は局の終わりまで
   * 残る。消すのは {@link Steps} の打牌。
   */
  private boolean missedRon;

  Hand(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    analyzer = new ShapeAnalyzer(variant);
  }

  /** 純手牌と面子の数。 */
  HandShape shape() {
    return shape;
  }

  /** 純手牌にある、その種類の枚数。 */
  int count(int kind) {
    return shape.count(kind);
  }

  boolean holds(int physicalId) {
    return set.contains(physicalId) && (copies[Tiles.kind(physicalId)] & copyBit(physicalId)) != 0;
  }

  int meldCount() {
    return shape.melds();
  }

  /** 何番目の面子かを受け、その面子の正準の番号を返す。 */
  int meld(int index) {
    return melds[Objects.checkIndex(index, shape.melds())];
  }

  int nuki() {
    return nuki;
  }

  long waits() {
    return waits;
  }

  /** 見逃しの振聴（同巡内か、立直の後）なら true。自分の河による振聴は含まない。 */
  boolean missedRon() {
    return missedRon;
  }

  /** 純手牌に赤 5 がある種類のマスク。 */
  long redFives() {
    long reds = 0;
    for (long fives = Tiles.FIVES & set.kinds(); fives != 0; fives &= fives - 1) {
      int kind = Long.numberOfTrailingZeros(fives);
      int red = Tiles.physicalId(kind, Tiles.RED_COPY);
      if (set.red(red) && holds(red)) {
        reds |= Tiles.bit(kind);
      }
    }
    return reds;
  }

  /** 純手牌にあるその種類の物理牌のうち、赤かどうかが合う牌で、いちばん小さい物理 id を返す。無ければ IllegalStateException を投げる。 */
  int pick(int kind, boolean red) {
    return pick(kind, red, TurnEvent.NONE);
  }

  /** {@link #pick(int, boolean)} と同じだが、指定した物理牌（手番でツモった牌など）は選ばない。 */
  int pick(int kind, boolean red, int excluded) {
    for (int held = copies[kind]; held != 0; held &= held - 1) {
      int physicalId = Tiles.physicalId(kind, Integer.numberOfTrailingZeros(held));
      if (physicalId != excluded && set.red(physicalId) == red) {
        return physicalId;
      }
    }
    throw new IllegalStateException(
        "No " + (red ? "red" : "plain") + " tile of kind " + kind + " other than " + excluded + " in " + shape);
  }

  void clear() {
    shape.clear();
    Arrays.fill(copies, (byte) 0);
    Arrays.fill(melds, (short) 0);
    nuki = 0;
    waits = 0;
    missedRon = false;
  }

  /** 配牌かツモの牌を足す。既に持っている牌のときと、手牌が既に 14 枚相当のときは IllegalStateException を投げる。 */
  void add(int physicalId) {
    int kind = requireTile(physicalId);
    if ((copies[kind] & copyBit(physicalId)) != 0) {
      throw new IllegalStateException("Tile " + physicalId + " is already in " + shape);
    }
    shape.add(kind);
    copies[kind] |= (byte) copyBit(physicalId);
    settle();
  }

  void discard(int physicalId) {
    take(physicalId);
    settle();
  }

  /**
   * 面子にする牌か抜く北を純手牌から除き、待ちを 0 にする。待ちは、続けて面子か抜きを足す呼び出しが求め直す。持っていない牌なら
   * IllegalStateException を投げる。
   */
  void take(int physicalId) {
    int kind = requireTile(physicalId);
    if ((copies[kind] & copyBit(physicalId)) == 0) {
      throw new IllegalStateException("Tile " + physicalId + " is not in " + shape);
    }
    shape.remove(kind);
    copies[kind] &= (byte) ~copyBit(physicalId);
    waits = 0;
  }

  /** チー・ポン・大明槓・暗槓を足す。面子の牌は先に {@link #take} で除く。 */
  void addMeld(int meldId) {
    Meld meld = Meld.of(meldId);
    if (meld.type() == Meld.Type.KAKAN || !meld.type().in(variant)) {
      throw new IllegalArgumentException(variant + " cannot add " + meld + " as a meld");
    }
    int index = shape.melds();
    shape.addMeld();
    melds[index] = (short) meldId;
    settle();
  }

  /**
   * その種類のポンを加槓にする。足す牌は先に {@link #take} で除いておく。ポンが無ければ IllegalStateException を投げる。
   *
   * @param kind ポンの種類
   * @param addedRed 足す牌が赤 5 なら true
   * @return 加槓の正準の番号
   */
  int upgrade(int kind, boolean addedRed) {
    for (int index = 0; index < shape.melds(); index++) {
      Meld pon = Meld.of(melds[index]);
      if (pon.type() == Meld.Type.PON && pon.base() == kind) {
        int kakan = Meld.kakan(pon, addedRed).id();
        melds[index] = (short) kakan;
        settle();
        return kakan;
      }
    }
    throw new IllegalStateException("No pon of kind " + kind + " to upgrade in " + shape);
  }

  /**
   * 抜いた北の数を 1 増やす。純手牌が分かる席では、抜く北を先に手牌から除いておく。北を抜かないルールのときと、5 枚目の北のときは
   * IllegalStateException を投げる。
   */
  void addNuki() {
    if (!variant.has(RuleFlag.NUKI) || nuki == Tiles.COPIES) {
      throw new IllegalStateException(variant + " cannot take out a north after " + nuki + " nuki");
    }
    nuki++;
    settle();
  }

  void setMissedRon(boolean missed) {
    missedRon = missed;
  }

  void copyFrom(Hand other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    shape.copyFrom(other.shape);
    System.arraycopy(other.copies, 0, copies, 0, copies.length);
    System.arraycopy(other.melds, 0, melds, 0, melds.length);
    nuki = other.nuki;
    waits = other.waits;
    missedRon = other.missedRon;
  }

  private void settle() {
    waits = shape.size() == Variant.HAND ? analyzer.waits(shape) : 0;
  }

  private int requireTile(int physicalId) {
    if (!set.contains(physicalId)) {
      throw new IllegalArgumentException(variant + " has no physical tile " + physicalId);
    }
    return Tiles.kind(physicalId);
  }

  private static int copyBit(int physicalId) {
    return 1 << Tiles.copy(physicalId);
  }
}
