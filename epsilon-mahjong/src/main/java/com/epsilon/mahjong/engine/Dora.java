package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * めくったドラ表示牌。表示牌はめくった順に物理 id で持ち、めくった時点の公開イベントの数も一緒に持つ。牌の種類ごとの重みは、
 * その種類をドラにする表示牌の数。保留は、明槓・加槓の槓ドラを後めくりにして、まだめくっていない状態のこと。
 *
 * <p>表示牌はめくった順に並ぶので、保留の表示牌は次にめくる表示牌になり、めくれば保留は消える。2 つ目の明槓・加槓の嶺上牌を
 * 引く前に前の保留をめくるので、保留は多くても 1 つ。
 */
final class Dora {

  private final TileSet set;
  private final int[] indicators = new int[WinContext.MAX_INDICATORS];
  private final int[] revealedAt = new int[WinContext.MAX_INDICATORS];
  private final byte[] weights = new byte[Tiles.KINDS];
  private int count;
  private boolean pending;

  Dora(Variant variant) {
    set = variant.tiles();
  }

  int count() {
    return count;
  }

  /** めくった順で何番目かを受け、その表示牌の物理 id を返す。 */
  int indicator(int index) {
    return indicators[Objects.checkIndex(index, count)];
  }

  /** めくった順で何番目かを受け、その表示牌をめくった時点の公開イベントの数を返す。 */
  int revealedAt(int index) {
    return revealedAt[Objects.checkIndex(index, count)];
  }

  /** その種類をドラにする表示牌の数。 */
  int weight(int kind) {
    return weights[kind];
  }

  boolean pending() {
    return pending;
  }

  void clear() {
    count = 0;
    pending = false;
    Arrays.fill(weights, (byte) 0);
  }

  /**
   * 次の表示牌をめくる。
   *
   * @param physicalId 表示牌の物理 id
   * @param event めくる時点の公開イベントの数
   */
  void reveal(int physicalId, int event) {
    if (count == indicators.length) {
      throw new IllegalStateException("At most " + indicators.length + " dora indicators");
    }
    if (!set.contains(physicalId)) {
      throw new IllegalArgumentException(set + " has no physical tile " + physicalId);
    }
    indicators[count] = physicalId;
    revealedAt[count] = event;
    count++;
    weights[set.doraOf(Tiles.kind(physicalId))]++;
    pending = false;
  }

  /** 明槓・加槓の槓ドラを後めくりにして、保留にする。既に保留があれば IllegalStateException を投げる。 */
  void defer() {
    if (pending) {
      throw new IllegalStateException("A kan dora is already pending");
    }
    pending = true;
  }

  void copyFrom(Dora other) {
    if (other.set != set) {
      throw new IllegalArgumentException("Tile set mismatch: expected " + set + " but found " + other.set);
    }
    System.arraycopy(other.indicators, 0, indicators, 0, indicators.length);
    System.arraycopy(other.revealedAt, 0, revealedAt, 0, revealedAt.length);
    System.arraycopy(other.weights, 0, weights, 0, weights.length);
    count = other.count;
    pending = other.pending;
  }
}
