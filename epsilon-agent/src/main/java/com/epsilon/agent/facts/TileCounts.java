package com.epsilon.agent.facts;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.codec.AkaMask;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 判断の時点での種類ごとの枚数を、記録 O だけから数える。持つのは、自分の純手牌、見えている枚数、自分から見えない枚数、ドラの
 * 重み。判断ごとに {@link #load} で作り直し、そのときメモリは確保しない。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>見えている枚数は {@code SeatView.visible} と同じ数え方で、河・全ての席の面子・抜いた北・めくった表示牌を足したもの。鳴かれた
 * 打牌は面子の側で 1 回だけ数える。見えない枚数は、その種類の牌の数（ルールで決まる）から、見えている枚数と自分の純手牌を引いた
 * もの。打牌も鳴きも牌を手から公開の場所へ移すだけなので、どの手を選んでも見えない枚数は判断の時点と変わらない。そのため、手を
 * 選んだ後の受け入れもこの枚数で数える。
 */
public final class TileCounts {

  private final Variant variant;
  private final TileSet set;
  private final Observation.EventRecord record = new Observation.EventRecord();
  private final int[] hand = new int[Tiles.KINDS];
  private final int[] visible = new int[Tiles.KINDS];
  private final int[] unseen = new int[Tiles.KINDS];
  private final int[] dora = new int[Tiles.KINDS];
  private final int[] indicators = new int[Tiles.KINDS];
  private int totalUnseen;
  private long handRed;
  private long seenRed;
  private long unseenRed;
  private long doraKinds;

  public TileCounts(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
  }

  /**
   * 記録 O から、判断の時点での枚数を数える。記録のルール（四人打ちか三人打ちか）が作ったときと違えば IllegalArgumentException を投げる。
   * 見えている枚数と手の枚数を足すと、その種類の牌の数を超える種類があれば IllegalStateException を投げる。
   */
  public void load(Observation o) {
    begin(o);
    for (int event = 0; event < o.events(); event++) {
      add(o.readEvent(event, record));
    }
    finish(o);
  }

  /**
   * ほかの集計と一緒にイベントを 1 回だけ読むときに、最初に呼ぶ。続けて記録 O の全てのイベントを前から {@link #add} に渡し、最後に
   * {@link #finish} を呼ぶと、{@link #load} と同じ値になる。
   *
   * @throws IllegalArgumentException 記録のルール（四人打ちか三人打ちか）が、作ったときと違うとき
   */
  public void begin(Observation o) {
    if (o.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    Arrays.fill(visible, 0);
    Arrays.fill(dora, 0);
    Arrays.fill(indicators, 0);
    handRed = AkaMask.redKinds(o.handAkaMask());
    seenRed = handRed | AkaMask.redKinds(o.indicatorAkaMask());
  }

  /** {@link #begin} の後に、イベントを前から順に 1 つずつ数える。 */
  public void add(Observation.EventRecord event) {
    if (event.kind() == EventKind.DISCARD) {
      if (event.discardCalledBy() < 0) {
        visible[event.discardTile()]++;
      }
      if (event.discardAka()) {
        seenRed |= Tiles.bit(event.discardTile());
      }
      return;
    }
    if (event.kind() == EventKind.NUKI) {
      visible[Tiles.PEI]++;
      return;
    }
    Meld meld = event.meld();
    if (meld.type() == Meld.Type.KAKAN) {
      visible[meld.base()]++;
    } else {
      for (int index = 0; index < meld.size(); index++) {
        visible[meld.tile(index)]++;
      }
    }
    if (meld.red()) {
      seenRed |= meld.kinds() & Tiles.FIVES;
    }
  }

  /**
   * 全てのイベントを {@link #add} に渡した後に呼び、表示牌と手を数えて読み終える。
   *
   * @throws IllegalStateException 見えている枚数と手の枚数を足すと、その種類の牌の数を超える種類があるとき
   */
  public void finish(Observation o) {
    doraKinds = 0;
    for (int index = 0; index < o.indicators(); index++) {
      int kind = o.indicatorTile(index);
      visible[kind]++;
      indicators[kind]++;
      int doraKind = set.doraOf(kind);
      dora[doraKind]++;
      doraKinds |= Tiles.bit(doraKind);
    }
    o.handCounts(hand);
    totalUnseen = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int left = set.copies(kind) - visible[kind] - hand[kind];
      if (left < 0) {
        throw new IllegalStateException(
            "More than " + set.copies(kind) + " tiles of kind " + kind + " are known: visible " + visible[kind]
                + ", hand " + hand[kind]);
      }
      unseen[kind] = left;
      totalUnseen += left;
    }
    unseenRed = set.redKinds() & ~seenRed;
  }

  public Variant variant() {
    return variant;
  }

  /** 自分の純手牌の枚数（ツモ牌を含む）。 */
  public int hand(int kind) {
    return hand[kind];
  }

  /** 見えている枚数（自分の純手牌を含まない）。 */
  public int visible(int kind) {
    return visible[kind];
  }

  /** 自分から見えない枚数。 */
  public int unseen(int kind) {
    return unseen[kind];
  }

  /** 見えない枚数を、全ての種類で足したもの。 */
  public int totalUnseen() {
    return totalUnseen;
  }

  /** 指定した種類をドラにする表示牌の数（ドラの重み）。 */
  public int dora(int kind) {
    return dora[kind];
  }

  /** ドラの種類（ドラの重みが 1 以上の種類）。 */
  public long doraKinds() {
    return doraKinds;
  }

  /** 指定した種類が、ドラ表示牌として出ている枚数。 */
  public int indicators(int kind) {
    return indicators[kind];
  }

  /** 自分の純手牌にある赤の 5 の種類。 */
  public long handRed() {
    return handRed;
  }

  /** 赤の 5 がまだ自分から見えていない 5 の種類（その赤が手・河・面子・表示牌のどこにも出ていない）。 */
  public long unseenRed() {
    return unseenRed;
  }

  /** 指定した種類（ビットの集合）の全てについて、見えない枚数を足したもの（生きている枚数）。 */
  public int live(long kinds) {
    int live = 0;
    for (long rest = kinds; rest != 0; rest &= rest - 1) {
      live += unseen[Long.numberOfTrailingZeros(rest)];
    }
    return live;
  }
}
