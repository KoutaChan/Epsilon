package com.epsilon.core;

import java.util.Objects;

/**
 * 局の公開の出来事（打牌・副露・加槓）の、追記だけの記録。出来事 e の番号は局の中の通し番号（0 始まり）。{@link River} と
 * {@link Hand} の副露の時系列は、この記録への添字だけを持つ。
 *
 * <p>不変:
 *
 * <ul>
 *   <li>書くのは {@link GameState} の 3 つの口（{@code commitDiscard}・{@code addMeld}・{@code replacePonWithKakan}）だけ。live・分岐・
 *       牌譜の再生（Reviewer）が同じ口を通る。
 *   <li>書き換えは 1 種類だけ: 鳴きの副露（チー・ポン・大明槓）を足すとき、元の打牌の {@link #calledBy} を埋める。元の打牌はいつも最後の
 *       出来事で、鳴かれていない、元の席の、鳴いた牌の種類の打牌。そうでなければ {@link IllegalStateException}。
 *   <li>加槓は元のポンの出来事を書き換えない。自分の出来事に元のポンの番号（{@link #ponEvent}）を持つ。
 *   <li>容量は {@link RoundLimits#MAX_PUBLIC_EVENTS}。超えたら {@link IllegalStateException}。黙って切らない。
 * </ul>
 *
 * <p>欄は基本型の配列（struct of arrays）で、出来事の種類ごとに意味を持つ欄が違う。打牌の欄（牌・赤・自摸切り・立直・鳴いた席・巡目・
 * 通し番号）は打牌で、副露の欄（面子・河の位置・元のポン）は副露と加槓で読む。ほかの種類の欄には中立の値（−1・{@code false}・{@code
 * null}）を書く。{@link Meld} は不変の共有の値なので参照を写してよい。持ち主は 1 つの {@link GameState} で、分岐は {@link #copyFrom}
 * で写す（配列ごとに arraycopy 1 回）。
 */
public final class PublicEventLog {

  /** 鳴かれていない打牌の {@link #calledBy}。 */
  public static final int NOT_CALLED = -1;

  /** 加槓でない出来事の {@link #ponEvent}。 */
  public static final int NO_PON = -1;

  private static final int CAPACITY = RoundLimits.MAX_PUBLIC_EVENTS;

  private final byte[] seat = new byte[CAPACITY];
  private final Meld[] meld = new Meld[CAPACITY];
  private final byte[] tile = new byte[CAPACITY];
  private final boolean[] aka = new boolean[CAPACITY];
  private final boolean[] tsumogiri = new boolean[CAPACITY];
  private final boolean[] riichi = new boolean[CAPACITY];
  private final byte[] calledBy = new byte[CAPACITY];
  private final short[] turn = new short[CAPACITY];
  private final short[] sequence = new short[CAPACITY];
  private final byte[] afterRiver = new byte[CAPACITY];
  private final byte[] ponEvent = new byte[CAPACITY];
  private int size;
  private int melds;

  /** 空の記録を作る。局に属さない {@link Hand} も空の記録を 1 つ持つ。 */
  PublicEventLog() {}

  /** 出来事の数（次に足す出来事の番号）。 */
  public int size() {
    return size;
  }

  /** 打牌なら {@code true}、副露・加槓なら {@code false}。 */
  public boolean isDiscard(int event) {
    return meld[at(event)] == null;
  }

  /** 出来事の作者の絶対席。 */
  public int seat(int event) {
    return seat[at(event)];
  }

  /** 打牌の牌の種類。 */
  public int tile(int event) {
    return tile[at(event)];
  }

  /** 打牌が赤の 5 なら {@code true}。 */
  public boolean aka(int event) {
    return aka[at(event)];
  }

  /** 打牌が自摸切りなら {@code true}。 */
  public boolean tsumogiri(int event) {
    return tsumogiri[at(event)];
  }

  /** 打牌が立直の宣言牌なら {@code true}。 */
  public boolean riichi(int event) {
    return riichi[at(event)];
  }

  /** 打牌を鳴いた絶対席。鳴かれていなければ {@link #NOT_CALLED}。 */
  public int calledBy(int event) {
    return calledBy[at(event)];
  }

  /** 打牌の 0 始まりの巡目（打った時の {@link GameState#getTurnNumber()} の値）。 */
  public int turn(int event) {
    return turn[at(event)];
  }

  /** 打牌の局の中の通し番号（全席を通した 0 始まりの打牌の連番）。 */
  public int sequence(int event) {
    return sequence[at(event)];
  }

  /** 副露・加槓の面子（加槓は加槓の面子）。打牌なら {@code null}。 */
  public Meld meld(int event) {
    return meld[at(event)];
  }

  /** 副露・加槓をした時の作者の河の枚数（鳴いた直後の打牌の河の位置）。 */
  public int afterRiver(int event) {
    return afterRiver[at(event)];
  }

  /** 加槓の元のポンの出来事の番号。加槓でなければ {@link #NO_PON}。 */
  public int ponEvent(int event) {
    return ponEvent[at(event)];
  }

  /** この局で副露（暗槓・加槓を含む）が 1 つでもあれば {@code true}。 */
  public boolean anyMeld() {
    return melds != 0;
  }

  /** 打牌を足し、その番号を返す。 */
  int appendDiscard(
      int seat,
      int tile,
      boolean aka,
      boolean tsumogiri,
      boolean riichi,
      int turn,
      int sequence) {
    requireRoom();
    int event = next(seat);
    this.tile[event] = (byte) tile;
    this.aka[event] = aka;
    this.tsumogiri[event] = tsumogiri;
    this.riichi[event] = riichi;
    this.calledBy[event] = NOT_CALLED;
    this.turn[event] = (short) turn;
    this.sequence[event] = (short) sequence;
    clearMeldFields(event);
    return event;
  }

  /**
   * 副露（チー・ポン・大明槓・暗槓）を足し、その番号を返す。鳴きなら、最後の出来事が元の打牌であることを確かめて鳴いた席を書く。
   *
   * @param seat 副露した絶対席
   * @param meld 足す面子（加槓は {@link #appendKakan}）
   * @param afterRiver 副露した時の作者の河の枚数
   * @throws IllegalStateException 鳴きの直前の出来事が元の打牌でないとき
   */
  int appendMeld(int seat, Meld meld, int afterRiver) {
    if (meld.type() == Meld.Type.KAKAN) {
      throw new IllegalArgumentException("a kakan replaces its pon: append it with appendKakan");
    }
    requireRoom();
    requireAfterRiver(afterRiver);
    if (meld.isCalledFromDiscard()) {
      markCalled(seat, meld);
    }
    int event = next(seat);
    writeMeld(event, meld, afterRiver, NO_PON);
    return event;
  }

  /**
   * 加槓を自分の出来事として足し、その番号を返す。元のポンの出来事は書き換えない。
   *
   * @param seat 加槓した絶対席
   * @param kakan 加槓の面子
   * @param ponEvent 元のポンの出来事の番号
   * @param afterRiver 加槓した時の作者の河の枚数
   */
  int appendKakan(int seat, Meld kakan, int ponEvent, int afterRiver) {
    Meld pon = meld[at(ponEvent)];
    if (this.seat[ponEvent] != seat || pon == null || !kakan.isKakanUpgradeOf(pon)) {
      throw new IllegalArgumentException(
          "kakan " + kakan + " by seat " + seat + " is not an upgrade of event " + ponEvent);
    }
    requireRoom();
    requireAfterRiver(afterRiver);
    int event = next(seat);
    writeMeld(event, kakan, afterRiver, ponEvent);
    return event;
  }

  /** {@code other} の中身を写す（分岐）。 */
  void copyFrom(PublicEventLog other) {
    int n = other.size;
    System.arraycopy(other.seat, 0, seat, 0, n);
    System.arraycopy(other.meld, 0, meld, 0, n);
    System.arraycopy(other.tile, 0, tile, 0, n);
    System.arraycopy(other.aka, 0, aka, 0, n);
    System.arraycopy(other.tsumogiri, 0, tsumogiri, 0, n);
    System.arraycopy(other.riichi, 0, riichi, 0, n);
    System.arraycopy(other.calledBy, 0, calledBy, 0, n);
    System.arraycopy(other.turn, 0, turn, 0, n);
    System.arraycopy(other.sequence, 0, sequence, 0, n);
    System.arraycopy(other.afterRiver, 0, afterRiver, 0, n);
    System.arraycopy(other.ponEvent, 0, ponEvent, 0, n);
    size = n;
    melds = other.melds;
  }

  /** 局の初めの空の記録に戻す。 */
  void clear() {
    size = 0;
    melds = 0;
  }

  /** 鳴きの元の打牌（最後の出来事）を確かめ、鳴いた席を書く。 */
  private void markCalled(int caller, Meld meld) {
    int last = size - 1;
    int source = (caller + meld.relativeSource().playerOffset()) % GameState.NUM_PLAYERS;
    if (last < 0
        || this.meld[last] != null
        || calledBy[last] != NOT_CALLED
        || seat[last] != source
        || tile[last] != meld.calledTileType()) {
      throw new IllegalStateException(
          "meld " + meld + " by seat " + caller + " does not follow its called discard");
    }
    calledBy[last] = (byte) caller;
  }

  private void requireRoom() {
    if (size == CAPACITY) {
      throw new IllegalStateException("a round has at most " + CAPACITY + " public events");
    }
  }

  private static void requireAfterRiver(int afterRiver) {
    if (afterRiver < 0 || afterRiver > RoundLimits.MAX_SEAT_DISCARDS) {
      throw new IllegalArgumentException("afterRiver out of range: " + afterRiver);
    }
  }

  /** 次の出来事の作者を書いて番号を返す。呼ぶ前に {@link #requireRoom} で容量を確かめる。 */
  private int next(int seat) {
    this.seat[size] = (byte) seat;
    return size++;
  }

  private void writeMeld(int event, Meld meld, int afterRiver, int ponEvent) {
    this.meld[event] = meld;
    this.afterRiver[event] = (byte) afterRiver;
    this.ponEvent[event] = (byte) ponEvent;
    tile[event] = -1;
    aka[event] = false;
    tsumogiri[event] = false;
    riichi[event] = false;
    calledBy[event] = NOT_CALLED;
    turn[event] = -1;
    sequence[event] = -1;
    melds++;
  }

  private void clearMeldFields(int event) {
    meld[event] = null;
    afterRiver[event] = -1;
    ponEvent[event] = NO_PON;
  }

  private int at(int event) {
    return Objects.checkIndex(event, size);
  }
}
