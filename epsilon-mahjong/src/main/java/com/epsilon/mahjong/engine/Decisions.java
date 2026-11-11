package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.ActionSection;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import java.util.Objects;
import java.util.function.IntPredicate;

/**
 * 判断の時点で、判断する席ごとの合法手。手番では 1 席が、他家の応答では最大で（席の数 − 1）席が判断する。エンジンが持ち、
 * 判断の時点ごとに書き直す。中身は局を次に進める呼び出しまで有効で、ここでは何も計算しない。合法手は行動の番号（{@link Action}）
 * で持ち、並びは合法手を作る側が決める。
 *
 * <p>判断は 1 席ずつ書く。書き始めてから合法手を足し、書き終える。合法手の無い判断は書き終えるときに捨て、判断 ID は残した判断に
 * だけ付ける。
 *
 * <p>ツモ和了かロンを合法手に持つ判断は、合法手を作るときに求めた和了の点を持ち、精算ではその点を使う。立直なら裏ドラを含むので、
 * 外へは出さない。
 */
public final class Decisions {

  /**
   * 1 席の判断が持つ合法手の上限。手番の打牌と立直の打牌は、それぞれ 14 以下になる（ツモの前の 13 枚の、種類と赤の組み合わせに
   * ツモ切りを足した数）。ほかはツモ和了・九種九牌・抜き・暗槓・加槓だけ。暗槓できる 4 枚があれば打牌の組み合わせはその分減り、
   * 加槓できる手は鳴いていて立直できないので、合計は 31 を超えない。
   */
  public static final int MAX_LEGAL = 32;

  private static final EngineDecisionKind[] KINDS = EngineDecisionKind.values();

  private final Variant variant;

  /** 行動の番号の数（{@link ActionSection#space}）。 */
  private final int space;

  private final long[] ids;
  private final byte[] seats;
  private final byte[] kinds;
  private final byte[] counts;

  /** 判断ごとの合法手。判断の番号 × 32 ＋ i の位置に、i 番目の合法手を置く。 */
  private final short[] legal;

  /** 判断ごとの和了の点。和了を合法手に持たない判断では意味が無い。 */
  private final WinScore[] wins;

  /** 判断ごとに、和了を合法手に持つなら true。 */
  private final boolean[] winning;

  private int size;

  /** 書きかけの判断がある。 */
  private boolean open;

  Decisions(Variant variant) {
    this.variant = variant;
    space = ActionSection.space(variant);
    int slots = variant.seats() - 1;
    ids = new long[slots];
    seats = new byte[slots];
    kinds = new byte[slots];
    counts = new byte[slots];
    legal = new short[slots * MAX_LEGAL];
    wins = new WinScore[slots];
    winning = new boolean[slots];
    for (int slot = 0; slot < slots; slot++) {
      wins[slot] = new WinScore();
    }
  }

  public int size() {
    return size;
  }

  /**
   * 判断 ID。番号は合法手のある判断にだけ使い、コピーしたエンジンでも同じ番号から続く。
   *
   * @param slot 何番目の判断か
   */
  public long id(int slot) {
    return ids[Objects.checkIndex(slot, size)];
  }

  /** 判断する絶対席。 */
  public int seat(int slot) {
    return seats[Objects.checkIndex(slot, size)];
  }

  public EngineDecisionKind kind(int slot) {
    return KINDS[kinds[Objects.checkIndex(slot, size)]];
  }

  public int legalCount(int slot) {
    return counts[Objects.checkIndex(slot, size)];
  }

  /**
   * 合法手の行動の番号。
   *
   * @param slot 何番目の判断か
   * @param index 何番目の合法手か
   */
  public int legal(int slot, int index) {
    return legal[Objects.checkIndex(slot, size) * MAX_LEGAL + Objects.checkIndex(index, counts[slot])];
  }

  /** その席の判断が何番目か。その席が判断しないなら −1。 */
  public int slotOf(int seat) {
    for (int slot = 0; slot < size; slot++) {
      if (seats[slot] == seat) {
        return slot;
      }
    }
    return -1;
  }

  void clear() {
    size = 0;
    open = false;
  }

  /** 席の判断を書き始める。合法手は {@link #add} で足し、{@link #close} で書き終える。 */
  void open(int seat, EngineDecisionKind kind) {
    if (open) {
      throw new IllegalStateException("Slot " + size + " is still open");
    }
    if (size == ids.length) {
      throw new IllegalStateException("At most " + ids.length + " decisions at a boundary");
    }
    if (!kind.in(variant)) {
      throw new IllegalArgumentException(variant + " has no decision kind " + kind);
    }
    if (slotOf(Objects.checkIndex(seat, variant.seats())) >= 0) {
      throw new IllegalStateException("Seat " + seat + " already has a decision");
    }
    seats[size] = (byte) seat;
    kinds[size] = (byte) kind.ordinal();
    counts[size] = 0;
    winning[size] = false;
    open = true;
  }

  void add(int code) {
    if (!open) {
      throw new IllegalStateException("No decision slot is open");
    }
    if (counts[size] == MAX_LEGAL) {
      throw new IllegalStateException("A decision holds at most " + MAX_LEGAL + " legal actions");
    }
    legal[size * MAX_LEGAL + counts[size]] = (short) Objects.checkIndex(code, space);
    counts[size]++;
  }

  /** ツモ和了かロンを合法手に足し、その和了の点をコピーして持つ。 */
  void addWin(int code, WinScore score) {
    Action.Type type = Action.of(code).type();
    if (type != Action.Type.TSUMO_AGARI && type != Action.Type.RON_AGARI) {
      throw new IllegalArgumentException("Not a win: " + Action.of(code));
    }
    add(code);
    wins[size].copyFrom(score);
    winning[size] = true;
  }

  /** 判断がツモ和了かロンを合法手に持つなら true。 */
  boolean hasWin(int slot) {
    return winning[Objects.checkIndex(slot, size)];
  }

  /** 和了を合法手に持つ判断で、その和了の点。立直なら裏ドラを含む。 */
  WinScore win(int slot) {
    return wins[Objects.checkIndex(slot, size)];
  }

  /**
   * 条件を満たさない合法手を除き、残りを元の並びのまま前に詰める。
   *
   * @param slot 何番目の判断か
   * @param kept 行動の番号を受け、残す手なら true を返す条件
   * @return 残った合法手の数
   */
  int retain(int slot, IntPredicate kept) {
    int base = Objects.checkIndex(slot, size) * MAX_LEGAL;
    int count = 0;
    for (int index = 0; index < counts[slot]; index++) {
      short code = legal[base + index];
      if (kept.test(code)) {
        legal[base + count++] = code;
      }
    }
    counts[slot] = (byte) count;
    return count;
  }

  /**
   * 書きかけの判断を書き終える。合法手があれば判断 ID を付けて残し、無ければ捨てる。
   *
   * @param id 付ける判断 ID
   * @return 残したなら true
   */
  boolean close(long id) {
    if (!open) {
      throw new IllegalStateException("No decision slot is open");
    }
    open = false;
    if (counts[size] == 0) {
      return false;
    }
    ids[size] = id;
    size++;
    return true;
  }

  void copyFrom(Decisions other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.ids, 0, ids, 0, ids.length);
    System.arraycopy(other.seats, 0, seats, 0, seats.length);
    System.arraycopy(other.kinds, 0, kinds, 0, kinds.length);
    System.arraycopy(other.counts, 0, counts, 0, counts.length);
    System.arraycopy(other.legal, 0, legal, 0, legal.length);
    System.arraycopy(other.winning, 0, winning, 0, winning.length);
    for (int slot = 0; slot < wins.length; slot++) {
      wins[slot].copyFrom(other.wins[slot]);
    }
    size = other.size;
    open = other.open;
  }
}
