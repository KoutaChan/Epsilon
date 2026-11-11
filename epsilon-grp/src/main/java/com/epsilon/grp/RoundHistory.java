package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * 1 つの半荘で、各局が始まったときの卓を並べた履歴。GruSinkhorn が読む。局が始まるたびに足し、半荘を分岐させるときは
 * {@code copy} か {@code copyFrom} でコピーする。行は RoundField の並びで書いた整数。
 *
 * <p>バージョンは中身を見分ける番号。局を足すと新しい番号に替わる。コピーした履歴は元と同じ番号を持ち、空の履歴は 0。
 * 番号は使い回さないので、バージョンが同じなら中身も同じで、Φ のメモと GruSinkhorn の状態のメモのキーに使える。中身が同じでも、
 * 別々に局を足した履歴はバージョンが違う（メモに当たらないだけで、値は変わらない）。
 *
 * <p>メモリを確保するのは、容量を超えて足すとき（容量を倍にする）と、容量が足りない {@code copyFrom} だけ。同時に使えるのは
 * 1 スレッドだけ。
 */
public final class RoundHistory {

  /** 最初の容量（局の数）。 */
  private static final int INITIAL_ROUNDS = 32;

  /** 最後に配ったバージョン。 */
  private static final AtomicLong STAMPS = new AtomicLong();

  private final Variant variant;
  private final int width;

  /** 行を並べた配列。並びは [局 × 幅 ＋ 列]。 */
  private int[] rows;

  private int size;
  private long stamp;

  public RoundHistory(Variant variant) {
    this(variant, INITIAL_ROUNDS * RoundField.width(variant));
  }

  private RoundHistory(Variant variant, int capacity) {
    this.variant = Objects.requireNonNull(variant, "variant");
    width = RoundField.width(variant);
    rows = new int[capacity];
  }

  public Variant variant() {
    return variant;
  }

  /** 局の数。 */
  public int size() {
    return size;
  }

  public int round(int index) {
    return value(index, RoundField.ROUND.offset(variant));
  }

  public int honba(int index) {
    return value(index, RoundField.HONBA.offset(variant));
  }

  public int kyotaku(int index) {
    return value(index, RoundField.KYOTAKU.offset(variant));
  }

  /**
   * 局が始まったときの点。
   *
   * @param index 何局目か（0 始まり）
   * @param seat 絶対席
   */
  public int score(int index, int seat) {
    Objects.checkIndex(seat, variant.seats());
    return value(index, RoundField.SCORE.offset(variant) + seat);
  }

  /** 局が始まったときの卓を足す。終局した卓や、ルールが違う卓では IllegalArgumentException を投げる。 */
  public void add(TableState start) {
    requireVariant(start.variant());
    if (start.gameOver()) {
      throw new IllegalArgumentException("A finished game starts no round: " + start);
    }
    int at = reserve();
    RoundField.write(start, rows, at);
    added();
  }

  /** 局が始まったときの卓を足す。ルールが違えば IllegalArgumentException を投げる。 */
  public void add(GrpState start) {
    requireVariant(start.variant());
    int at = reserve();
    RoundField.write(start, rows, at);
    added();
  }

  /** 空にする（容量は残す）。 */
  public void clear() {
    size = 0;
    stamp = 0;
  }

  /** 渡した履歴と同じ中身とバージョンにする。容量が足りなければメモリを確保する。ルールが違えば IllegalArgumentException を投げる。 */
  public void copyFrom(RoundHistory other) {
    requireVariant(other.variant);
    int length = other.size * width;
    if (rows.length < length) {
      rows = new int[other.rows.length];
    }
    System.arraycopy(other.rows, 0, rows, 0, length);
    size = other.size;
    stamp = other.stamp;
  }

  /** 中身もバージョンも容量も同じコピーを作る。 */
  public RoundHistory copy() {
    RoundHistory copy = new RoundHistory(variant, rows.length);
    copy.copyFrom(this);
    return copy;
  }

  long stamp() {
    return stamp;
  }

  /**
   * 行の中の 1 つの値。
   *
   * @param index 何局目か（0 始まり）
   * @param column 行の中の列
   */
  int value(int index, int column) {
    Objects.checkIndex(index, size);
    return rows[index * width + column];
  }

  /** 次の行を書く位置を返す。容量が足りなければ倍に広げる。 */
  private int reserve() {
    int at = size * width;
    if (rows.length < at + width) {
      rows = Arrays.copyOf(rows, Math.max(2 * rows.length, at + width));
    }
    return at;
  }

  private void added() {
    size++;
    stamp = STAMPS.incrementAndGet();
  }

  private void requireVariant(Variant other) {
    if (other != variant) {
      throw new IllegalArgumentException("Variant mismatch: history " + variant + ", table " + other);
    }
  }

  @Override
  public String toString() {
    return "RoundHistory[" + variant + " size=" + size + " stamp=" + stamp + "]";
  }
}
