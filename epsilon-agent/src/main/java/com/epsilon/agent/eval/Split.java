package com.epsilon.agent.eval;

import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * 席の分け方。スロットごとに何席を受け持つかを並べたもの。どのスロットも 1 席以上で、和は卓の席の数。設定では "1v2"・"2v2"・"1v1v1v1" の
 * 形で書く。
 *
 * <p>割り当ては、席 0 から順に座るスロットの番号を並べた列で表す（各スロットは受け持つ席の数だけ現れる）。割り当ての番号は、その列を辞書順に
 * 並べたときの順位。スロット 0 が 1 席の分け方では、割り当て p はスロット 0 を席 p に置く。
 *
 * @param seats スロットごとの席の数（2 スロット以上）
 */
public record Split(List<Integer> seats) {

  /** 書き方でスロットを区切る文字。 */
  private static final String SEPARATOR = "v";

  public Split {
    seats = List.copyOf(seats);
    if (seats.size() < 2) {
      throw new IllegalArgumentException("A split seats at least two slots: " + seats);
    }
    for (int count : seats) {
      if (count < 1) {
        throw new IllegalArgumentException("Every slot of a split takes a seat: " + seats);
      }
    }
  }

  /**
   * スロット 0 が 1 席、スロット 1 が残りの席を受け持つ分け方。
   *
   * @param variant 卓のルール（四人打ちか三人打ちか）
   * @return 分け方
   */
  public static Split oneAgainstRest(Variant variant) {
    return new Split(List.of(1, variant.seats() - 1));
  }

  /**
   * "1v2" の形の書き方を読む。
   *
   * @param text 書き方
   * @param slots 卓のスロットの数
   * @param variant 卓のルール（四人打ちか三人打ちか）
   * @return 分け方
   * @throws IllegalArgumentException 形が違うか、スロットの数が合わないか、席の数の和が卓の席の数でないか、0 席のスロットがあるとき
   */
  public static Split parse(String text, int slots, Variant variant) {
    Objects.requireNonNull(text, "text");
    String[] parts = text.trim().split(SEPARATOR, -1);
    List<Integer> seats = new ArrayList<>(parts.length);
    for (String part : parts) {
      if (part.isEmpty() || !part.chars().allMatch(Character::isDigit) || part.length() > 2) {
        throw new IllegalArgumentException("A split is written like 1v2: '" + text + "'");
      }
      seats.add(Integer.parseInt(part));
    }
    if (seats.size() != slots) {
      throw new IllegalArgumentException("Split " + text + " has " + seats.size() + " slots, the table " + slots);
    }
    Split split = new Split(seats);
    if (split.seatCount() != variant.seats()) {
      throw new IllegalArgumentException(
          "Split " + text + " seats " + split.seatCount() + ", " + variant + " has " + variant.seats());
    }
    return split;
  }

  /** @return スロットの数 */
  public int slots() {
    return seats.size();
  }

  /** @return 全てのスロットの席の数の和 */
  public int seatCount() {
    int sum = 0;
    for (int count : seats) {
      sum += count;
    }
    return sum;
  }

  /**
   * @param slot スロットの番号
   * @return そのスロットの席の数
   */
  public int seatsOf(int slot) {
    return seats.get(slot);
  }

  /** @return 割り当ての数（スロットごとの席の数から求める多項係数） */
  public int placements() {
    int[] left = new int[seats.size()];
    for (int slot = 0; slot < left.length; slot++) {
      left[slot] = seats.get(slot);
    }
    return arrangements(left);
  }

  /**
   * 割り当てで、席ごとに座るスロットを書く。
   *
   * @param placement 割り当ての番号（0 から {@link #placements} − 1）
   * @param slotOfSeat 席ごとのスロットの番号を書く先（長さは席の数以上）
   */
  public void slotsOf(int placement, int[] slotOfSeat) {
    Objects.checkIndex(placement, placements());
    int[] left = new int[seats.size()];
    for (int slot = 0; slot < left.length; slot++) {
      left[slot] = seats.get(slot);
    }
    int rank = placement;
    int count = seatCount();
    for (int seat = 0; seat < count; seat++) {
      for (int slot = 0; slot < left.length; slot++) {
        if (left[slot] == 0) {
          continue;
        }
        left[slot]--;
        int after = arrangements(left);
        if (rank < after) {
          slotOfSeat[seat] = slot;
          break;
        }
        rank -= after;
        left[slot]++;
      }
    }
  }

  /** "1v2" の形。 */
  @Override
  public String toString() {
    StringBuilder text = new StringBuilder();
    for (int slot = 0; slot < seats.size(); slot++) {
      if (slot > 0) {
        text.append(SEPARATOR);
      }
      text.append(seats.get(slot));
    }
    return text.toString();
  }

  /** スロットごとの残りの席の数から、並べ方の数（多項係数）を求める。 */
  private static int arrangements(int[] left) {
    int total = 0;
    long result = 1;
    for (int count : left) {
      for (int k = 1; k <= count; k++) {
        total++;
        result = result * total / k;
      }
    }
    return Math.toIntExact(result);
  }
}
