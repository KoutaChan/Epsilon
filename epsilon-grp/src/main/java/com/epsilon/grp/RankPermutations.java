package com.epsilon.grp;

import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * 最終順位の S! 通りの順列の表。ルールごとに 1 つ持つ。順列は相対席ごとの順位（0 始まり）を並べたもので、番号はその並びの辞書順
 * で付ける。どのメソッドもメモリを確保しない。
 */
public final class RankPermutations {

  private static final Map<Variant, RankPermutations> TABLES = tables();

  private final Variant variant;
  private final int seats;
  private final int count;

  /** 各順列で席が取る順位。並びは [順列 × S ＋ 席]。 */
  private final byte[] table;

  /** {@link #rotate} の値を前もって求めた表。並びは [ずらす席の数 × S! ＋ 順列]。 */
  private final int[] rotations;

  private RankPermutations(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    count = factorial(seats);
    table = new byte[count * seats];
    int[] current = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      current[seat] = seat;
    }
    for (int permutation = 0; permutation < count; permutation++) {
      for (int seat = 0; seat < seats; seat++) {
        table[permutation * seats + seat] = (byte) current[seat];
      }
      advance(current);
    }
    rotations = new int[seats * count];
    for (int shift = 0; shift < seats; shift++) {
      for (int permutation = 0; permutation < count; permutation++) {
        for (int seat = 0; seat < seats; seat++) {
          current[seat] = table[permutation * seats + Math.floorMod(seat - shift, seats)];
        }
        rotations[shift * count + permutation] = index(current);
      }
    }
  }

  public static RankPermutations of(Variant variant) {
    return TABLES.get(Objects.requireNonNull(variant, "variant"));
  }

  public Variant variant() {
    return variant;
  }

  public int count() {
    return count;
  }

  /**
   * 順列の中で相対席が取る順位を返す。
   *
   * @param permutation 順列の番号
   * @param seat 相対席
   */
  public int rank(int permutation, int seat) {
    Objects.checkIndex(permutation, count);
    Objects.checkIndex(seat, seats);
    return table[permutation * seats + seat];
  }

  /** 相対席ごとの順位の並びから、順列の番号を求める。順列になっていなければ IllegalArgumentException を投げる。 */
  public int index(int[] ranks) {
    if (ranks.length != seats) {
      throw new IllegalArgumentException(variant + " needs " + seats + " ranks, got " + ranks.length);
    }
    for (int seat = 0; seat < seats; seat++) {
      boolean repeated = false;
      for (int earlier = 0; earlier < seat; earlier++) {
        repeated |= ranks[earlier] == ranks[seat];
      }
      if (ranks[seat] < 0 || ranks[seat] >= seats || repeated) {
        throw new IllegalArgumentException("Ranks are not a permutation of 0.." + (seats - 1));
      }
    }
    int index = 0;
    for (int seat = 0; seat < seats; seat++) {
      int smallerLater = 0;
      for (int later = seat + 1; later < seats; later++) {
        smallerLater += ranks[later] < ranks[seat] ? 1 : 0;
      }
      index = index * (seats - seat) + smallerLater;
    }
    return index;
  }

  /**
   * 終局の点で決まる順列の番号を返す。順位は点の高い順で、同点なら起家に近い席（絶対席の小さい席）が上（Placement）。
   *
   * @param finalScores 絶対席ごとの最終の点
   * @param dealer 親の絶対席。順列の席は、この席を 0 とする相対席
   */
  public int realized(int[] finalScores, int dealer) {
    if (finalScores.length != seats) {
      throw new IllegalArgumentException(variant + " needs " + seats + " scores, got " + finalScores.length);
    }
    Objects.checkIndex(dealer, seats);
    int index = 0;
    for (int seat = 0; seat < seats; seat++) {
      int rank = Placement.rank(finalScores, Seats.absolute(dealer, seat, seats));
      int smallerLater = 0;
      for (int later = seat + 1; later < seats; later++) {
        smallerLater += Placement.rank(finalScores, Seats.absolute(dealer, later, seats)) < rank ? 1 : 0;
      }
      index = index * (seats - seat) + smallerLater;
    }
    return index;
  }

  /**
   * 親が d 席先へ移った卓の相対席で数えた順列 π を、元の卓の相対席で数えた順列 σ に直す。σ(r) = π((r − d) mod S)。次局を
   * 数え上げるとき、親が移った後の卓でネットワークが出した値を、元の卓の相対席に直すのに使う。
   *
   * @param permutation 移った後の卓の相対席で数えた順列 π の番号
   * @param shift 親が移った席の数 d
   * @return σ の番号
   */
  public int rotate(int permutation, int shift) {
    Objects.checkIndex(permutation, count);
    return rotations[Math.floorMod(shift, seats) * count + permutation];
  }

  /**
   * 順列の確率から、絶対席ごとの Φ_i = Σ_π p(π)·u_{π(i)} を求める。π の番号の順に、π の中では相対席の順に足す。
   *
   * @param probabilities 順列ごとの確率（長さ S!）。席は親を 0 とする相対席
   * @param dealer 親の絶対席
   * @param out 出力先（長さ S）
   */
  public void phi(double[] probabilities, int dealer, RankUtility utility, double[] out) {
    if (probabilities.length != count || out.length != seats) {
      throw new IllegalArgumentException(
          "Expected " + count + " probabilities and " + seats + " values, got " + probabilities.length + " and "
              + out.length);
    }
    Objects.checkIndex(dealer, seats);
    Arrays.fill(out, 0);
    for (int permutation = 0; permutation < count; permutation++) {
      double probability = probabilities[permutation];
      for (int seat = 0; seat < seats; seat++) {
        out[Seats.absolute(dealer, seat, seats)] +=
            probability * utility.value(variant, table[permutation * seats + seat]);
      }
    }
  }

  /**
   * 順列の確率から、席ごとの順位の周辺確率を求める。席は順列と同じ相対席のまま。
   *
   * @param probabilities 順列ごとの確率（長さ S!）
   * @param out 出力先（長さ S²。並びは [席 × S ＋ 順位]）
   */
  public void marginals(double[] probabilities, double[] out) {
    marginals(probabilities, 0, out);
  }

  /**
   * 順列の確率から、絶対席ごとの順位の周辺確率を求める。π の番号の順に、π の中では相対席の順に足す。
   *
   * @param probabilities 順列ごとの確率（長さ S!）。席は親を 0 とする相対席
   * @param dealer 親の絶対席
   * @param out 出力先（長さ S²。並びは [絶対席 × S ＋ 順位]）
   */
  public void marginals(double[] probabilities, int dealer, double[] out) {
    if (probabilities.length != count || out.length != seats * seats) {
      throw new IllegalArgumentException(
          "Expected " + count + " probabilities and " + seats * seats + " marginals, got "
              + probabilities.length + " and " + out.length);
    }
    Objects.checkIndex(dealer, seats);
    Arrays.fill(out, 0);
    for (int permutation = 0; permutation < count; permutation++) {
      for (int seat = 0; seat < seats; seat++) {
        out[Seats.absolute(dealer, seat, seats) * seats + table[permutation * seats + seat]] +=
            probabilities[permutation];
      }
    }
  }

  /** 辞書順で次の順列にする。最後の順列なら何もしない。 */
  private static void advance(int[] order) {
    int pivot = order.length - 2;
    while (pivot >= 0 && order[pivot] > order[pivot + 1]) {
      pivot--;
    }
    if (pivot < 0) {
      return;
    }
    int swap = order.length - 1;
    while (order[swap] < order[pivot]) {
      swap--;
    }
    exchange(order, pivot, swap);
    for (int left = pivot + 1, right = order.length - 1; left < right; left++, right--) {
      exchange(order, left, right);
    }
  }

  private static int factorial(int n) {
    int product = 1;
    for (int factor = 2; factor <= n; factor++) {
      product *= factor;
    }
    return product;
  }

  private static void exchange(int[] order, int i, int j) {
    int held = order[i];
    order[i] = order[j];
    order[j] = held;
  }

  private static Map<Variant, RankPermutations> tables() {
    Map<Variant, RankPermutations> tables = new EnumMap<>(Variant.class);
    for (Variant variant : Variant.values()) {
      tables.put(variant, new RankPermutations(variant));
    }
    return tables;
  }
}
