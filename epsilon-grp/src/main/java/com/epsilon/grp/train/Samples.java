package com.epsilon.grp.train;

import java.util.Arrays;
import java.util.SplittableRandom;

/** 決まった乱数の種で、番号を重複なく抜き出す。単調性の確かめと、同点のまわりの段差の測定で使う。 */
final class Samples {

  private Samples() {}

  /** 0 から総数 − 1 までの番号から重複なく min(個数, 総数) 個を選び、昇順に並べて返す。選び方は、途中で止める Fisher–Yates。 */
  static int[] choose(int size, int count, long seed) {
    int[] order = new int[size];
    for (int index = 0; index < size; index++) {
      order[index] = index;
    }
    int chosen = Math.min(count, size);
    SplittableRandom random = new SplittableRandom(seed);
    for (int index = 0; index < chosen; index++) {
      int other = index + random.nextInt(size - index);
      int held = order[index];
      order[index] = order[other];
      order[other] = held;
    }
    int[] picked = Arrays.copyOf(order, chosen);
    Arrays.sort(picked);
    return picked;
  }
}
