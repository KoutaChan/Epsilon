package com.epsilon.calculate.scoring;

import java.util.Objects;

/**
 * 同じ手牌形を複数の和了条件で評価するための固定容量バッファ。各位置に条件ごとの最良結果を保持する。{@code configure}で条件を設定し、{@code
 * size}で使用する範囲を指定する。
 */
public final class HandScoreBatchBuffer {
  final WinMethod[] methods;
  final WinConditions[] conditions;
  final int[] ownedAkaMasks;
  final long[] closedContextYakuMasks;
  final long[] openContextYakuMasks;
  final long[] contextYakumanMasks;
  final HandScoreBuffer[] results;
  int size;

  public HandScoreBatchBuffer(int capacity) {
    if (capacity < 1) throw new IllegalArgumentException("capacity must be positive");
    methods = new WinMethod[capacity];
    conditions = new WinConditions[capacity];
    ownedAkaMasks = new int[capacity];
    closedContextYakuMasks = new long[capacity];
    openContextYakuMasks = new long[capacity];
    contextYakumanMasks = new long[capacity];
    results = new HandScoreBuffer[capacity];
    for (int i = 0; i < capacity; i++) results[i] = new HandScoreBuffer();
  }

  public HandScoreBatchBuffer configure(
      int index, WinMethod method, WinConditions condition, int ownedAkaMask) {
    Objects.requireNonNull(method, "method");
    Objects.requireNonNull(condition, "condition");
    if (methods[index] != method || conditions[index] != condition) {
      // 不変の局況だけを準備し、手牌ごとの門前判定は採点時に行う。
      closedContextYakuMasks[index] = YakuRules.evaluateContextYakuMask(true, method, condition);
      openContextYakuMasks[index] = YakuRules.evaluateContextYakuMask(false, method, condition);
      contextYakumanMasks[index] = YakuRules.evaluateContextYakumanMask(method, condition);
      methods[index] = method;
      conditions[index] = condition;
    }
    ownedAkaMasks[index] = ownedAkaMask;
    return this;
  }

  public HandScoreBatchBuffer size(int size) {
    if (size < 1 || size > results.length) throw new IllegalArgumentException("invalid batch size");
    this.size = size;
    return this;
  }

  public int size() {
    return size;
  }

  public HandScoreBuffer result(int index) {
    if (index < 0 || index >= size) throw new IndexOutOfBoundsException(index);
    return results[index];
  }
}
