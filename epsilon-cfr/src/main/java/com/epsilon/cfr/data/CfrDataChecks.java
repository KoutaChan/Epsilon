package com.epsilon.cfr.data;

import java.util.HashSet;

final class CfrDataChecks {
  static final int MAX_ACTIONS = 4096;
  static final int MAX_OBSERVATION_BYTES = 16 * 1024 * 1024;

  private CfrDataChecks() {}

  static void decision(
      int[] actions, int selected, double[] policy, ObservationPayload observation) {
    if (actions == null
        || policy == null
        || observation == null
        || actions.length == 0
        || actions.length > MAX_ACTIONS
        || actions.length != policy.length
        || selected < 0
        || selected >= actions.length
        || observation.size() > MAX_OBSERVATION_BYTES) {
      throw new IllegalArgumentException("Invalid CFR decision shape");
    }
    // 通常の合法手数では一時コレクションを作らない。大きな入力だけは二乗の比較を避ける。
    var ids = actions.length <= 64 ? null : new HashSet<Integer>();
    double sum = 0;
    for (int i = 0; i < actions.length; i++) {
      if (actions[i] < 0
          || duplicate(actions, i, ids)
          || !Double.isFinite(policy[i])
          || policy[i] < 0
          || policy[i] > 1) {
        throw new IllegalArgumentException("Invalid action identity or target policy");
      }
      sum += policy[i];
    }
    if (Math.abs(sum - 1) > 1e-9) {
      throw new IllegalArgumentException("Target policy must sum to one: " + sum);
    }
  }

  private static boolean duplicate(int[] actions, int index, HashSet<Integer> ids) {
    if (ids != null) return !ids.add(actions[index]);
    for (int previous = 0; previous < index; previous++) {
      if (actions[previous] == actions[index]) return true;
    }
    return false;
  }

  static void logWeight(double value) {
    if (!Double.isFinite(value) && value != Double.NEGATIVE_INFINITY) {
      throw new IllegalArgumentException("Invalid log weight: " + value);
    }
  }
}
