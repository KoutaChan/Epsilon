package com.epsilon.cfr.algorithm;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/** 符号付き後悔から方策を求め、すべての合法手に正の選択確率を与えて探索する。 */
public final class RegretMatching {
  private RegretMatching() {}

  public static double[] policy(double[] regrets) {
    if (regrets == null || regrets.length == 0) throw new IllegalArgumentException("No actions");
    double max = 0;
    for (double regret : regrets) {
      if (!Double.isFinite(regret)) throw new IllegalArgumentException("Nonfinite regret");
      max = Math.max(max, regret);
    }
    double[] policy = new double[regrets.length];
    if (max == 0) {
      Arrays.fill(policy, 1.0 / regrets.length);
    } else {
      double sum = 0;
      for (int a = 0; a < regrets.length; a++) sum += policy[a] = Math.max(0, regrets[a]) / max;
      for (int a = 0; a < regrets.length; a++) policy[a] /= sum;
    }
    return policy;
  }

  public static double[] behavior(double[] policy, double epsilon) {
    validatePolicy(policy);
    if (!Double.isFinite(epsilon) || epsilon <= 0 || epsilon > 1) {
      throw new IllegalArgumentException("Outcome sampling epsilon must be in (0,1]");
    }
    double[] behavior = new double[policy.length];
    for (int a = 0; a < policy.length; a++) {
      behavior[a] = (1 - epsilon) * policy[a] + epsilon / policy.length;
      if (behavior[a] <= 0) throw new IllegalArgumentException("Exploration underflow");
    }
    return behavior;
  }

  public static int sample(double[] policy, RandomGenerator random) {
    validatePolicy(policy);
    double value = random.nextDouble();
    int lastPositive = -1;
    for (int a = 0; a < policy.length; a++) {
      if (policy[a] > 0) lastPositive = a;
      value -= policy[a];
      if (value < 0) return a;
    }
    return lastPositive;
  }

  public static void validatePolicy(double[] policy) {
    if (policy == null || policy.length == 0) throw new IllegalArgumentException("No actions");
    double sum = 0;
    for (double p : policy) {
      if (!Double.isFinite(p) || p < 0 || p > 1)
        throw new IllegalArgumentException("Invalid probability");
      sum += p;
    }
    if (Math.abs(sum - 1) > 1e-9) throw new IllegalArgumentException("Policy must sum to one");
  }
}
