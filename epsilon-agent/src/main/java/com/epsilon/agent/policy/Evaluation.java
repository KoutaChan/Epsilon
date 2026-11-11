package com.epsilon.agent.policy;

import java.util.Objects;

/**
 * 1 つの判断の評価。配列は合法手の順に並べる。作った側は配列の所有権ごと渡し、渡した後は書き換えない。
 *
 * @param probabilities 手ごとの確率（0〜1）。和は 1 で、ずれてよい幅は {@link #SUM_TOLERANCE}
 * @param values 手ごとの価値（効用の単位で、有限の値）。価値を出さない系列では null
 * @param greedy その系列が貪欲に選ぶ手（合法手の何番目か）。確率が最大の手とは限らない
 */
public record Evaluation(float[] probabilities, float[] values, int greedy) {

  /** 確率の和が 1 からずれてよい幅。 */
  public static final double SUM_TOLERANCE = 1e-4;

  public Evaluation {
    Objects.requireNonNull(probabilities, "probabilities");
    if (probabilities.length == 0) {
      throw new IllegalArgumentException("No legal actions");
    }
    double sum = 0;
    for (float probability : probabilities) {
      if (!(probability >= 0 && probability <= 1)) {
        throw new IllegalArgumentException("Invalid probability: " + probability);
      }
      sum += probability;
    }
    if (Math.abs(sum - 1) > SUM_TOLERANCE) {
      throw new IllegalArgumentException("Probabilities sum to " + sum);
    }
    if (values != null) {
      if (values.length != probabilities.length) {
        throw new IllegalArgumentException(values.length + " values for " + probabilities.length + " probabilities");
      }
      for (float value : values) {
        if (!Float.isFinite(value)) {
          throw new IllegalArgumentException("Invalid value: " + value);
        }
      }
    }
    if (greedy < 0 || greedy >= probabilities.length) {
      throw new IllegalArgumentException("Greedy action " + greedy + " is not among " + probabilities.length);
    }
  }
}
