package com.epsilon.spi;

import java.util.Objects;

/**
 * 牌譜解析の 1 判断の評価。合法手の順に、表示用の確率（必須）と手の価値（価値を推定するモデルだけ）を持つ。
 *
 * <p>確率は全合法手で正規化した 0..1。価値はモデルの目的の効用の単位（大きいほど良い）で、比べられるのは同じ判断の手どうしの差だけ。
 * 配列は呼び出し元へ所有権ごと渡し、作成後は変更しない。
 *
 * @param probabilities 合法手ごとの確率（合計 1、許容 {@value #SUM_TOLERANCE}）
 * @param values 合法手ごとの価値（有限）。価値を推定しないモデルは null
 */
public record ReviewEvaluation(float[] probabilities, float[] values) {
  /** 確率の合計と 1 の差の許容。 */
  public static final double SUM_TOLERANCE = 1e-4;

  /**
   * 価値が効用の単位のモデルの表示用の温度の既定。画面の「ほぼ同等」の境目（損失 0.002）と同じで、0.002 劣る手は最善手の 1/e 倍、0.005 劣る手
   * （悪手の境目）は約 1/12 倍の推奨度になる。
   */
  public static final double DISPLAY_TEMPERATURE = 0.002;

  public ReviewEvaluation {
    Objects.requireNonNull(probabilities, "probabilities");
    if (probabilities.length == 0) throw new IllegalArgumentException("No legal actions.");
    double sum = 0;
    for (float probability : probabilities) {
      if (!Float.isFinite(probability) || probability < 0 || probability > 1)
        throw new IllegalArgumentException("Invalid probability: " + probability);
      sum += probability;
    }
    if (Math.abs(sum - 1) > SUM_TOLERANCE)
      throw new IllegalArgumentException("Probabilities do not sum to one: " + sum);
    if (values != null) {
      if (values.length != probabilities.length)
        throw new IllegalArgumentException(
            values.length + " values for " + probabilities.length + " probabilities.");
      for (float value : values)
        if (!Float.isFinite(value)) throw new IllegalArgumentException("Invalid value: " + value);
    }
  }

  /** 方策の確率だけを出すモデルの評価。 */
  public static ReviewEvaluation policy(float[] probabilities) {
    return new ReviewEvaluation(probabilities, null);
  }

  /** 手の価値と、それから作った表示用の確率を出すモデルの評価。 */
  public static ReviewEvaluation valued(float[] values, float[] probabilities) {
    return new ReviewEvaluation(probabilities, Objects.requireNonNull(values, "values"));
  }

  /**
   * 価値から表示用の確率 softmax(value/temperature) を作る（全合法手）。
   *
   * @param temperature 表示用の温度（正）
   */
  public static float[] softmax(float[] values, double temperature) {
    if (!(temperature > 0)) throw new IllegalArgumentException("temperature must be positive");
    double max = Double.NEGATIVE_INFINITY;
    for (float value : values) max = Math.max(max, value);
    double[] weights = new double[values.length];
    double total = 0;
    for (int a = 0; a < values.length; a++) {
      weights[a] = Math.exp((values[a] - max) / temperature);
      total += weights[a];
    }
    float[] probabilities = new float[values.length];
    for (int a = 0; a < values.length; a++) probabilities[a] = (float) (weights[a] / total);
    return probabilities;
  }

  /** 合法手の数。 */
  public int actions() {
    return probabilities.length;
  }

  /** 価値を持つか。 */
  public boolean valued() {
    return values != null;
  }

  /**
   * 合法手ごとの損失 = 最大の価値 − その手の価値（0 以上、最善の手は 0）。新しい配列を返す。
   *
   * @throws IllegalStateException 価値を持たない評価
   */
  public float[] losses() {
    if (values == null) throw new IllegalStateException("The evaluation has no values.");
    float best = Float.NEGATIVE_INFINITY;
    for (float value : values) best = Math.max(best, value);
    float[] losses = new float[values.length];
    for (int a = 0; a < values.length; a++) losses[a] = best - values[a];
    return losses;
  }
}
