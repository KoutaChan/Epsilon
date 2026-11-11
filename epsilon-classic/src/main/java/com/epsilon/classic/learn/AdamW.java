package com.epsilon.classic.learn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import java.util.List;

/**
 * AdamW（重み減衰を分離する形）の更新と、その状態（パラメーターごとの 1 次・2 次のモーメントとステップ数）。更新は DJL の Adam と同じで、
 * 同じ 1 回の呼び出し（{@code NDArrayEx.adamUpdate}）で行う。状態はこのクラスが持つので、学習を区切って続けるときに書き出して読み戻せる
 * （{@link PretrainState}）。係数は DJL の既定と同じ（β1 0.9・β2 0.999・ε 1e-8、勾配の倍率 1、勾配のクリップなし）。
 *
 * <p>モーメントはパラメーターと同じデバイスの FP32 で、パラメーターのマネージャーに付く。
 */
final class AdamW {

  static final float BETA1 = 0.9f;
  static final float BETA2 = 0.999f;
  static final float EPSILON = 1e-8f;

  private static final float RESCALE = 1f;
  private static final float NO_CLIP = -1f;

  private final List<NDArray> weights;
  private final NDArray[] means;
  private final NDArray[] variances;
  private final float learningRate;
  private final float weightDecay;
  private long steps;

  /**
   * モーメントとステップ数を 0 にして始める。
   *
   * @param weights 更新するパラメーターの配列（この順で状態を持つ）
   */
  AdamW(List<NDArray> weights, double learningRate, double weightDecay) {
    this.weights = List.copyOf(weights);
    this.learningRate = (float) learningRate;
    this.weightDecay = (float) weightDecay;
    means = new NDArray[weights.size()];
    variances = new NDArray[weights.size()];
    for (int index = 0; index < means.length; index++) {
      means[index] = weights.get(index).zerosLike();
      variances[index] = weights.get(index).zerosLike();
    }
  }

  /** 1 ステップ進める。パラメーターごとの勾配で、全てのパラメーターを更新する。勾配はこの呼び出しが 0 に戻す。 */
  void step(NDArray[] gradients) {
    steps++;
    double coef1 = 1 - Math.pow(BETA1, steps);
    double coef2 = 1 - Math.pow(BETA2, steps);
    float corrected = (float) (learningRate * Math.sqrt(coef2) / coef1);
    for (int index = 0; index < means.length; index++) {
      NDArray weight = weights.get(index);
      weight
          .getNDArrayInternal()
          .adamUpdate(
              new NDList(weight, gradients[index], means[index], variances[index]),
              new NDList(weight),
              learningRate,
              corrected,
              weightDecay,
              RESCALE,
              NO_CLIP,
              BETA1,
              BETA2,
              EPSILON,
              true,
              true);
    }
  }

  long steps() {
    return steps;
  }

  /** パラメーターの順に並べた 1 次のモーメント。 */
  float[][] means() {
    return values(means);
  }

  /** パラメーターの順に並べた 2 次のモーメント。 */
  float[][] variances() {
    return values(variances);
  }

  /** 書き出した状態に戻す。 */
  void restore(long steps, float[][] means, float[][] variances) {
    if (means.length != this.means.length || variances.length != this.variances.length) {
      throw new IllegalArgumentException(
          "The optimizer state holds " + means.length + " tensors, the network " + this.means.length);
    }
    this.steps = steps;
    for (int index = 0; index < means.length; index++) {
      this.means[index].set(means[index]);
      this.variances[index].set(variances[index]);
    }
  }

  private static float[][] values(NDArray[] arrays) {
    float[][] values = new float[arrays.length][];
    for (int index = 0; index < arrays.length; index++) {
      values[index] = arrays[index].toFloatArray();
    }
    return values;
  }
}
