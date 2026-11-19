package com.epsilon.prism.learn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.optimizer.AdamW;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

/** AdamW の分母に加える ε の扱い。 */
public enum AdamEpsilon {
  /** DJL の固定値 1e-8 を使う。 */
  FIXED,
  /** バイアス補正後の勾配の目盛りで ε が教師のゲインの 1e-8 倍になるようにする。 */
  TEMPO;

  Optimizer optimizer(Tracker rate, float gain) {
    var builder = Optimizer.adamW().optLearningRateTracker(rate).optWeightDecays(0f);
    return this == FIXED ? builder.build() : new TempoAdamW(builder, rate, gain);
  }

  /** DJL のモーメントと歩数を使い、補正前の √v に渡す ε だけを毎歩変える。 */
  private static final class TempoAdamW extends AdamW {
    private static final float BETA1 = 0.9f, BETA2 = 0.999f;
    private final Tracker rate;
    private final float gain;

    TempoAdamW(Builder builder, Tracker rate, float gain) {
      super(builder);
      this.rate = rate;
      this.gain = gain;
    }

    @Override
    public void update(String parameterId, NDArray weight, NDArray grad) {
      int t = updateCount(parameterId);
      double root = Math.sqrt(1 - Math.pow(BETA2, t));
      float lr = rate.getNewValue(t);
      float corrected = (float) (lr * root / (1 - Math.pow(BETA1, t)));
      float epsilon = (float) (1e-8 * gain * root);
      NDArray mean =
          withDefaultState(
              getState("means"), parameterId, weight.getDevice(), k -> weight.zerosLike());
      NDArray variance =
          withDefaultState(
              getState("variances"), parameterId, weight.getDevice(), k -> weight.zerosLike());
      weight
          .getNDArrayInternal()
          .adamUpdate(
              new NDList(weight, grad, mean, variance),
              new NDList(weight),
              lr,
              corrected,
              getWeightDecay(),
              rescaleGrad,
              clipGrad,
              BETA1,
              BETA2,
              epsilon,
              true,
              true);
    }
  }
}
