package com.epsilon.prism.model;

import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;
import com.epsilon.prism.run.Averaging;

/**
 * 同じ形の網の重みの混ぜ方（方策の記憶 {@link Averaging}。{@link
 * PrismNetwork#mixIn}）。
 */
public final class ParameterMix {
  private ParameterMix() {}

  /**
   * target の重みを source の重みと混ぜる: w ← (1 − eta)·w + eta·w_source（要素ごと、FP32）。eta = 1 なら source の写し。パラメーターは名前で
   * 突き合わせる。
   *
   * @throws IllegalArgumentException eta が (0, 1] の外か、パラメーターの名前か形が違う
   */
  public static void mix(
      PairList<String, Parameter> target, PairList<String, Parameter> source, double eta) {
    if (!(eta > 0 && eta <= 1)) throw new IllegalArgumentException("eta must be in (0, 1]: " + eta);
    if (source.size() != target.size())
      throw new IllegalArgumentException("Models have different parameter lists");
    for (int i = 0; i < target.size(); i++) {
      String name = target.keyAt(i);
      var destination = target.valueAt(i).getArray();
      Parameter parameter = source.get(name);
      if (parameter == null || !parameter.getArray().getShape().equals(destination.getShape()))
        throw new IllegalArgumentException("Parameter " + name + " differs between models");
      float[] values = parameter.getArray().toType(DataType.FLOAT32, false).toFloatArray();
      if (eta < 1) {
        float[] own = destination.toType(DataType.FLOAT32, false).toFloatArray();
        for (int k = 0; k < values.length; k++)
          values[k] = (float) ((1 - eta) * own[k] + eta * values[k]);
      }
      destination.set(values);
    }
  }
}
