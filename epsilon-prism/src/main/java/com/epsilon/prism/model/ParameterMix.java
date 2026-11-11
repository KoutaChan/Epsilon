package com.epsilon.prism.model;

import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;

/** 同じ形の 2 つのネットワークの重みを混ぜる（方策の記憶に使う。{@link PrismNetwork#mixIn}）。 */
public final class ParameterMix {
  private ParameterMix() {}

  /**
   * 混ぜ先の重みに、混ぜ元の重みを割合 η で混ぜる。要素ごとに FP32 で w ← (1 − η)·w + η·w_混ぜ元 とし、η = 1 なら混ぜ元をそのまま写す。
   * パラメーターは名前で突き合わせる。
   *
   * @param target 混ぜ先のパラメーター（書き換える）
   * @param source 混ぜ元のパラメーター
   * @param eta 混ぜ元の割合 η
   * @throws IllegalArgumentException η が (0, 1] の外のとき、またはパラメーターの名前か形が 2 つのネットワークで違うとき
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
