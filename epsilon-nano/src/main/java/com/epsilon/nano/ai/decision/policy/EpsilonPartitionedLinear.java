package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;

/** 一つのLinearの入力行列を列方向へ分割し、ブロードキャスト前の構成要素を個別に射影する。 */
final class EpsilonPartitionedLinear {

  private EpsilonPartitionedLinear() {}

  /**
   * {@code concat(components) * W'} を、各構成要素と対応する重み部分ビューの積の和として計算する。
   *
   * <p>候補間で共通な状態や節点表現を小さい形状のまま射影するため、ブロードキャスト後の同じ行列積を候補数だけ繰り返さない。 パラメーターと勾配経路は元のLinearと同一である。
   */
  static NDArray apply(
      Linear linear, ParameterStore parameterStore, boolean training, NDArray... components) {
    Parameter weightParameter = linear.getDirectParameters().get("weight");
    Parameter biasParameter = linear.getDirectParameters().get("bias");
    NDArray weight = parameterStore.getValue(weightParameter, components[0].getDevice(), training);
    long[] columnStarts = new long[components.length];
    long columnStart = 0;
    int accumulatorIndex = 0;
    for (int componentIndex = 0; componentIndex < components.length; componentIndex++) {
      columnStarts[componentIndex] = columnStart;
      NDArray component = components[componentIndex];
      long width = component.getShape().get(component.getShape().dimension() - 1);
      columnStart += width;
      if (component.getShape().size() > components[accumulatorIndex].getShape().size()) {
        accumulatorIndex = componentIndex;
      }
    }

    NDArray accumulator = components[accumulatorIndex];
    long accumulatorWidth = accumulator.getShape().get(accumulator.getShape().dimension() - 1);
    NDArray result =
        Linear.linear(
                accumulator,
                weight.get(
                    ":,{}:{}",
                    columnStarts[accumulatorIndex],
                    columnStarts[accumulatorIndex] + accumulatorWidth),
                parameterStore.getValue(biasParameter, accumulator.getDevice(), training))
            .singletonOrThrow();
    for (int componentIndex = 0; componentIndex < components.length; componentIndex++) {
      if (componentIndex == accumulatorIndex) {
        continue;
      }
      NDArray component = components[componentIndex];
      long width = component.getShape().get(component.getShape().dimension() - 1);
      result.addi(
          Linear.linear(
                  component,
                  weight.get(
                      ":,{}:{}",
                      columnStarts[componentIndex],
                      columnStarts[componentIndex] + width),
                  null)
              .singletonOrThrow());
    }
    return result;
  }
}
