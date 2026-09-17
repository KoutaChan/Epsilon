package com.epsilon.nano.ai.decision.runtime;

import ai.djl.engine.Autocast;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.epsilon.config.settings.DecisionComputePrecision;

/** Decision の順伝播で使う自動混合精度の設定を、自己対局学習と牌譜学習で共有する。 */
public final class EpsilonDecisionAutocast {

  private EpsilonDecisionAutocast() {}

  /** 指定精度の自動混合精度有効範囲を開く。FLOAT32では何もしない有効範囲を返す。 */
  public static Autocast open(NDManager manager, DecisionComputePrecision precision) {
    return switch (precision) {
      case FLOAT32 -> NoOpAutocast.INSTANCE;
      case FLOAT16 -> Engine.getInstance().newAutocast(manager.getDevice(), DataType.FLOAT16, true);
      case BFLOAT16 ->
          Engine.getInstance().newAutocast(manager.getDevice(), DataType.BFLOAT16, true);
    };
  }

  private enum NoOpAutocast implements Autocast {
    INSTANCE;

    @Override
    public void close() {}
  }
}
