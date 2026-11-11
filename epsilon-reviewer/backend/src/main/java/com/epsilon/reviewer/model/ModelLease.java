package com.epsilon.reviewer.model;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.review.ReviewModel;

/** 開いたモデルの貸し出し。評価は、モデルごとの公平なロックで 1 つずつ行う。閉じると、モデルをレジストリに返す。 */
public final class ModelLease implements ReviewModel, AutoCloseable {
  private final ModelRegistry registry;
  private final ModelRegistry.OpenModel model;
  private boolean closed;

  ModelLease(ModelRegistry registry, ModelRegistry.OpenModel model) {
    this.registry = registry;
    this.model = model;
  }

  public ModelInfo info() {
    return model.entry.info();
  }

  /** 価値を出すのは PRISM の系列だけ。ほかの系列の評価は確率だけを持つ。 */
  @Override
  public boolean valued() {
    return model.entry.series() == Series.PRISM;
  }

  /**
   * 表示の確率を作る温度。models.json で上書きしていればその値を、無ければ方策の温度を返す。
   *
   * @throws IllegalStateException 上書きが無く、方策も温度を持たないとき（価値を出さない系列では呼ばない）
   */
  @Override
  public double temperature() {
    Double written = model.entry.temperature();
    if (written != null) return written;
    return model
        .policy
        .resultNow()
        .temperature()
        .orElseThrow(
            () ->
                new IllegalStateException(
                    "Model " + model.entry.key().modelId() + " has no display temperature."));
  }

  @Override
  public Variant variant() {
    return model.entry.variant();
  }

  @Override
  public Evaluation evaluate(Observation observation) {
    model.evaluation.lock();
    try {
      return model.policy.resultNow().evaluate(observation);
    } finally {
      model.evaluation.unlock();
    }
  }

  @Override
  public void close() {
    if (closed) return;
    closed = true;
    registry.release(model);
  }
}
