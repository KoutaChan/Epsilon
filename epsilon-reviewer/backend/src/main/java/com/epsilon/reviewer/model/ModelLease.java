package com.epsilon.reviewer.model;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.spi.ReviewEvaluation;
import java.util.List;

/** 開いたモデルの貸し出し。評価はモデルごとの公平な鍵で 1 つずつ行う。閉じると返す。 */
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

  @Override
  public double temperature() {
    return model.entry.temperature();
  }

  @Override
  public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legal) {
    model.evaluation.lock();
    try {
      return model.policy.resultNow().evaluate(observation, legal);
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
