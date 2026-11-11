package com.epsilon.reviewer.model;

import com.epsilon.reviewer.engine.format.ModelInfo;

/** モデルの 1 つのバージョン（モデルの名前と revision の組）を指すキー。 */
public record ModelKey(String modelId, String revision) {
  public static ModelKey of(ModelInfo info) {
    return new ModelKey(info.modelId(), info.revision());
  }
}
