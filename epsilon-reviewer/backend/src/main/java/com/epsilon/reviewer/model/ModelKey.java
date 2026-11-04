package com.epsilon.reviewer.model;

import com.epsilon.reviewer.engine.format.ModelInfo;

/** モデルの版を指す鍵。 */
public record ModelKey(String modelId, String revision) {
  public static ModelKey of(ModelInfo info) {
    return new ModelKey(info.modelId(), info.revision());
  }
}
