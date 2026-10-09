package com.epsilon.cfr.collection;

import com.epsilon.cfr.data.ObservationPayload;
import java.util.Objects;

/** 探索を加える前の目標方策と、本人が観測できる情報だけを符号化した保存用のモデル入力。 */
public record Prediction(double[] targetPolicy, ObservationPayload observationPayload) {
  public Prediction {
    targetPolicy = targetPolicy.clone();
    Objects.requireNonNull(observationPayload, "observationPayload");
  }

  @Override
  public double[] targetPolicy() {
    return targetPolicy.clone();
  }
}
