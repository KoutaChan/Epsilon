package com.epsilon.prism.kernel;

import java.util.Objects;

/**
 * バックボーンの 1 層。
 *
 * @param role 層の種類（全トークンの層か、局・席・牌の層か）
 * @param ffn FFN の幅
 */
public record EncoderLayerShape(LayerRole role, FeedForwardShape ffn) {

  /**
   * 層の種類と FFN が null でないことを確かめる。
   *
   * @throws NullPointerException どちらかが null のとき
   */
  public EncoderLayerShape {
    Objects.requireNonNull(role, "role");
    Objects.requireNonNull(ffn, "ffn");
  }
}
