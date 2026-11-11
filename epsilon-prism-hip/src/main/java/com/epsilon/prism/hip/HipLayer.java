package com.epsilon.prism.hip;

import java.util.Objects;

/**
 * バックボーンの 1 層。
 *
 * @param role 層の種類（全トークンの層か、局・席・牌の層か）
 * @param ffn FFN の幅
 */
public record HipLayer(LayerRole role, HipFfn ffn) {

  /**
   * 層の種類と FFN が null でないことを確かめる。
   *
   * @throws NullPointerException どちらかが null のとき
   */
  public HipLayer {
    Objects.requireNonNull(role, "role");
    Objects.requireNonNull(ffn, "ffn");
  }
}
