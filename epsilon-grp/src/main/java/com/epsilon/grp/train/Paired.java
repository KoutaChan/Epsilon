package com.epsilon.grp.train;

/** 平均（対応のある差、または残差）と、半荘をクラスタとして求めた標準誤差。半荘が 2 つ未満なら標準誤差は NaN。 */
public record Paired(double mean, double standardError) {

  public double z() {
    return mean / standardError;
  }
}
