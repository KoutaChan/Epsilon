package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Key;
import java.util.Objects;

/**
 * 帳簿の係数を当てはめるときの重みと、係数の縮め先のセクション。
 *
 * @param fit 帳簿の係数を当てはめるときの重み
 * @param shrink 係数の縮め先
 */
public record LossWeightSettings(
    @Key(doc = "帳簿の係数を当てはめるときの重み", value = "UNIFORM") Weighting fit,
    @Key(doc = "係数の縮め先", value = "ZERO") Shrink shrink) {

  /** 帳簿の係数を当てはめるときの重み。 */
  public enum Weighting {
    /** 行の中で一様に中心化した最小二乗。 */
    UNIFORM,
    /** π = softmax(s/λ) で中心化した、重みつきの最小二乗。 */
    POLICY
  }

  /** 係数の縮め先。 */
  public enum Shrink {
    /** 0 へ縮める。 */
    ZERO,
    /** 前のコホートの同じ区分の係数へ縮める。 */
    PREVIOUS_COHORT
  }

  public LossWeightSettings {
    Objects.requireNonNull(fit, "fit");
    Objects.requireNonNull(shrink, "shrink");
  }
}
