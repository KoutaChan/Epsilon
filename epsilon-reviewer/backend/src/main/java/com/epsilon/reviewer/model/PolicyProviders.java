package com.epsilon.reviewer.model;

import com.epsilon.spi.ReviewPolicyProvider;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceLoader;

/** 系列 ID → 方策の提供元。クラスパスの ServiceLoader から 1 回だけ集める。 */
public final class PolicyProviders {
  private final Map<String, ReviewPolicyProvider> bySeries;

  private PolicyProviders(Map<String, ReviewPolicyProvider> bySeries) {
    this.bySeries = bySeries;
  }

  /** @throws IllegalStateException 同じ系列の提供元が 2 つある */
  public static PolicyProviders load() {
    Map<String, ReviewPolicyProvider> found = new HashMap<>();
    for (ReviewPolicyProvider provider : ServiceLoader.load(ReviewPolicyProvider.class))
      if (found.putIfAbsent(provider.seriesId(), provider) != null)
        throw new IllegalStateException("Duplicate review policy series: " + provider.seriesId());
    return new PolicyProviders(Map.copyOf(found));
  }

  public Optional<ReviewPolicyProvider> find(String series) {
    return Optional.ofNullable(bySeries.get(series));
  }
}
