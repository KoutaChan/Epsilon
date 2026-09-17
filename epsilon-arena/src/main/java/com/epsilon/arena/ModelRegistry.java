package com.epsilon.arena;

import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.ModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 同じ系列・実パス・設定のモデルは一度だけ開く。動的クラス探索はしない。 */
public final class ModelRegistry implements AutoCloseable {
  private final Map<String, ModelProvider> providers = new LinkedHashMap<>();
  private final Map<ModelSpec, BatchedPolicy> opened = new LinkedHashMap<>();
  private final PolicyExecutionContext execution;

  public ModelRegistry(List<ModelProvider> providers, PolicyExecutionContext execution) {
    this.execution = execution;
    for (ModelProvider provider : providers) {
      if (this.providers.put(provider.seriesId(), provider) != null)
        throw new IllegalArgumentException("Duplicate series: " + provider.seriesId());
    }
  }

  public Participant open(ModelSpec requested) throws IOException {
    ModelProvider provider = providers.get(requested.series());
    if (provider == null)
      throw new IllegalArgumentException("Unknown series: " + requested.series());
    Map<String, String> options = new LinkedHashMap<>(requested.options());
    if (options.containsKey("settings"))
      options.put("settings", Path.of(options.get("settings")).toRealPath().toString());
    ModelSpec key = new ModelSpec(requested.series(), requested.checkpoint().toRealPath(), options);
    BatchedPolicy policy = opened.get(key);
    if (policy == null) {
      policy = provider.open(key.checkpoint(), key.options(), execution);
      opened.put(key, policy);
    }
    return new Participant(key.series() + " / " + key.checkpoint(), policy);
  }

  @Override
  public void close() {
    Throwable failure = null;
    List<BatchedPolicy> policies = new ArrayList<>(opened.values());
    opened.clear();
    for (int index = policies.size() - 1; index >= 0; index--) {
      try {
        policies.get(index).close();
      } catch (RuntimeException | Error error) {
        if (failure == null) failure = error;
        else failure.addSuppressed(error);
      }
    }
    if (failure instanceof RuntimeException runtime) throw runtime;
    if (failure instanceof Error error) throw error;
  }
}
