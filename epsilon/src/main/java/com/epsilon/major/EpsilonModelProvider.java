package com.epsilon.major;

import com.epsilon.major.ai.decision.runtime.NativePolicy;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.ModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.file.Path;
import java.util.Map;

/** epsilon のチェックポイントを読み込み、この系列の入力形式で推論を実行する。 */
public final class EpsilonModelProvider implements ModelProvider {
  @Override
  public String seriesId() {
    return "epsilon";
  }

  @Override
  public BatchedPolicy open(
      Path checkpoint, Map<String, String> options, PolicyExecutionContext execution) {
    return NativePolicy.open(checkpoint, options, execution);
  }
}
