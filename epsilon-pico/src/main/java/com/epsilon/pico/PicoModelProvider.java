package com.epsilon.pico;

import com.epsilon.pico.ai.decision.runtime.NativePolicy;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.ModelProvider;
import com.epsilon.spi.PolicyExecutionContext;
import java.nio.file.Path;
import java.util.Map;

/** epsilon-pico のチェックポイントを読み込み、この系列の入力形式で推論を実行する。 */
public final class PicoModelProvider implements ModelProvider {
  @Override
  public String seriesId() {
    return "epsilon-pico";
  }

  @Override
  public BatchedPolicy open(
      Path checkpoint, Map<String, String> options, PolicyExecutionContext execution) {
    return NativePolicy.open(checkpoint, options, execution);
  }
}
