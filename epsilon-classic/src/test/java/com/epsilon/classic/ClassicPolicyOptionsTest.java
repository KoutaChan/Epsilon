package com.epsilon.classic;

import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Path;
import org.testng.annotations.Test;

/**
 * classic の系列には専用のカーネルが無い。方策を開く設定で専用のカーネルを選ぶと、チェックポイントを読む前に拒むことを確かめる。
 */
public class ClassicPolicyOptionsTest {

  @Test
  public void nativeKernelsAreRefusedBeforeTheCheckpointIsRead() {
    PolicyOptions nativeKernels = new PolicyOptions("cpu", Precision.BFLOAT16, 8, Kernels.NATIVE);
    for (Architecture architecture : Architecture.values()) {
      IllegalArgumentException refused =
          expectThrows(
              IllegalArgumentException.class,
              () -> ModelFamily.of(architecture.series()).open(Path.of("no-such-checkpoint"), Variant.TENHOU_4P, nativeKernels));
      assertTrue(refused.getMessage().contains("has no native kernels"), refused.getMessage());
    }
  }
}
