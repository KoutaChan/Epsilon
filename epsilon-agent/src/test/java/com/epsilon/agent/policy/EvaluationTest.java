package com.epsilon.agent.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import org.testng.annotations.Test;

/** 評価と方策の設定は、作るときに値を確かめる。確率の和が 1 でない、値が有限でない、デバイス名が cpu・gpuN でないといった値は例外にする。 */
public class EvaluationTest {

  @Test
  public void acceptsProbabilitiesThatSumToOne() {
    Evaluation policyOnly = new Evaluation(new float[] {0.25f, 0.75f}, null, 0);
    assertNull(policyOnly.values());
    assertEquals(policyOnly.greedy(), 0);
    Evaluation valued = new Evaluation(new float[] {0.5f, 0.5f}, new float[] {-0.25f, 0.5f}, 1);
    assertEquals(valued.values().length, 2);
  }

  @Test
  public void rejectsBrokenProbabilities() {
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(new float[0], null, 0));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(new float[] {0.5f, 0.4f}, null, 0));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(new float[] {1.5f, -0.5f}, null, 0));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(new float[] {Float.NaN, 1f}, null, 0));
  }

  @Test
  public void rejectsBrokenValuesAndGreedyActions() {
    float[] probabilities = {0.5f, 0.5f};
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(probabilities, new float[] {0f}, 0));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(probabilities, new float[] {0f, Float.NaN}, 0));
    expectThrows(
        IllegalArgumentException.class,
        () -> new Evaluation(probabilities, new float[] {0f, Float.POSITIVE_INFINITY}, 0));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(probabilities, null, 2));
    expectThrows(IllegalArgumentException.class, () -> new Evaluation(probabilities, null, -1));
  }

  @Test
  public void optionsNameACpuOrAnIndexedGpu() {
    assertEquals(new PolicyOptions("gpu1", Precision.BFLOAT16, 2048, Kernels.NATIVE).device(), "gpu1");
    assertEquals(PolicyOptions.defaults(), new PolicyOptions("cpu", Precision.FLOAT32, 8, Kernels.EAGER));
    assertEquals(PolicyOptions.defaults().onDevice("gpu0").device(), "gpu0");
    expectThrows(IllegalArgumentException.class, () -> new PolicyOptions("gpu:0", Precision.FLOAT32, 8, Kernels.EAGER));
    expectThrows(IllegalArgumentException.class, () -> new PolicyOptions("gpu", Precision.FLOAT32, 8, Kernels.EAGER));
    expectThrows(IllegalArgumentException.class, () -> new PolicyOptions("cpu", Precision.FLOAT32, 0, Kernels.EAGER));
    expectThrows(NullPointerException.class, () -> new PolicyOptions("cpu", null, 8, Kernels.EAGER));
  }
}
