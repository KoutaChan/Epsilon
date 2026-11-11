package com.epsilon.prism.hip;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import java.lang.ref.Reference;
import java.nio.ByteBuffer;

import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * カーネルの計時を GPU 1 枚で確かめる。時間が返るのは、最後に回したときに計時が入っていたカーネルだけで、ほかは NaN になる。計時を
 * 入れる・切る順がどうであっても、時間を読む呼び出しは失敗しない。
 */
public class HipPlanTimingTest {

  private static final int ROWS = 37;
  private static final int EVENTS = 100;

  private static final MlpCase CASE =
      MlpCase.random(1, TestPack.architecture(512, TestPack.FOUR_PLAYERS), HipKernel.MLP_ALL, 0, ROWS, EVENTS);

  private HipLibrary library;
  private PtStream stream;

  @BeforeClass(groups = "rocm")
  public void openStream() {
    library = HipLibrary.load();
    stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }

  @AfterClass(groups = "rocm", alwaysRun = true)
  public void closeStream() {
    if (stream != null) {
      stream.close();
    }
  }

  /** 計時を切ったまま回した後で計時を入れても、どのカーネルの時間も NaN になる。次に回せば、その回の時間が返る。 */
  @Test(groups = "rocm")
  public void runWithoutTimingReportsNoTime() {
    try (HipPlan plan = open()) {
      run(plan);
      plan.profile(true);
      assertNoTime(plan.kernelMillis());
      run(plan);
      assertTimed(plan.kernelMillis());
    }
  }

  /** 計時を入れて回した後に、計時を切って回し直す。その後で計時を入れても、前の回の時間は返らない。 */
  @Test(groups = "rocm")
  public void timeOfAnEarlierRunIsNotReported() {
    try (HipPlan plan = open()) {
      plan.profile(true);
      run(plan);
      assertTimed(plan.kernelMillis());
      plan.profile(false);
      run(plan);
      plan.profile(true);
      assertNoTime(plan.kernelMillis());
    }
  }

  /** 重みを写した計画。 */
  private HipPlan open() {
    HipPlan plan = HipPlan.create(library, stream, CASE.shape());
    plan.uploadWeights(CASE.pack(), 1);
    return plan;
  }

  /** 入力を写してカーネルを 1 回回し、終わるのを待つ。写し元は写し終わるまで生かす。 */
  private static void run(HipPlan plan) {
    ByteBuffer residual = CASE.input(HipRegion.RESIDUAL);
    ByteBuffer attention = CASE.input(HipRegion.ATTENTION);
    plan.copyIn(HipRegion.RESIDUAL, residual, CASE.bytes(HipRegion.RESIDUAL));
    plan.copyIn(HipRegion.ATTENTION, attention, CASE.bytes(HipRegion.ATTENTION));
    plan.run(CASE.kernel, CASE.layer, CASE.batch());
    plan.synchronize();
    Reference.reachabilityFence(residual);
    Reference.reachabilityFence(attention);
  }

  private static void assertNoTime(float[] millis) {
    for (HipKernel kernel : HipKernel.values()) {
      assertTrue(Float.isNaN(millis[kernel.ordinal()]), kernel + " reports " + millis[kernel.ordinal()] + " ms");
    }
  }

  /** 回したカーネルだけに時間がある。 */
  private static void assertTimed(float[] millis) {
    for (HipKernel kernel : HipKernel.values()) {
      float ms = millis[kernel.ordinal()];
      if (kernel == CASE.kernel) {
        assertTrue(ms >= 0 && Float.isFinite(ms), kernel + " reports " + ms + " ms");
      } else {
        assertTrue(Float.isNaN(ms), kernel + " reports " + ms + " ms");
      }
    }
  }
}
