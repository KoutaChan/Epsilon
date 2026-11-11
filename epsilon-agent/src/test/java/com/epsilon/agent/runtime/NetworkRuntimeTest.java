package com.epsilon.agent.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicIntegerArray;
import org.testng.annotations.Test;

/** デバイス名の読み方と、GPU を使わないランタイムの動き（符号化のスレッドプール、受付の集計のログ、閉じ方）を確かめる。 */
public class NetworkRuntimeTest {

  @Test
  public void devicesAreOneCpuOrDistinctGpus() {
    assertEquals(NetworkDevices.parse("cpu").devices(), List.of(Device.cpu()));
    NetworkDevices gpus = NetworkDevices.parse("gpu1, gpu0");
    assertEquals(gpus.devices(), List.of(Device.gpu(1), Device.gpu(0)));
    assertEquals(gpus.primary(), Device.gpu(1));
    for (String names : List.of("cpu,gpu0", "gpu0,gpu0", "cpu,cpu", "gpu", "gpu01", "gpu:0", "", "gpu0,")) {
      expectThrows(IllegalArgumentException.class, () -> NetworkDevices.parse(names));
    }
    expectThrows(IllegalArgumentException.class, () -> new NetworkDevices(List.of()));
    assertEquals(NetworkDevices.of(Device.cpu(), Device.cpu()).size(), 2);
  }

  @Test
  public void encodingCoversEveryRowOnce() throws Exception {
    try (NetworkRuntime runtime = new NetworkRuntime(3, 2, 4)) {
      for (int rows : new int[] {1, 127, 128, 1000}) {
        AtomicIntegerArray written = new AtomicIntegerArray(rows);
        runtime
            .encodeRows(
                rows,
                (from, to) -> {
                  for (int row = from; row < to; row++) {
                    written.incrementAndGet(row);
                  }
                })
            .get(5, TimeUnit.SECONDS);
        for (int row = 0; row < rows; row++) {
          assertEquals(written.get(row), 1, "row " + row + " of " + rows);
        }
      }
    }
  }

  @Test
  public void aCpuRuntimeLogsAndClosesWithoutGpuResources() {
    NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
    expectThrows(IllegalArgumentException.class, () -> runtime.streams(Device.cpu()));
    var admission =
        new InferenceAdmission<String, String, Integer>(
            1,
            (owner, key) -> 1,
            (owner, key) -> DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(owner)),
            owner -> true);
    admission.add("model", "shape", 1, 0);
    admission.encodingFailed(admission.admit(0, false));
    runtime.awaitInferenceIdle();
    runtime.logInference(admission.metrics());
    runtime.resetInferenceMetrics();
    runtime.awaitIdle();
    runtime.close();
    runtime.close();
    expectThrows(IllegalStateException.class, () -> runtime.encodeRows(1, (from, to) -> {}));
    expectThrows(IllegalStateException.class, () -> runtime.streams(Device.gpu(0)));
  }

  @Test
  public void anInferenceRuntimeReportsItsCapacities() {
    try (NetworkRuntime runtime = new NetworkRuntime(3, 2, 4)) {
      assertEquals(runtime.slots(), 2);
      assertEquals(runtime.readyBatches(), 4);
    }
  }

  /** 学習だけに使うランタイムは、符号化も推論のパイプラインも持たない。することは、ストリームの貸し出しと待ちだけ。 */
  @Test
  public void aLearningRuntimeRefusesInference() {
    NetworkRuntime runtime = NetworkRuntime.forLearning();
    expectThrows(IllegalStateException.class, () -> runtime.encodeRows(1, (from, to) -> {}));
    expectThrows(IllegalStateException.class, () -> runtime.lease(Device.gpu(0)));
    expectThrows(IllegalStateException.class, runtime::slots);
    expectThrows(IllegalStateException.class, runtime::readyBatches);
    expectThrows(IllegalArgumentException.class, () -> runtime.streams(Device.cpu()));
    runtime.awaitIdle();
    runtime.close();
    expectThrows(IllegalStateException.class, () -> runtime.streams(Device.gpu(0)));
  }

  @Test
  public void capacitiesAreChecked() {
    expectThrows(IllegalArgumentException.class, () -> new NetworkRuntime(0, 1, 1));
    expectThrows(IllegalArgumentException.class, () -> new NetworkRuntime(1, 0, 1));
    expectThrows(IllegalArgumentException.class, () -> new NetworkRuntime(1, 5, 5));
    expectThrows(IllegalArgumentException.class, () -> new NetworkRuntime(1, 2, 1));
  }
}
