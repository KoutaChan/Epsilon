package com.epsilon.network;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import com.epsilon.config.settings.DeviceSettings;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.annotations.Test;

public class NetworkDeviceResolverTest {
  @Test
  public void explicitDevicesPreserveRoleOrderAndDuplicatesWithoutNativeDetection() {
    DeviceSettings settings = new DeviceSettings(false, "CPU, cpu", "gpu:2, CUDA:0,1,gpu2", "cpu");
    assertEquals(
        NetworkDeviceResolver.learner(settings), NetworkDevices.of(Device.cpu(), Device.cpu()));
    assertEquals(
        NetworkDeviceResolver.inference(settings),
        NetworkDevices.of(Device.gpu(2), Device.gpu(0), Device.gpu(1), Device.gpu(2)));
    assertEquals(NetworkDeviceResolver.grp(settings), NetworkDevices.of(Device.cpu()));
    assertEquals(
        NetworkDeviceResolver.resolve(
            "learner",
            " cpu ",
            true,
            () -> {
              throw new AssertionError("CPU must not detect GPUs");
            }),
        NetworkDevices.of(Device.cpu()));
    for (String invalid : new String[] {"cpu,", ",cpu", "gpu:-1", "cuda:x", "gpu", "1,,2"}) {
      IllegalArgumentException error =
          expectThrows(
              IllegalArgumentException.class,
              () -> NetworkDeviceResolver.resolve("learner", invalid, false));
      assertTrue(error.getMessage().contains("learner"));
    }
  }

  @Test
  public void autoAndVisibleIndicesUseOneDetectionPerResolution() {
    AtomicInteger detections = new AtomicInteger();
    assertEquals(
        NetworkDeviceResolver.resolve(
            "inference",
            "gpu1,cpu,0",
            true,
            () -> {
              detections.incrementAndGet();
              return 2;
            }),
        NetworkDevices.of(Device.gpu(1), Device.cpu(), Device.gpu(0)));
    assertEquals(detections.get(), 1);
    for (String spec : new String[] {"", " Auto ", "all"}) {
      assertEquals(
          NetworkDeviceResolver.resolve("inference", spec, false, () -> 2),
          NetworkDevices.of(Device.gpu(0), Device.gpu(1)));
      assertEquals(
          NetworkDeviceResolver.resolve("inference", spec, false, () -> 0),
          NetworkDevices.of(Device.cpu()));
    }
    assertTrue(
        expectThrows(
                IllegalArgumentException.class,
                () -> NetworkDeviceResolver.resolve("grp", "gpu:0", true, () -> 0))
            .getMessage()
            .contains("no GPU"));
    assertTrue(
        expectThrows(
                IllegalArgumentException.class,
                () -> NetworkDeviceResolver.resolve("grp", "gpu:2", true, () -> 2))
            .getMessage()
            .contains("outside visible range"));
  }
}
