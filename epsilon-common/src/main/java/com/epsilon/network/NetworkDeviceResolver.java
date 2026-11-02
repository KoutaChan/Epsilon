package com.epsilon.network;

import ai.djl.Device;
import ai.djl.engine.Engine;
import com.epsilon.config.settings.DeviceSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.IntSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 全モデルで共通のデバイス指定、自動選択、可視 GPU インデックス検証。 */
public final class NetworkDeviceResolver {
  private static final Logger log = LoggerFactory.getLogger(NetworkDeviceResolver.class);

  private NetworkDeviceResolver() {}

  public static NetworkDevices learner(DeviceSettings settings) {
    return resolve("learner", settings.learner(), settings.validateIndices());
  }

  public static NetworkDevices inference(DeviceSettings settings) {
    return resolve("inference", settings.inference(), settings.validateIndices());
  }

  public static NetworkDevices grp(DeviceSettings settings) {
    return resolve("grp", settings.grp(), settings.validateIndices());
  }

  /** GPU を番号順に選択し、GPU がなければ CPU を使う。maxGpus が 0 以下なら全 GPU。 */
  public static NetworkDevices preferred(int maxGpus) {
    return preferred(maxGpus, detectedGpuCount());
  }

  /** 空欄・auto・all は自動選択。それ以外は指定順と重複を保持する。 */
  public static NetworkDevices resolve(String role, String deviceSpec, boolean validateGpuIndices) {
    return resolve(role, deviceSpec, validateGpuIndices, NetworkDeviceResolver::detectedGpuCount);
  }

  private static int detectedGpuCount() {
    NetworkRuntime.initialize();
    return Engine.getInstance().getGpuCount();
  }

  static NetworkDevices resolve(
      String role, String deviceSpec, boolean validateGpuIndices, IntSupplier detectedGpuCount) {
    Objects.requireNonNull(role, "role");
    String spec = Objects.requireNonNull(deviceSpec, "deviceSpec").trim();
    if (spec.isEmpty() || "auto".equalsIgnoreCase(spec) || "all".equalsIgnoreCase(spec)) {
      return preferred(0, detectedGpuCount.getAsInt());
    }
    List<Device> devices = new ArrayList<>();
    int gpuCount = -1;
    boolean detected = false;
    for (String rawToken : spec.split(",", -1)) {
      String token = rawToken.trim();
      String normalized = token.toLowerCase(Locale.ROOT);
      if ("cpu".equals(normalized)) {
        devices.add(Device.cpu());
        continue;
      }
      if (normalized.startsWith("gpu:")) {
        normalized = normalized.substring("gpu:".length());
      } else if (normalized.startsWith("gpu")) {
        normalized = normalized.substring("gpu".length());
      } else if (normalized.startsWith("cuda:")) {
        normalized = normalized.substring("cuda:".length());
      }
      int index;
      try {
        index = Integer.parseInt(normalized);
      } catch (NumberFormatException failure) {
        throw new IllegalArgumentException("Invalid " + role + " device: " + token, failure);
      }
      if (index < 0) {
        throw new IllegalArgumentException("Negative GPU index in " + role + " devices: " + token);
      }
      if (validateGpuIndices && !detected) {
        gpuCount = detectedGpuCount.getAsInt();
        detected = true;
      }
      if (gpuCount == 0) {
        throw new IllegalArgumentException(
            "GPU configured for " + role + " but no GPU is visible: " + token);
      }
      if (gpuCount > 0 && index >= gpuCount) {
        throw new IllegalArgumentException(
            "GPU index outside visible range in "
                + role
                + " devices: token="
                + token
                + " visibleGpus="
                + gpuCount);
      }
      devices.add(Device.gpu(index));
    }
    return NetworkDevices.copyOf(devices);
  }

  private static NetworkDevices preferred(int maxGpus, int gpuCount) {
    if (gpuCount > 0) {
      int limit = maxGpus > 0 ? Math.min(gpuCount, maxGpus) : gpuCount;
      List<Device> devices = new ArrayList<>(limit);
      for (int index = 0; index < limit; index++) devices.add(Device.gpu(index));
      return NetworkDevices.copyOf(devices);
    }
    log.warn("GPU count is {}", gpuCount);
    return NetworkDevices.of(Device.cpu());
  }
}
