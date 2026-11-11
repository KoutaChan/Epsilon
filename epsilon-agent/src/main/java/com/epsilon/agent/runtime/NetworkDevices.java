package com.epsilon.agent.runtime;

import ai.djl.Device;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * ネットワークを置くデバイスの一覧。先頭が主のデバイスで、学習では主のネットワークを置き、推論では最初の実行先になる。
 *
 * <p>同じデバイスが何度出てもよい。CPU のテストでは、1 つのデバイスに複数のレーンを置く。ただし設定から読む一覧（{@link #parse}）は、
 * cpu を 1 つか、重複しない GPU の一覧に限る。
 */
public record NetworkDevices(List<Device> devices) implements Iterable<Device> {

  private static final Pattern NAME = Pattern.compile("cpu|gpu(0|[1-9]\\d*)");

  public NetworkDevices {
    devices = List.copyOf(devices);
    if (devices.isEmpty()) {
      throw new IllegalArgumentException("Network devices must not be empty");
    }
  }

  public static NetworkDevices of(Device... devices) {
    return new NetworkDevices(List.of(devices));
  }

  /** 名前（cpu か gpuN）が表すデバイスを返す。名前の形が違えば IllegalArgumentException を投げる。 */
  public static Device device(String name) {
    Matcher matcher = NAME.matcher(Objects.requireNonNull(name, "name"));
    if (!matcher.matches()) {
      throw new IllegalArgumentException("Device must be cpu or gpuN: " + name);
    }
    return matcher.group(1) == null ? Device.cpu() : Device.gpu(Integer.parseInt(matcher.group(1)));
  }

  /** カンマで区切ったデバイスの名前を読む。cpu を 1 つか、重複しない GPU の一覧でなければ IllegalArgumentException を投げる。 */
  public static NetworkDevices parse(String names) {
    List<Device> devices = new ArrayList<>();
    for (String name : names.split(",", -1)) {
      devices.add(device(name.trim()));
    }
    boolean cpu = devices.stream().anyMatch(device -> !device.isGpu());
    if ((cpu && devices.size() > 1) || new HashSet<>(devices).size() < devices.size()) {
      throw new IllegalArgumentException("Devices must be one cpu or distinct GPUs: " + names);
    }
    return new NetworkDevices(devices);
  }

  public Device primary() {
    return devices.getFirst();
  }

  public Device get(int index) {
    return devices.get(index);
  }

  public int size() {
    return devices.size();
  }

  @Override
  public Iterator<Device> iterator() {
    return devices.iterator();
  }
}
