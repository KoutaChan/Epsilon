package com.epsilon.nano.ai.network;

import ai.djl.Device;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/** ネットワークの実行に使うデバイスを、主デバイスが先頭となる順序で保持する。 */
public final class NetworkDevices implements Iterable<Device> {

  private final List<Device> devices;

  private NetworkDevices(List<Device> devices) {
    if (devices.isEmpty()) {
      throw new IllegalArgumentException("Network devices must not be empty");
    }
    this.devices = List.copyOf(devices);
  }

  /**
   * 指定順を維持したデバイス集合を作る。
   *
   * <p>同じデバイスを複数実行単位へ割り当てるCPU テスト等を許すため、重複は保持する。
   *
   * @param devices 主デバイスを先頭に並べた1台以上のデバイス
   * @return 変更できないデバイス集合
   */
  public static NetworkDevices of(Device... devices) {
    Objects.requireNonNull(devices, "devices");
    ArrayList<Device> copy = new ArrayList<>(devices.length);
    for (Device device : devices) {
      copy.add(Objects.requireNonNull(device, "device"));
    }
    return new NetworkDevices(copy);
  }

  /**
   * コレクションの反復順を維持したデバイス集合を作る。
   *
   * @param devices 主デバイスを先頭に並べた1台以上のデバイス
   * @return 変更できないデバイス集合
   */
  public static NetworkDevices copyOf(Collection<? extends Device> devices) {
    Objects.requireNonNull(devices, "devices");
    ArrayList<Device> copy = new ArrayList<>(devices.size());
    for (Device device : devices) {
      copy.add(Objects.requireNonNull(device, "device"));
    }
    return new NetworkDevices(copy);
  }

  /**
   * @return 更新の基準となるモデル配置等に使う先頭デバイス
   */
  public Device primary() {
    return getFirst();
  }

  /**
   * @return 先頭デバイス
   * @throws java.util.NoSuchElementException デバイスが存在しない場合
   */
  public Device getFirst() {
    return devices.getFirst();
  }

  /**
   * @return 末尾デバイス
   * @throws java.util.NoSuchElementException デバイスが存在しない場合
   */
  public Device getLast() {
    return devices.getLast();
  }

  /**
   * 指定位置のデバイスを返す。
   *
   * @param index 0始まりの位置
   * @return 指定位置のデバイス
   */
  public Device get(int index) {
    return devices.get(index);
  }

  /**
   * @return デバイス数
   */
  public int size() {
    return devices.size();
  }

  /**
   * @param device 検索するデバイス
   * @return 同じデバイスが一つ以上含まれるなら {@code true}
   */
  public boolean contains(Device device) {
    return devices.contains(device);
  }

  /**
   * 先頭から指定台数だけを保持する集合を返す。
   *
   * @param count 取り出す台数
   * @return 主デバイスと順序を維持した部分集合
   */
  public NetworkDevices first(int count) {
    if (count < 1 || count > devices.size()) {
      throw new IllegalArgumentException(
          "Network device prefix size must be in [1," + devices.size() + "]: " + count);
    }
    if (count == devices.size()) {
      return this;
    }
    return new NetworkDevices(devices.subList(0, count));
  }

  /**
   * 集合内の指定デバイスを先頭へ移した順序を返す。
   *
   * <p>元の集合とデバイス重複数は変えず、指定デバイスと現在の先頭だけを入れ替える。
   *
   * @param primary 新しい主デバイス
   * @return 指定したデバイスと先頭を入れ替えた集合
   */
  public NetworkDevices withPrimary(Device primary) {
    Objects.requireNonNull(primary, "primary");
    int index = devices.indexOf(primary);
    if (index < 0) {
      throw new IllegalArgumentException(
          "Primary device is not present in network devices: primary="
              + primary
              + " devices="
              + devices);
    }
    if (index == 0) {
      return this;
    }
    ArrayList<Device> ordered = new ArrayList<>(devices);
    Collections.swap(ordered, 0, index);
    return new NetworkDevices(ordered);
  }

  /**
   * @return 読み取り専用のデバイス一覧
   */
  public List<Device> asList() {
    return devices;
  }

  /**
   * DJLの配列APIへ渡すための防御的複製を返す。
   *
   * @return 呼び出し側が変更してもこの集合へ影響しない配列
   */
  public Device[] toArray() {
    return devices.toArray(Device[]::new);
  }

  @Override
  public Iterator<Device> iterator() {
    return devices.iterator();
  }

  @Override
  public boolean equals(Object other) {
    return this == other || (other instanceof NetworkDevices that && devices.equals(that.devices));
  }

  @Override
  public int hashCode() {
    return devices.hashCode();
  }

  @Override
  public String toString() {
    return devices.toString();
  }
}
