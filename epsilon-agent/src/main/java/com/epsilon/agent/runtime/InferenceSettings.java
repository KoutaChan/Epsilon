package com.epsilon.agent.runtime;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.List;
import java.util.Objects;

/**
 * 推論のセクション。収集と評価の卓が、推論のリソースを開くときに読む。計算の型はデバイスからは決めず、このセクションのキーで決める。
 * 計算の型を替えると、集める行が変わる。ほかのキーが変えるのは推論の速さだけで、結果の分布は変わらない（ビット単位では変わりうる）。
 */
public record InferenceSettings(
    @Key(doc = "推論に使うデバイスの一覧（cpu を 1 つか、重複しない GPU の一覧）", value = "[cpu]", change = Change.MACHINE)
        List<String> devices,
    @Key(doc = "計算の型", value = "FLOAT32", change = Change.COLLECT) Precision precision,
    @Key(doc = "推論のバックエンド", value = "EAGER", change = Change.MACHINE) Kernels kernels,
    @Key(doc = "1 回の推論に入れる行数の上限", value = "8", change = Change.MACHINE) @Range(min = 1) int maxBatch,
    @Key(doc = "入力を符号化するスレッドの数", value = "20", change = Change.MACHINE) @Range(min = 1) int encoders,
    @Key(doc = "GPU ごとに同時に実行するバッチの数", value = "2", change = Change.MACHINE) @Range(min = 1, max = 4)
        int slots,
    @Key(doc = "GPU ごとに、符号化を終えて実行を待てるバッチの数（同時に実行するバッチの数以上）", value = "4", change = Change.MACHINE)
        @Range(min = 1)
        int readyBatches) {

  public InferenceSettings {
    devices = List.copyOf(devices);
    Objects.requireNonNull(precision, "precision");
    Objects.requireNonNull(kernels, "kernels");
    try {
      NetworkDevices.parse(String.join(",", devices));
    } catch (IllegalArgumentException invalid) {
      throw new InvalidSetting("devices", invalid.getMessage());
    }
    if (readyBatches < slots) {
      throw new InvalidSetting("readyBatches", "readyBatches " + readyBatches + " is below slots " + slots);
    }
  }

  /** 推論に使うデバイスの一覧。先頭が最初の実行先。 */
  public NetworkDevices networkDevices() {
    return NetworkDevices.parse(String.join(",", devices));
  }
}
