package com.epsilon.prism.settings;

import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.List;
import java.util.Objects;

/**
 * 学習器の設定のセクション。計算の型は、マスターの重みを持つデバイスで決まる（GPU なら BFLOAT16、CPU なら FLOAT32）。
 * ほかの型を指定すると設定の誤りになる。
 *
 * @param devices 学習に使うデバイスの並び（先頭がマスター、残りはデータ並列のレプリカ）
 * @param precision 学習器の計算の型
 * @param decodeThreads 行を読み出してエンコードするスレッドの数
 */
public record LearnerSettings(
    @Key(doc = "学習に使うデバイスの並び（先頭がマスター、残りはデータ並列のレプリカ）", value = "[cpu]", change = Change.MACHINE)
        List<String> devices,
    @Key(doc = "学習器の計算の型", value = "FLOAT32", change = Change.FIT) Precision precision,
    @Key(doc = "行を読み出してエンコードするスレッドの数", value = "48", change = Change.MACHINE) @Range(min = 1)
        int decodeThreads) {

  public LearnerSettings {
    devices = List.copyOf(devices);
    Objects.requireNonNull(precision, "precision");
    NetworkDevices parsed;
    try {
      parsed = NetworkDevices.parse(String.join(",", devices));
    } catch (IllegalArgumentException invalid) {
      throw new InvalidSetting("devices", invalid.getMessage());
    }
    Precision computed = parsed.primary().isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
    if (precision != computed) {
      throw new InvalidSetting(
          "precision",
          "the learner computes in " + computed + " on " + parsed.primary() + " in this jar, not " + precision);
    }
  }

  /** 学習に使うデバイスの並びを返す。先頭がマスター。 */
  public NetworkDevices networkDevices() {
    return NetworkDevices.parse(String.join(",", devices));
  }
}
