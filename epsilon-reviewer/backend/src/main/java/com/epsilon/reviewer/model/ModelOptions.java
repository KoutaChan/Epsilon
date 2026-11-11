package com.epsilon.reviewer.model;

import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;

/**
 * models.json に書くモデルごとの options。方策を開く引数（{@link PolicyOptions} と同じキーと既定値）と、価値から表示の確率を作る温度。
 *
 * @param temperature 温度の上書き。無ければ null で、方策の温度を使う
 */
public record ModelOptions(
    @Key(doc = "推論のデバイス（cpu か gpuN）", value = "cpu") String device,
    @Key(doc = "計算に使う数値の型", value = "FLOAT32") Precision precision,
    @Key(doc = "1 回の推論にまとめる行数の上限", value = "8") @Range(min = 1) int maxBatch,
    @Key(doc = "推論のバックエンド（専用のカーネルを持たない系列では、開くときに断る）", value = "EAGER") Kernels kernels,
    @Key(doc = "価値から表示の確率を作る温度（単位は効用。書かなければ方策の温度を使う）", presence = Presence.ABSENT)
        @Range(min = 0, minInclusive = false)
        Double temperature) {

  public ModelOptions {
    try {
      NetworkDevices.device(device);
    } catch (IllegalArgumentException invalid) {
      throw new InvalidSetting("device", "expected cpu or gpuN");
    }
  }

  /** 方策を開く引数を作る。 */
  PolicyOptions policy() {
    return new PolicyOptions(device, precision, maxBatch, kernels);
  }
}
