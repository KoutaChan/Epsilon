package com.epsilon.agent.policy;

import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import java.util.List;
import java.util.Objects;

/**
 * 方策を開く設定。打つ側の全ての入口（評価の相手、牌譜、天鳳、RiichiLab、Reviewer）が、この設定で方策を開く。ネットワークを持たない
 * 系列はこの設定を読まない。どのキーも、次に起動して方策を開き直したときから効く。
 *
 * @param device 推論のデバイス。cpu か gpuN で、{@link NetworkDevices#device} が読む形
 * @param precision 計算の型
 * @param maxBatch 1 回の推論に入れる行の上限
 * @param kernels 推論のバックエンド。専用のカーネルを持たない系列で {@link Kernels#NATIVE} を指定すると、開くときに拒む
 */
public record PolicyOptions(
    @Key(doc = "推論のデバイス（cpu か gpuN）", value = "cpu", change = Change.MACHINE) String device,
    @Key(doc = "計算の型", value = "FLOAT32", change = Change.MACHINE) Precision precision,
    @Key(doc = "1 回の推論に入れる行の上限", value = "8", change = Change.MACHINE) @Range(min = 1) int maxBatch,
    @Key(doc = "推論のバックエンド（専用のカーネルが無い系列で NATIVE にすると、開くときに拒む）", value = "EAGER", change = Change.MACHINE)
        Kernels kernels) {

  private static final PolicyOptions DEFAULTS =
      Settings.section(PolicyOptions.class, "policy", List.of()).value();

  public PolicyOptions {
    Objects.requireNonNull(device, "device");
    try {
      NetworkDevices.device(device);
    } catch (IllegalArgumentException invalid) {
      throw new InvalidSetting("device", invalid.getMessage());
    }
    Objects.requireNonNull(precision, "precision");
    Objects.requireNonNull(kernels, "kernels");
    if (maxBatch < 1) {
      throw new InvalidSetting("maxBatch", "maxBatch must be positive: " + maxBatch);
    }
  }

  /** 全てのキーを、アノテーションに書いた既定値にした設定。 */
  public static PolicyOptions defaults() {
    return DEFAULTS;
  }

  /** デバイスだけを替えた設定。 */
  public PolicyOptions onDevice(String other) {
    return new PolicyOptions(other, precision, maxBatch, kernels);
  }
}
