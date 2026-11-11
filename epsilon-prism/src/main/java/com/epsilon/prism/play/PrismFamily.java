package com.epsilon.prism.play;

import ai.djl.Device;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.BehaviourSettings;
import com.epsilon.prism.variant.PrismVariant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * PRISM の系列（{@link Series#PRISM}）。公開したバージョン（snapshots/&lt;t&gt;）の打つネットワークを開き、{@link PrismPolicy} を返す。
 * デバイス・計算の型・行の上限・推論のバックエンドは、渡された方策の設定をそのまま使う。run の設定は読まないので、同値な打牌は既定の
 * 切り方（{@link BehaviourSettings#defaults}）で切る。META-INF/services に登録してあり、ServiceLoader で見つかる。
 */
public final class PrismFamily implements ModelFamily {

  @Override
  public Series series() {
    return Series.PRISM;
  }

  /**
   * @param runtime 呼び出し元のランタイム。GPU ではこの上にバージョンを載せる。null なら、GPU では方策が自分のランタイムを作って持つ
   * @throws IllegalStateException そのルールの PRISM のモジュールが classpath に無いとき、または開いたネットワークの入力が指定したルールのものでないとき
   * @throws IllegalArgumentException 専用のカーネルを BFLOAT16 以外の計算の型で開こうとしたとき
   */
  @Override
  public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime)
      throws IOException {
    Objects.requireNonNull(options, "options");
    String label = Series.PRISM.label(checkpoint);
    PrismVariant.of(variant);
    Device device = NetworkDevices.device(options.device());
    NetworkRuntime owned = device.isGpu() && runtime == null ? new NetworkRuntime(1, 1, 1) : null;
    try {
      NetworkPolicy network =
          Snapshot.open(
              checkpoint,
              List.of(device),
              runtime != null ? runtime : owned,
              options.maxBatch(),
              options.precision(),
              InferencePath.of(options.kernels()));
      if (network.format().variant() != variant) {
        Variant found = network.format().variant();
        network.close();
        throw new IllegalStateException(
            "Variant mismatch: expected " + variant + " but found " + found + " in " + checkpoint);
      }
      return new PrismPolicy(network, variant, label, owned, BehaviourSettings.defaults().twinDiscard());
    } catch (IOException | RuntimeException | Error failure) {
      if (owned != null) {
        try {
          owned.close();
        } catch (RuntimeException | Error closing) {
          failure.addSuppressed(closing);
        }
      }
      throw failure;
    }
  }
}
