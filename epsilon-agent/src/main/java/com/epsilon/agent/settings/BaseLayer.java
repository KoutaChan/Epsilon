package com.epsilon.agent.settings;

import java.util.Objects;
import java.util.function.Function;

/**
 * いちばん下に入れる層（四人打ち・三人打ちごとの既定値）。上の層を重ねた生の木で、1 つのキーの値を見て選ぶ。
 *
 * @param key 層を選ぶキーのパス（{@code run.variant}）。全体の record のセクションをたどった先にある、値が 1 つのキー
 * @param layer キーの値（文字列にしたもの）から層を作る処理。作れない値には IllegalArgumentException を投げる
 */
public record BaseLayer(String key, Function<String, Layer> layer) {

  public BaseLayer {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(layer, "layer");
  }
}
