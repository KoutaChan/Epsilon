package com.epsilon.reviewer.model;

import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.format.ModelInfo;
import java.nio.file.Path;

/**
 * models.json に書いたモデル 1 つ。
 *
 * @param checkpoint チェックポイントのフォルダー（絶対パス）。チェックポイントを使わない系列は null
 * @param options 方策を開く設定
 * @param variant モデルが読む牌譜のルール
 * @param sha256 登録したチェックポイントの SHA-256（{@code CheckpointFiles.sha256} の値）。無ければ null
 * @param temperature 価値から表示の確率を作る温度の上書き（options.temperature）。無ければ null で、方策の温度を使う
 * @param available 使えるか。系列を扱うクラスが jar に無いモデル、チェックポイントを読めないモデル（シンボリックリンクを含むか、
 *     ファイルが 1 つも無いものも）、開くときにチェックポイントが登録したときと違っていたモデルは false
 */
public record ModelEntry(
    ModelKey key,
    String displayName,
    String version,
    Series series,
    Path checkpoint,
    PolicyOptions options,
    Variant variant,
    String sha256,
    Double temperature,
    boolean available) {
  public ModelInfo info() {
    return new ModelInfo(key.modelId(), key.revision(), displayName, version);
  }

  /** 使えなくしたモデル。ほかの値は変えない。 */
  ModelEntry unavailable() {
    return new ModelEntry(
        key,
        displayName,
        version,
        series,
        checkpoint,
        options,
        variant,
        sha256,
        temperature,
        false);
  }
}
