package com.epsilon.reviewer.model;

import com.epsilon.reviewer.engine.format.ModelInfo;
import java.nio.file.Path;
import java.util.Map;

/**
 * models.json の 1 モデル。
 *
 * @param checkpoint チェックポイントのフォルダー（絶対パス）
 * @param options 提供元へ渡す選択肢
 * @param sha256 登録したチェックポイントの SHA-256（{@link CheckpointDigest}）。無ければ null
 * @param temperature 価値の評価の表示の温度（options.temperature か既定）
 */
public record ModelEntry(
    ModelKey key,
    String displayName,
    String version,
    String series,
    Path checkpoint,
    Map<String, String> options,
    String sha256,
    double temperature,
    Availability availability) {
  public ModelInfo info() {
    return new ModelInfo(key.modelId(), key.revision(), displayName, version);
  }

  ModelEntry with(Availability changed) {
    return new ModelEntry(
        key, displayName, version, series, checkpoint, options, sha256, temperature, changed);
  }
}
