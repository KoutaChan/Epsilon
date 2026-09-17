package com.epsilon.reviewer.model;

import com.epsilon.reviewer.dto.ModelInfo;
import java.nio.file.Path;
import java.util.Map;

/** パスを HTTP 応答へ公開しない、サーバー内の固定モデル定義です。 */
public record ModelDefinition(
    String modelId,
    String revision,
    String displayName,
    String version,
    String series,
    Path checkpoint,
    Map<String, String> options) {
  /** オプションは作成側から所有権を移譲し、構築後は変更しない。 */
  public ModelInfo info() {
    return new ModelInfo(modelId, revision, displayName, version, series);
  }
}
