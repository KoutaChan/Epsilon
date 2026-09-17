package com.epsilon.reviewer.dto;

import java.util.List;

/** 再生用の局面差分と定期的な全状態、および各判断でモデルが出力した方策確率を含む解析結果。 */
public record ReviewResult(
    int formatVersion,
    String resultId,
    String createdAt,
    RecordMetadata metadata,
    ModelInfo model,
    List<ReplayRound> rounds) {
  public static final int CURRENT_FORMAT_VERSION = 4;
}
