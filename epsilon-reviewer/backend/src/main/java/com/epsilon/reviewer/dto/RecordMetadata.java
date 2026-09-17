package com.epsilon.reviewer.dto;

import java.util.List;

/** 入力牌譜から確定できた情報。最終点が取得できないときは null とする。 */
public record RecordMetadata(
    String source,
    String fileName,
    List<String> names,
    List<Integer> finalScores,
    String finalScoresSource,
    int roundCount) {}
