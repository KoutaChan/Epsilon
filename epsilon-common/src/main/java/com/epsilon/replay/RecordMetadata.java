package com.epsilon.replay;

import java.util.List;

/** 元牌譜の対局情報。未記録の点数とルールは null とし、既定値で補わない。 */
public record RecordMetadata(
    RecordFormat format,
    List<String> names,
    int[] finalScores,
    String finalScoresSource,
    String ruleDescription) {}
