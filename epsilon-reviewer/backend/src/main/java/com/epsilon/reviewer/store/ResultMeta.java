package com.epsilon.reviewer.store;

import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.load.RecordSummary;

/**
 * meta.json の中身。結果のインデックスの 1 件になる。
 *
 * @param owner オーナーの id（{@link OwnerId#value()}）
 * @param createdAt 解析した時刻（ISO-8601、UTC、ミリ秒まで）
 */
public record ResultMeta(
    int format,
    String resultId,
    String owner,
    String createdAt,
    ModelInfo model,
    RecordSummary record) {}
