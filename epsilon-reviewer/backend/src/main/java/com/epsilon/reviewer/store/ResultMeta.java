package com.epsilon.reviewer.store;

import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.load.RecordSummary;

/**
 * meta.json（索引の 1 件）。
 *
 * @param owner 持ち主の id（{@link OwnerId#value()}）
 * @param createdAt 解析した時刻（ISO-8601、UTC、ミリ秒まで）
 */
public record ResultMeta(
    int format,
    String resultId,
    String owner,
    String createdAt,
    ModelInfo model,
    RecordSummary record) {}
