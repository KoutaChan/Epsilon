package com.epsilon.reviewer;

import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.store.ResultMeta;

public record HistoryEntry(
    String resultId, String createdAt, RecordSummary record, ModelInfo model) {
  static HistoryEntry of(ResultMeta meta) {
    return new HistoryEntry(meta.resultId(), meta.createdAt(), meta.record(), meta.model());
  }
}
