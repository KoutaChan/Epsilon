package com.epsilon.reviewer.record;

import com.epsilon.reviewer.engine.load.RecordSummary;

/** 読み込んで解析を待つ牌譜（recordId は 30 分だけ使える）。 */
public record ImportedRecord(String recordId, RecordSummary record) {}
