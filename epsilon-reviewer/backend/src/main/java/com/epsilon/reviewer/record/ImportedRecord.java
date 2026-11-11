package com.epsilon.reviewer.record;

import com.epsilon.reviewer.engine.load.RecordSummary;

/** 読み込んで、解析を待っている牌譜（{@code recordId} が使えるのは 30 分だけ）。 */
public record ImportedRecord(String recordId, RecordSummary record) {}
