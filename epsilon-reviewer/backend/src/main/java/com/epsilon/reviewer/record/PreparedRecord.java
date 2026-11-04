package com.epsilon.reviewer.record;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.store.OwnerId;

/**
 * 読み込んで解析を待つ牌譜。
 *
 * @param source 受け取ったままのバイト列（保存の record.gz になる）
 */
public record PreparedRecord(RecordId id, OwnerId owner, LoadedRecord record, byte[] source) {}
