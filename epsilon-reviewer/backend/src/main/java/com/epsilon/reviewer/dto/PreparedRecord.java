package com.epsilon.reviewer.dto;

import com.epsilon.replay.ReplayRecord;

/** 合法な再生を確認済みの入力。共通牌譜の所有権を解析ジョブへ渡す。 */
public record PreparedRecord(RecordMetadata metadata, ReplayRecord replay) {}
