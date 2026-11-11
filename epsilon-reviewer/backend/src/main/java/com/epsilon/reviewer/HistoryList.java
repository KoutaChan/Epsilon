package com.epsilon.reviewer;

import java.util.List;

/** 解析の履歴。新しい解析が先に並ぶ。 */
public record HistoryList(List<HistoryEntry> results) {}
