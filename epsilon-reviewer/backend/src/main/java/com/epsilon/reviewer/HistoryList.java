package com.epsilon.reviewer;

import java.util.List;

/** 履歴（新しい解析が先）。 */
public record HistoryList(List<HistoryEntry> results) {}
