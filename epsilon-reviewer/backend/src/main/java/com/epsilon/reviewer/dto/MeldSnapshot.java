package com.epsilon.reviewer.dto;

import java.util.List;

/** 横向き牌と加槓牌を区別した表示順の面子。 */
public record MeldSnapshot(
    String type, List<String> tiles, int from, int calledIndex, int addedIndex) {}
