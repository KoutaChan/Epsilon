package com.epsilon.reviewer.dto;

import java.util.List;

/** 確率は全合法行動で正規化済みの 0..1。赤牌・ツモ切りを保持する。 */
public record ActionCandidate(
    int actionId,
    String type,
    List<String> tiles,
    float probability,
    boolean chosen,
    boolean tsumogiri) {}
