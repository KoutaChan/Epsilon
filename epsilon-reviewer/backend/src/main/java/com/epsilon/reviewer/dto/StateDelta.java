package com.epsilon.reviewer.dto;

import java.util.List;

/** 直前の一意な局面からの表示差分。麻雀ルールの再実行を必要としない。 */
public record StateDelta(
    List<PlayerDelta> players,
    ArrayPatch<String> dora,
    Integer honba,
    Integer kyotaku,
    Integer remaining,
    Integer activeSeat) {}
