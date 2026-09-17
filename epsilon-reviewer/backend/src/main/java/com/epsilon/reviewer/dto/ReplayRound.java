package com.epsilon.reviewer.dto;

import java.util.List;

/** 本場を区別する一局分の状態列とイベント・判断列。 */
public record ReplayRound(
    String id,
    int roundIndex,
    int honba,
    int dealer,
    List<StateFrame> states,
    List<ReplayStep> steps) {}
