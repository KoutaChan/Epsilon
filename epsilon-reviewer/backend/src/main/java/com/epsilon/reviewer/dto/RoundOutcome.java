package com.epsilon.reviewer.dto;

import com.epsilon.replay.WinDetails;
import java.util.List;

/** 牌譜に記録された局終了の事実。点棒増減は当該イベント一件分で、未記録なら null。 */
public record RoundOutcome(
    String type,
    Integer actor,
    Integer target,
    String winningTile,
    List<Integer> deltas,
    WinDetails details) {}
