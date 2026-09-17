package com.epsilon.reviewer.dto;

import java.util.List;

/** イベントまたはAI判断と、その表示状態への参照。 */
public record ReplayStep(
    int index,
    int eventIndex,
    int stateId,
    String eventType,
    int actor,
    boolean decision,
    Integer causeEventIndex,
    List<ActionCandidate> candidates,
    RoundOutcome outcome) {}
