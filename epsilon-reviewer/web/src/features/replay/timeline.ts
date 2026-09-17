import type { ReviewStep } from "../../api/types";

export type ReplayStage = "table" | "round-result" | "match-result";
export interface ReplayPosition {
  kind: ReplayStage;
  event: ReviewStep;
  snapshot: ReviewStep;
  decisions: readonly ReviewStep[];
  outcomeEvents: readonly ReviewStep[];
}

// 保存された盤面・判断を借用し、同じ和了機会の複数ロンも一つの精算にまとめる。
export function buildPlaybackPositions(
  steps: readonly ReviewStep[],
  includeMatchResult = false,
): readonly ReplayPosition[] {
  let current = {
    kind: "table" as ReplayStage,
    event: steps[0],
    snapshot: steps[0],
    decisions: [] as ReviewStep[],
    outcomeEvents: [] as ReviewStep[],
  };
  const positions = [current];
  let matchStep: ReviewStep | null = null;
  let causeEventIndex = steps[0].eventIndex;
  const positionsByEvent = new Map<number, typeof current>([
    [causeEventIndex, current],
  ]);
  for (let index = 1; index < steps.length; index++) {
    const step = steps[index];
    if (step.decision) {
      const target = positionsByEvent.get(step.causeEventIndex);
      if (!target)
        throw new Error("Decision references an unavailable replay event.");
      if (target.decisions.length === 0) target.snapshot = step;
      target.decisions.push(step);
      continue;
    }
    switch (step.eventType) {
      case "hora":
      case "ryukyoku":
        if (current.kind !== "round-result") {
          current = {
            kind: "round-result",
            event: step,
            snapshot: step,
            decisions: [],
            outcomeEvents: [],
          };
          positions.push(current);
        }
        current.snapshot = step;
        current.outcomeEvents.push(step);
        continue;
      case "end_kyoku":
        if (current.kind === "round-result") current.snapshot = step;
        continue;
      case "end_game":
        matchStep = step;
        continue;
      case "reach":
      case "none":
        continue;
      case "reach_accepted":
      case "dora":
        // 判断後のドラ・得点変化を、変化前の選択率と同じ盤面にしない。
        if (current.decisions.length === 0) {
          current.snapshot = step;
          continue;
        }
    }
    if (step.eventType !== "dora" && step.eventType !== "reach_accepted")
      causeEventIndex = step.eventIndex;
    current = {
      kind: "table",
      event: step,
      snapshot: step,
      decisions: [],
      outcomeEvents: [],
    };
    positions.push(current);
    positionsByEvent.set(causeEventIndex, current);
    positionsByEvent.set(step.eventIndex, current);
  }
  if (
    includeMatchResult &&
    (matchStep !== null || current.kind === "round-result")
  ) {
    positions.push({
      kind: "match-result",
      event: matchStep ?? current.event,
      snapshot: matchStep ?? current.snapshot,
      decisions: [],
      outcomeEvents: current.outcomeEvents,
    });
  }
  return positions;
}
