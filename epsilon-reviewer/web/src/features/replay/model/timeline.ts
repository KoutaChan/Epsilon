import { ReviewFormatError } from "./formatError.ts";
import type {
  DecisionStep,
  EventStep,
  ReplayModel,
  ReplayStep,
} from "./types.ts";

export type ReplayStage = "table" | "round-result" | "match-result";
export interface ReplayPosition {
  readonly kind: ReplayStage;
  readonly event: EventStep;
  /** 卓に出す局面の手順（判断があれば最初の判断）。 */
  readonly snapshot: ReplayStep;
  readonly decisions: readonly DecisionStep[];
  readonly outcomeEvents: readonly EventStep[];
}
/** 局ごとの再生の位置。 */
export type Timelines = readonly (readonly ReplayPosition[])[];

interface OpenPosition {
  kind: ReplayStage;
  event: EventStep;
  snapshot: ReplayStep;
  decisions: DecisionStep[];
  outcomeEvents: EventStep[];
}

/** 全ての局の位置。対局の結果の位置は最後の局だけに付ける。 */
export function buildTimelines(model: ReplayModel): Timelines {
  return model.rounds.map((round, index) =>
    buildPlaybackPositions(round.steps, index === model.rounds.length - 1),
  );
}

// 保存された盤面・判断を借用し、同じ和了機会の複数ロンも一つの精算にまとめる。
export function buildPlaybackPositions(
  steps: readonly ReplayStep[],
  includeMatchResult = false,
): readonly ReplayPosition[] {
  const first = steps[0];
  if (!first || first.decision)
    throw new ReviewFormatError("A round must start with an event.");
  let current: OpenPosition = {
    kind: "table",
    event: first,
    snapshot: first,
    decisions: [],
    outcomeEvents: [],
  };
  const positions = [current];
  let matchStep: EventStep | null = null;
  let causeEventIndex = first.eventIndex;
  const positionsByEvent = new Map<number, OpenPosition>([
    [causeEventIndex, current],
  ]);
  for (let index = 1; index < steps.length; index++) {
    const step = steps[index]!;
    if (step.decision) {
      const target = positionsByEvent.get(step.causeEventIndex);
      if (!target)
        throw new ReviewFormatError(
          "Decision references an unavailable replay event.",
        );
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
