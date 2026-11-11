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
  /** 卓に出す局面の手順（判断があれば、最初の判断の手順）。 */
  readonly snapshot: ReplayStep;
  readonly decisions: readonly DecisionStep[];
  readonly outcomeEvents: readonly EventStep[];
}
/** 再生の位置の列を、局ごとに並べたもの。 */
export type Timelines = readonly (readonly ReplayPosition[])[];

interface OpenPosition {
  kind: ReplayStage;
  event: EventStep;
  snapshot: ReplayStep;
  decisions: DecisionStep[];
  outcomeEvents: EventStep[];
}

/** 全ての局について再生の位置を作る。対局の結果の位置は、最後の局にだけ付ける。 */
export function buildTimelines(model: ReplayModel): Timelines {
  return model.rounds.map((round, index) =>
    buildPlaybackPositions(round.steps, index === model.rounds.length - 1),
  );
}

// 保存された局面と判断は、写さずにそのまま使う。同じ牌へのダブロン・トリロンは、1 つの精算の位置にまとめる。
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
        // 今の位置にまだ判断が無ければ、立直の成立やドラの表示をその位置の局面に入れる。
        // 判断があれば新しい位置を作る。変わる前の選択率と、変わった後の局面を同じ画面に出さないため。
        // 入れたイベントをきっかけにする判断（大明槓・加槓の後の打牌でめくったドラの次の応答）も、その位置に置く。
        if (current.decisions.length === 0) {
          current.snapshot = step;
          positionsByEvent.set(step.eventIndex, current);
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
