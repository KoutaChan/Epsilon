import { useCallback, useEffect, useMemo, useReducer } from "react";
import type { ReviewResult } from "../../api/types";
import { playbackReducer } from "./playback";
import { buildPlaybackPositions } from "./timeline";
import {
  createReplayStateReader,
  maximumPlayerRowWidth,
} from "./stateReader.ts";
import { measurePlayerRowWidth } from "./renderer/tileLayout";

export function usePlayback(result: ReviewResult, initialSeat: number) {
  const [state, dispatch] = useReducer(playbackReducer, {
    roundIndex: 0,
    stepIndex: 0,
    seat: initialSeat,
    reveal: true,
    playing: false,
    animateDraw: false,
  });
  const timelines = useMemo(
    () =>
      result.rounds.map((round, index) =>
        buildPlaybackPositions(round.steps, index === result.rounds.length - 1),
      ),
    [result],
  );
  const reader = useMemo(() => createReplayStateReader(result), [result]);
  const playerRowWidth = useMemo(
    () => maximumPlayerRowWidth(result, measurePlayerRowWidth),
    [result],
  );
  const round = result.rounds[state.roundIndex];
  const nextRound = result.rounds[state.roundIndex + 1];
  const positions = timelines[state.roundIndex];
  const {
    snapshot: snapshotStep,
    event: eventStep,
    decisions,
    kind: stage,
    outcomeEvents: outcomeSteps,
  } = positions[state.stepIndex];
  const snapshot = reader.readSnapshot(state.roundIndex, snapshotStep);
  const event = reader.readSnapshot(state.roundIndex, eventStep);
  const eventPrevious =
    eventStep.eventType === "kakan"
      ? reader.readSnapshot(state.roundIndex, round.steps[eventStep.index - 1])
      : null;
  const roundStart = reader.readSnapshot(state.roundIndex, round.steps[0]);
  const outcomeEvents = outcomeSteps.map((step) =>
    reader.readSnapshot(state.roundIndex, step),
  );
  const recentEvents = positions
    .slice(Math.max(0, state.stepIndex - 2), state.stepIndex + 1)
    .map((position) => ({
      kind: position.kind,
      event: reader.readSnapshot(state.roundIndex, position.event),
      previous:
        position.event.eventType === "kakan"
          ? reader.readSnapshot(
              state.roundIndex,
              round.steps[position.event.index - 1],
            )
          : null,
    }));
  const candidates =
    decisions.find((decision) => decision.actor === state.seat)?.candidates ??
    [];
  const last = positions.length - 1;
  const canPrevious = state.roundIndex > 0 || state.stepIndex > 0;
  const canNext = nextRound !== undefined || state.stepIndex < last;
  const canPlay = stage === "table" && canNext;
  const move = useCallback(
    (delta: number) => dispatch({ type: "step", delta, timelines }),
    [timelines],
  );
  const togglePlay = useCallback(
    () => dispatch({ type: "play", timelines }),
    [timelines],
  );
  useEffect(() => {
    if (!state.playing) return;
    const timer = setInterval(
      () => dispatch({ type: "step", delta: 1, timelines, automatic: true }),
      850,
    );
    return () => clearInterval(timer);
  }, [state.playing, timelines]);
  return {
    ...state,
    playerRowWidth,
    round,
    nextRound,
    positions,
    recentEvents,
    roundStart,
    stage,
    outcomeEvents,
    event,
    eventPrevious,
    snapshot,
    candidates,
    last,
    canPrevious,
    canNext,
    canPlay,
    dispatch,
    move,
    togglePlay,
  };
}
export type Playback = ReturnType<typeof usePlayback>;
