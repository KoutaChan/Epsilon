import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  type Dispatch,
} from "react";
import { readCandidates } from "./model/candidates.ts";
import {
  playbackReducer,
  type PlaybackAction,
  type PlaybackState,
} from "./model/playback.ts";
import type { RecentReplayEvent } from "./model/recentEvents.ts";
import {
  createStateReader,
  maximumPlayerRowWidth,
  type StateReader,
} from "./model/stateReader.ts";
import type {
  ReplayPosition,
  ReplayStage,
  Timelines,
} from "./model/timeline.ts";
import type {
  Candidate,
  DecisionStep,
  EventSnapshot,
  ReplayModel,
  ReplayRound,
  ReplayStep,
  Snapshot,
} from "./model/types.ts";
import { measurePlayerRowWidth } from "./renderer/tileLayout.ts";

const AUTOPLAY_INTERVAL_MS = 850;

/** 今の位置で画面が使う局面とイベント。位置（局・手順・席）が変わったときだけ作り直す。 */
interface PositionView {
  readonly stage: ReplayStage;
  /** 卓に出す局面（判断があれば、最初の判断の局面）。 */
  readonly snapshot: Snapshot;
  readonly event: EventSnapshot;
  /** 加槓の直前の局面。加槓でなければ null。 */
  readonly eventPrevious: Snapshot | null;
  readonly roundStart: Snapshot;
  readonly outcomeEvents: readonly EventSnapshot[];
  /** この局の直近 3 つの位置（今の位置を含む）のイベント。 */
  readonly recentEvents: readonly RecentReplayEvent[];
  /** 見ている席の判断。無ければ null。 */
  readonly decision: DecisionStep | null;
  readonly candidates: readonly Candidate[];
}

export interface Playback extends PlaybackState, PositionView {
  readonly model: ReplayModel;
  readonly round: ReplayRound;
  readonly nextRound: ReplayRound | undefined;
  readonly positions: readonly ReplayPosition[];
  /** 対局全体で、席の列が最も広くなるときの幅。カメラを固定するのに使う。 */
  readonly rowWidth: number;
  readonly last: number;
  readonly canPrevious: boolean;
  readonly canNext: boolean;
  readonly canPlay: boolean;
  readonly dispatch: Dispatch<PlaybackAction>;
  move(delta: number): void;
  togglePlay(): void;
}

export function usePlayback(
  model: ReplayModel,
  timelines: Timelines,
  initialSeat: number,
): Playback {
  const [state, dispatch] = useReducer(
    playbackReducer,
    initialSeat,
    (seat): PlaybackState => ({
      roundIndex: 0,
      stepIndex: 0,
      seat,
      reveal: true,
      playing: false,
      animateDraw: false,
    }),
  );
  const reader = useMemo(() => createStateReader(model), [model]);
  const rowWidth = useMemo(
    () => maximumPlayerRowWidth(model, measurePlayerRowWidth),
    [model],
  );
  const { roundIndex, stepIndex, seat } = state;
  const round = model.rounds[roundIndex]!;
  const positions = timelines[roundIndex]!;
  const view = useMemo(
    () => readPosition(model, reader, round, positions, stepIndex, seat),
    [model, reader, round, positions, stepIndex, seat],
  );
  const last = positions.length - 1;
  const nextRound = model.rounds[roundIndex + 1];
  const canNext = nextRound !== undefined || stepIndex < last;
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
      AUTOPLAY_INTERVAL_MS,
    );
    return () => clearInterval(timer);
  }, [state.playing, timelines]);
  return {
    ...state,
    ...view,
    model,
    round,
    nextRound,
    positions,
    rowWidth,
    last,
    canPrevious: roundIndex > 0 || stepIndex > 0,
    canNext,
    canPlay: view.stage === "table" && canNext,
    dispatch,
    move,
    togglePlay,
  };
}

function readPosition(
  model: ReplayModel,
  reader: StateReader,
  round: ReplayRound,
  positions: readonly ReplayPosition[],
  stepIndex: number,
  seat: number,
): PositionView {
  const position = positions[stepIndex]!;
  const read = <S extends ReplayStep>(step: S) =>
    reader.readSnapshot(round.index, step);
  // 加槓では、どの副露に牌を足したかを調べるため、直前の手順の局面も読む。
  const previous = (event: ReplayPosition["event"]) =>
    event.eventType === "kakan" ? read(round.steps[event.index - 1]!) : null;
  const decision =
    position.decisions.find((entry) => entry.actor === seat) ?? null;
  return {
    stage: position.kind,
    snapshot: read(position.snapshot),
    event: read(position.event),
    eventPrevious: previous(position.event),
    roundStart: read(round.steps[0]!),
    outcomeEvents: position.outcomeEvents.map(read),
    recentEvents: positions
      .slice(Math.max(0, stepIndex - 2), stepIndex + 1)
      .map((entry) => ({
        kind: entry.kind,
        event: read(entry.event),
        previous: previous(entry.event),
      })),
    decision,
    candidates: decision ? readCandidates(decision.set, model.temperature) : [],
  };
}
