import type { ReplayPosition } from "./timeline";

export interface PlaybackState {
  roundIndex: number;
  stepIndex: number;
  seat: number;
  reveal: boolean;
  playing: boolean;
  animateDraw: boolean;
}
type PlaybackTimelines = readonly (readonly ReplayPosition[])[];
export type PlaybackAction =
  | {
      type: "step";
      delta: number;
      timelines: PlaybackTimelines;
      automatic?: boolean;
    }
  | { type: "round"; index: number }
  | { type: "seat"; seat: number }
  | { type: "seek"; index: number }
  | { type: "reveal"; reveal: boolean }
  | { type: "play"; timelines: PlaybackTimelines }
  | { type: "pause" };
export function playbackReducer(
  state: PlaybackState,
  action: PlaybackAction,
): PlaybackState {
  switch (action.type) {
    case "step": {
      const { timelines } = action;
      let roundIndex = state.roundIndex;
      let stepIndex = state.stepIndex;
      const direction = Math.sign(action.delta);
      for (let remaining = Math.abs(action.delta); remaining > 0; remaining--) {
        // 精算画面の確認後は手動で次へ進める。
        if (
          action.automatic &&
          timelines[roundIndex][stepIndex].kind !== "table"
        )
          break;
        if (direction > 0) {
          if (stepIndex < timelines[roundIndex].length - 1) stepIndex++;
          else if (roundIndex < timelines.length - 1) {
            roundIndex++;
            stepIndex = 0;
          } else break;
        } else if (direction < 0) {
          if (stepIndex > 0) stepIndex--;
          else if (roundIndex > 0) {
            roundIndex--;
            stepIndex = timelines[roundIndex].length - 1;
          } else break;
        }
      }
      const position = timelines[roundIndex][stepIndex];
      const moved =
        roundIndex !== state.roundIndex || stepIndex !== state.stepIndex;
      const hasNext =
        roundIndex < timelines.length - 1 ||
        stepIndex < timelines[roundIndex].length - 1;
      return {
        ...state,
        roundIndex,
        stepIndex,
        playing:
          Boolean(action.automatic) && position.kind === "table" && hasNext,
        animateDraw: action.delta === 1 && moved && position.kind === "table",
      };
    }
    case "round":
      return {
        ...state,
        roundIndex: action.index,
        stepIndex: 0,
        playing: false,
        animateDraw: false,
      };
    case "seat":
      return {
        ...state,
        seat: action.seat,
        playing: false,
        animateDraw: false,
      };
    case "seek":
      return {
        ...state,
        stepIndex: action.index,
        playing: false,
        animateDraw: false,
      };
    case "reveal":
      return { ...state, reveal: action.reveal, animateDraw: false };
    case "play": {
      const positions = action.timelines[state.roundIndex];
      const hasNext =
        state.roundIndex < action.timelines.length - 1 ||
        state.stepIndex < positions.length - 1;
      return {
        ...state,
        playing:
          positions[state.stepIndex].kind === "table" &&
          hasNext &&
          !state.playing,
      };
    }
    case "pause":
      return { ...state, playing: false };
  }
}
