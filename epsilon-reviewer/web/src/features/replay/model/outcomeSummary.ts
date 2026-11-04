import type { WinDetails } from "../../../../../protocol/result.ts";
import type {
  EventSnapshot,
  PlayerSnapshot,
  ReplayModel,
  Snapshot,
} from "./types.ts";

export interface WinningHand {
  readonly player: PlayerSnapshot;
  readonly method: "tsumo" | "ron";
  readonly target: number;
  readonly hand: readonly string[];
  readonly winningTile: string;
  /** 牌譜に採点の内訳が無ければ null。 */
  readonly details: WinDetails | null;
}

export function collectWinningHands(
  outcomes: readonly EventSnapshot[],
): readonly WinningHand[] {
  return outcomes.flatMap<WinningHand>((step) => {
    if (step.eventType !== "hora") return [];
    const { outcome } = step;
    const player = step.players[outcome.actor]!;
    return [
      {
        player,
        method: outcome.target === outcome.actor ? "tsumo" : "ron",
        target: outcome.target,
        hand: player.hand,
        winningTile: outcome.winningTile,
        details: outcome.details ?? null,
      },
    ];
  });
}

// リーチ棒の支払いを含む局全体の増減と、和了単体の収支を区別する。
export function roundScoreChanges(
  start: Snapshot,
  settled: Snapshot,
): readonly number[] {
  return settled.players.map(
    (player) => player.score - start.players[player.seat]!.score,
  );
}

export function rankFinalPlayers(
  players: readonly PlayerSnapshot[],
  finalScores: readonly number[] | undefined,
) {
  const ordered = players
    .map((player) => ({
      player,
      score: finalScores?.[player.seat] ?? player.score,
    }))
    .sort(
      (left, right) =>
        right.score - left.score || left.player.seat - right.player.seat,
    );
  return ordered.map((entry) => ({
    ...entry,
    rank: ordered.findIndex((other) => other.score === entry.score) + 1,
  }));
}

export function hasRecordedMatchEnd(model: ReplayModel): boolean {
  return (
    model.record.finalScores !== undefined ||
    model.rounds
      .at(-1)!
      .steps.some((step) => !step.decision && step.eventType === "end_game")
  );
}
