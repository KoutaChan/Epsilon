import type {
  PlayerSnapshot,
  ReviewResult,
  Snapshot,
  WinDetails,
} from "../../api/types";

export interface WinningHand {
  player: PlayerSnapshot;
  method: "tsumo" | "ron";
  target: number;
  hand: readonly string[];
  winningTile: string;
  details: WinDetails | null;
}

export function collectWinningHands(
  outcomes: readonly Snapshot[],
): readonly WinningHand[] {
  return outcomes.flatMap<WinningHand>((step) => {
    const outcome = step.outcome!;
    if (outcome.type !== "hora") return [];
    const player = step.players[outcome.actor];
    return [
      {
        player,
        method: outcome.target === outcome.actor ? "tsumo" : "ron",
        target: outcome.target,
        hand: player.hand,
        winningTile: outcome.winningTile,
        details: outcome.details,
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
    (player) => player.score - start.players[player.seat].score,
  );
}

export function rankFinalPlayers(
  players: readonly PlayerSnapshot[],
  finalScores: readonly number[] | null,
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

export function hasRecordedMatchEnd(result: ReviewResult): boolean {
  return (
    result.metadata.finalScores != null ||
    result.rounds.at(-1)!.steps.some((step) => step.eventType === "end_game")
  );
}
