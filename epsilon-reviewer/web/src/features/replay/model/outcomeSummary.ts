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
  /** 採点の内訳。牌譜に内訳が無ければ null。 */
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

// 局の始めから精算の後までの点数の増減を、席ごとに返す。立直棒の支払いも含むので、和了の収支だけとは合わない。
export function roundScoreChanges(
  start: Snapshot,
  settled: Snapshot,
): readonly number[] {
  return settled.players.map(
    (player) => player.score - start.players[player.seat]!.score,
  );
}

/** 最終の点数の高い順に並べ、1 から順位を付ける。同点は起家に近い席（席の番号の小さい方）を上にし、順位は重ねない。 */
export function rankFinalPlayers(
  players: readonly PlayerSnapshot[],
  finalScores: readonly number[] | undefined,
) {
  return players
    .map((player) => ({
      player,
      score: finalScores?.[player.seat] ?? player.score,
    }))
    .sort(
      (left, right) =>
        right.score - left.score || left.player.seat - right.player.seat,
    )
    .map((entry, index) => ({ ...entry, rank: index + 1 }));
}

export function hasRecordedMatchEnd(model: ReplayModel): boolean {
  return (
    model.record.finalScores !== undefined ||
    model.rounds
      .at(-1)!
      .steps.some((step) => !step.decision && step.eventType === "end_game")
  );
}
