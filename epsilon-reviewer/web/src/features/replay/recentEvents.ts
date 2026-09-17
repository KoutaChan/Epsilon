import type { PlayerSnapshot, Snapshot } from "../../api/types";
import type { MessageKey } from "../../language/catalog";
import type { ReplayStage } from "./timeline";

export interface RecentReplayEvent {
  kind: ReplayStage;
  event: Snapshot;
  previous: Snapshot | null;
}

export interface ReplayEventDetails {
  player: PlayerSnapshot | null;
  action: MessageKey;
  tiles: readonly string[];
}

const eventMessageKeys: Record<string, MessageKey> = {
  start_game: "event.start_game",
  start_kyoku: "event.start_kyoku",
  end_kyoku: "event.end_kyoku",
  end_game: "event.end_game",
  reach: "event.reach",
  reach_accepted: "event.reach_accepted",
  none: "event.none",
  ryukyoku: "event.ryukyoku",
  chi: "action.chi",
  pon: "action.pon",
  daiminkan: "action.daiminkan",
  ankan: "action.ankan",
  kakan: "action.kakan",
  dora: "event.dora",
};

// 表示する実イベントから牌を取り出す。加槓は直前の局面と比較して対象の副露を特定する。
export function describeReplayEvent(
  step: Snapshot,
  previous: Snapshot | null,
  reveal: boolean,
  seat: number,
): ReplayEventDetails {
  const player = step.actor >= 0 ? step.players[step.actor] : null;
  if (step.outcome?.type === "hora") {
    return {
      player,
      action:
        step.outcome.actor === step.outcome.target
          ? "action.tsumo"
          : "action.ron",
      tiles: [step.outcome.winningTile],
    };
  }
  switch (step.eventType) {
    case "tsumo":
      return {
        player,
        action: "replay.draw",
        tiles: [reveal || step.actor === seat ? player!.draw! : "back"],
      };
    case "dahai": {
      const discard = player!.river.at(-1)!;
      return {
        player,
        action: discard.riichi
          ? discard.tsumogiri
            ? "action.riichiTsumogiri"
            : "action.riichi"
          : discard.tsumogiri
            ? "action.dahaiTsumogiri"
            : "action.dahai",
        tiles: [discard.tile],
      };
    }
    case "chi":
    case "pon":
    case "daiminkan":
    case "ankan":
      return {
        player,
        action: eventMessageKeys[step.eventType],
        tiles: player!.melds.at(-1)!.tiles,
      };
    case "kakan": {
      const before = previous!.players[step.actor].melds;
      const meld = player!.melds.find(
        (value, index) =>
          value.type === "KAKAN" && before[index].type === "PON",
      )!;
      return { player, action: "action.kakan", tiles: meld.tiles };
    }
    case "dora":
      return { player, action: "event.dora", tiles: [step.dora.at(-1)!] };
    default:
      return { player, action: eventMessageKeys[step.eventType], tiles: [] };
  }
}
