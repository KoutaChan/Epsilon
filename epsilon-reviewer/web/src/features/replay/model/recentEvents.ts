import type { MessageKey } from "../../../language/catalog.ts";
import type { ReplayStage } from "./timeline.ts";
import type { EventSnapshot, PlayerSnapshot, Snapshot } from "./types.ts";

export interface RecentReplayEvent {
  readonly kind: ReplayStage;
  readonly event: EventSnapshot;
  /** 加槓の直前の局面。ほかのイベントでは null。 */
  readonly previous: Snapshot | null;
}

export interface ReplayEventDetails {
  readonly player: PlayerSnapshot | null;
  readonly action: MessageKey;
  readonly tiles: readonly string[];
}

const EVENT_MESSAGES = {
  start_kyoku: "event.start_kyoku",
  end_kyoku: "event.end_kyoku",
  end_game: "event.end_game",
  reach: "event.reach",
  reach_accepted: "event.reach_accepted",
  none: "event.none",
  ryukyoku: "event.ryukyoku",
} as const satisfies Record<string, MessageKey>;

// 表示するイベントから牌を取り出す。加槓では、直前の局面と比べて、どの副露に牌を足したかを調べる。
export function describeReplayEvent(
  step: EventSnapshot,
  previous: Snapshot | null,
  reveal: boolean,
  seat: number,
): ReplayEventDetails {
  const player = step.actor >= 0 ? step.players[step.actor]! : null;
  switch (step.eventType) {
    case "hora":
      return {
        player,
        action:
          step.outcome.actor === step.outcome.target
            ? "action.tsumo"
            : "action.ron",
        tiles: [step.outcome.winningTile],
      };
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
      // 抜いた北は副露の列で面子より後ろに並ぶので、北を除いた最後の副露が新しい面子になる。
      return {
        player,
        action: `action.${step.eventType}`,
        tiles: player!.melds.filter((meld) => meld.type !== "NUKI").at(-1)!.tiles,
      };
    case "kita":
      return { player, action: "action.nuki", tiles: ["N"] };
    case "kakan": {
      const before = previous!.players[step.actor]!.melds;
      const meld = player!.melds.find(
        (value, index) =>
          value.type === "KAKAN" && before[index]?.type === "PON",
      )!;
      return { player, action: "action.kakan", tiles: meld.tiles };
    }
    case "dora":
      return { player, action: "event.dora", tiles: [step.dora.at(-1)!] };
    default:
      return { player, action: EVENT_MESSAGES[step.eventType], tiles: [] };
  }
}
