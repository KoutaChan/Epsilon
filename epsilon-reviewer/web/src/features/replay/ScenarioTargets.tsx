import type { KeyboardEvent, ReactNode } from "react";
import { tileName } from "../../components/Tile";
import { playerName, seatName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { ScreenPoint, TableTargets } from "./renderer/types.ts";
import {
  canEditHand,
  canEditRiver,
  handOrigin,
  targetKey,
  type ScenarioTarget,
} from "./scenario/scenario.ts";
import type { Scenario } from "./scenario/useScenario.ts";
import type { Playback } from "./usePlayback.ts";

/** 押せる所の共通の印。変えた所は薄紫、選んでいる所は白の縁。編集の吹き出しは data-scenario-key で位置を合わせる。 */
function useTargetState(
  scenario: Scenario,
  target: ScenarioTarget,
  label: string,
) {
  const { t } = useLanguage();
  const key = targetKey(target);
  const edited = scenario.changes.some((change) => targetKey(change) === key);
  const description = edited ? t("scenario.changed", { target: label }) : label;
  return {
    "data-scenario-key": key,
    "data-edited": edited,
    "aria-pressed":
      scenario.selection !== null && targetKey(scenario.selection) === key,
    "aria-label": description,
    description,
  };
}

/** 卓の外の DOM（ドラ表示牌・供託・本場）を押せる所にする。 */
export function ScenarioPick({
  scenario,
  target,
  label,
  className,
  children,
}: {
  scenario: Scenario;
  target: ScenarioTarget;
  label: string;
  className: string;
  children?: ReactNode;
}) {
  const { description, ...state } = useTargetState(scenario, target, label);
  return (
    <button
      type="button"
      className={`em-scenario-pick ${className}`}
      {...state}
      title={description}
      onClick={() => scenario.select(target)}
    >
      {children}
    </button>
  );
}

function TargetShape({
  scenario,
  target,
  points,
  label,
}: {
  scenario: Scenario;
  target: ScenarioTarget;
  points: readonly ScreenPoint[];
  label: string;
}) {
  const { description, ...state } = useTargetState(scenario, target, label);
  const choose = () => scenario.select(target);
  return (
    <polygon
      points={points.map((point) => point.join(",")).join(" ")}
      role="button"
      tabIndex={0}
      {...state}
      onClick={choose}
      onKeyDown={(event: KeyboardEvent) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        choose();
      }}
    >
      <title>{description}</title>
    </polygon>
  );
}

/**
 * 3D の卓の上に、中央の点数・河の牌・自分の手牌の輪郭を当たり枠として重ねる。形は描画が送る。手牌は卓の上に
 * 描くので最後に重ねる。
 */
export function ScenarioTargets({
  scenario,
  targets,
  playback,
}: {
  scenario: Scenario;
  targets: TableTargets;
  playback: Playback;
}) {
  const { t, locale } = useLanguage();
  const { seat, round, decision } = playback;
  const { snapshot, handOrigins } = scenario.edited;
  const viewer = snapshot.players[seat]!;
  const name = (target: number) =>
    playerName(snapshot.players[target]!.name, target, t);
  const label = (target: string, value: string) =>
    t("scenario.targetValue", { target, value });
  return (
    <svg
      className="em-scenario-targets"
      role="group"
      aria-label={t("scenario.targets")}
    >
      {targets.scores.map((shape) => (
        <TargetShape
          key={`score:${shape.seat}`}
          scenario={scenario}
          target={{ kind: "score", seat: shape.seat }}
          points={shape.points}
          label={label(
            t("scenario.score", {
              name: name(shape.seat),
              seat: seatName(shape.seat, round.dealer, t),
            }),
            t("scenario.points", {
              points:
                snapshot.players[shape.seat]!.score.toLocaleString(locale),
            }),
          )}
        />
      ))}
      {targets.rivers
        .filter((shape) => canEditRiver(snapshot, shape.seat, shape.index))
        .map((shape) => {
          const tile = snapshot.players[shape.seat]!.river[shape.index]!;
          return (
            <TargetShape
              key={`river:${shape.seat}:${shape.index}`}
              scenario={scenario}
              target={{ kind: "river", seat: shape.seat, index: shape.index }}
              points={shape.points}
              label={label(
                t("scenario.turn", {
                  name: name(shape.seat),
                  turn: shape.index + 1,
                }),
                t("scenario.tileStyle", {
                  tile: tileName(tile.tile, t),
                  style: t(
                    tile.tsumogiri ? "scenario.tsumogiri" : "scenario.tedashi",
                  ),
                }),
              )}
            />
          );
        })}
      {canEditHand(snapshot, seat, decision) &&
        targets.hands.map((shape) => {
          const draw = shape.index === viewer.hand.length;
          return (
            <TargetShape
              key={`hand:${shape.index}`}
              scenario={scenario}
              target={{
                kind: "hand",
                index: handOrigin(handOrigins, shape.index),
              }}
              points={shape.points}
              label={label(
                draw
                  ? t("scenario.drawTile")
                  : t("scenario.handTile", { number: shape.index + 1 }),
                tileName(draw ? viewer.draw! : viewer.hand[shape.index]!, t),
              )}
            />
          );
        })}
    </svg>
  );
}
