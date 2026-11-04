import type { ReactNode } from "react";
import { X } from "lucide-react";
import { Tile, tileName } from "../../components/Tile";
import { playerName, seatName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import {
  SCORE_STEP,
  STICK_LIMIT,
  targetKey,
  tileChoices,
  type ScenarioTarget,
  type TileChoice,
  type TileTarget,
} from "./scenario/scenario.ts";
import type { Scenario } from "./scenario/useScenario.ts";
import type { Playback } from "./usePlayback.ts";

const SCORE_STEPS = [
  -10 * SCORE_STEP,
  -SCORE_STEP,
  SCORE_STEP,
  10 * SCORE_STEP,
];

interface EditorProps<T extends ScenarioTarget> {
  target: T;
  scenario: Scenario;
  playback: Playback;
}

/** 編集の中身の枠。見出し・元の値・閉じる、中身、元に戻す。 */
function EditorFrame({
  title,
  was,
  target,
  scenario,
  children,
}: {
  title: string;
  was: string;
  target: ScenarioTarget;
  scenario: Scenario;
  children: ReactNode;
}) {
  const { t } = useLanguage();
  const key = targetKey(target);
  const edited = scenario.changes.some((change) => targetKey(change) === key);
  return (
    <>
      <div className="em-scenario-editor-head">
        <strong>{title}</strong>
        <span className="em-scenario-was">
          {t("scenario.was", { value: was })}
        </span>
        <button
          className="em-scenario-close"
          aria-label={t("scenario.close")}
          title={t("scenario.close")}
          onClick={() => scenario.select(null)}
        >
          <X aria-hidden="true" />
        </button>
      </div>
      {children}
      <button
        className="em-scenario-restore"
        disabled={!edited}
        onClick={() => scenario.revert(target)}
      >
        {t("scenario.restore")}
      </button>
    </>
  );
}

/** 牌を選び直す一覧。選んだら吹き出しを閉じる。 */
function TilePalette({
  target,
  scenario,
  playback,
  onPick,
}: EditorProps<TileTarget> & { onPick: (tile: string) => void }) {
  const { t } = useLanguage();
  const choices = tileChoices(scenario.edited, playback.seat, target);
  const pick = (tile: string) => {
    onPick(tile);
    scenario.select(null);
  };
  return (
    <div
      className="em-scenario-palette"
      role="group"
      aria-label={t("scenario.palette")}
    >
      {choices.map((row, index) => (
        <div key={index} className="em-scenario-palette-row">
          {row.map((choice) => (
            <PaletteTile key={choice.tile} choice={choice} onPick={pick} />
          ))}
        </div>
      ))}
    </div>
  );
}

function PaletteTile({
  choice,
  onPick,
}: {
  choice: TileChoice;
  onPick: (tile: string) => void;
}) {
  const { t } = useLanguage();
  const tile = tileName(choice.tile, t);
  const label = choice.current
    ? t("scenario.tileCurrent", { tile })
    : choice.disabled
      ? t("scenario.tileFull", { tile })
      : t("scenario.tileVisible", { tile, count: choice.visible });
  return (
    <button
      className="em-scenario-tile"
      aria-pressed={choice.current}
      disabled={choice.disabled}
      aria-label={label}
      title={label}
      onClick={() => onPick(choice.tile)}
    >
      <Tile tile={choice.tile} />
    </button>
  );
}

export function RiverEditor({
  target,
  scenario,
  playback,
}: EditorProps<Extract<ScenarioTarget, { kind: "river" }>>) {
  const { t } = useLanguage();
  const tile =
    scenario.edited.snapshot.players[target.seat]!.river[target.index]!;
  const original = scenario.original.players[target.seat]!.river[target.index]!;
  const style = (tsumogiri: boolean) =>
    t(tsumogiri ? "scenario.tsumogiri" : "scenario.tedashi");
  const name = playerName(
    scenario.original.players[target.seat]!.name,
    target.seat,
    t,
  );
  const turn = t("scenario.turn", { name, turn: target.index + 1 });
  // 自分の打牌の手出し・ツモ切りは、その牌が入った所（配牌かツモ）で決まるので変えない。
  const ownRiver = target.seat === playback.seat;
  return (
    <EditorFrame
      title={
        original.riichi
          ? t("scenario.withRiichi", {
              title: turn,
              riichi: t("scenario.riichiTile"),
            })
          : turn
      }
      was={t("scenario.tileStyle", {
        tile: tileName(original.tile, t),
        style: style(original.tsumogiri === true),
      })}
      target={target}
      scenario={scenario}
    >
      {!ownRiver && (
        <div
          className="em-scenario-segment"
          role="group"
          aria-label={t("scenario.discardStyle")}
        >
          {[false, true].map((tsumogiri) => (
            <button
              key={String(tsumogiri)}
              aria-pressed={(tile.tsumogiri === true) === tsumogiri}
              onClick={() =>
                scenario.setRiver(target.seat, target.index, { tsumogiri })
              }
            >
              {style(tsumogiri)}
            </button>
          ))}
        </div>
      )}
      <TilePalette
        target={target}
        scenario={scenario}
        playback={playback}
        onPick={(picked) =>
          scenario.setRiver(target.seat, target.index, { tile: picked })
        }
      />
    </EditorFrame>
  );
}

export function DoraEditor({
  target,
  scenario,
  playback,
}: EditorProps<Extract<ScenarioTarget, { kind: "dora" }>>) {
  const { t } = useLanguage();
  return (
    <EditorFrame
      title={
        scenario.edited.snapshot.dora.length > 1
          ? t("scenario.doraNumbered", { number: target.index + 1 })
          : t("scenario.dora")
      }
      was={tileName(scenario.original.dora[target.index]!, t)}
      target={target}
      scenario={scenario}
    >
      <TilePalette
        target={target}
        scenario={scenario}
        playback={playback}
        onPick={(picked) => scenario.setDora(target.index, picked)}
      />
    </EditorFrame>
  );
}

/** 自分の手牌の 1 枚。見出しは今の並びの何枚目か、ツモ牌。 */
export function HandEditor({
  target,
  scenario,
  playback,
}: EditorProps<Extract<ScenarioTarget, { kind: "hand" }>>) {
  const { t } = useLanguage();
  const original = scenario.original.players[playback.seat]!;
  const draw = target.index === original.hand.length;
  return (
    <EditorFrame
      title={
        draw
          ? t("scenario.drawTile")
          : t("scenario.handTile", {
              number: scenario.edited.handOrigins.indexOf(target.index) + 1,
            })
      }
      was={tileName(draw ? original.draw! : original.hand[target.index]!, t)}
      target={target}
      scenario={scenario}
    >
      <TilePalette
        target={target}
        scenario={scenario}
        playback={playback}
        onPick={(picked) => scenario.setHand(target.index, picked)}
      />
    </EditorFrame>
  );
}

export function ScoreEditor({
  target,
  scenario,
  playback,
}: EditorProps<Extract<ScenarioTarget, { kind: "score" }>>) {
  const { t, locale } = useLanguage();
  const value = scenario.edited.snapshot.players[target.seat]!.score;
  const original = scenario.original.players[target.seat]!.score;
  const amount = (points: number) => Math.abs(points).toLocaleString(locale);
  return (
    <EditorFrame
      title={t("scenario.score", {
        name: playerName(
          scenario.original.players[target.seat]!.name,
          target.seat,
          t,
        ),
        seat: seatName(target.seat, playback.round.dealer, t),
      })}
      was={original.toLocaleString(locale)}
      target={target}
      scenario={scenario}
    >
      <div className="em-scenario-number">{value.toLocaleString(locale)}</div>
      <div className="em-scenario-steps">
        {SCORE_STEPS.map((step) => (
          <button
            key={step}
            aria-label={t(
              step < 0 ? "scenario.decrease" : "scenario.increase",
              {
                amount: amount(step),
              },
            )}
            onClick={() => scenario.setScore(target.seat, value + step)}
          >
            {t(step < 0 ? "scenario.minus" : "scenario.plus", {
              amount: amount(step),
            })}
          </button>
        ))}
      </div>
    </EditorFrame>
  );
}

export function SticksEditor({
  target,
  scenario,
  playback,
}: EditorProps<Extract<ScenarioTarget, { kind: "honba" | "kyotaku" }>>) {
  const { t } = useLanguage();
  const value = scenario.edited.snapshot[target.kind];
  // この局で判断までに出た立直の棒は供託から外せない（局の始めの供託を負にしない）。
  const minimum =
    target.kind === "kyotaku"
      ? scenario.original.kyotaku - playback.roundStart.kyotaku
      : 0;
  return (
    <EditorFrame
      title={t(target.kind === "honba" ? "scenario.honba" : "scenario.kyotaku")}
      was={String(scenario.original[target.kind])}
      target={target}
      scenario={scenario}
    >
      <div className="em-scenario-steps">
        <button
          aria-label={t("scenario.decrease", { amount: 1 })}
          disabled={value <= minimum}
          onClick={() => scenario.setSticks(target.kind, value - 1)}
        >
          {t("scenario.minus", { amount: 1 })}
        </button>
        <span className="em-scenario-number">{value}</span>
        <button
          aria-label={t("scenario.increase", { amount: 1 })}
          disabled={value >= STICK_LIMIT}
          onClick={() => scenario.setSticks(target.kind, value + 1)}
        >
          {t("scenario.plus", { amount: 1 })}
        </button>
      </div>
    </EditorFrame>
  );
}
