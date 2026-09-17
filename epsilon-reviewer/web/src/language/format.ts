import type { Model, ReviewRound } from "../api/types";
import type { Translate } from "./catalog";

export const modelKey = (model: Pick<Model, "modelId" | "revision">) =>
  JSON.stringify([model.modelId, model.revision]);
export const modelLabel = (model: Model) =>
  `${model.displayName} ${model.version}`.trim();
export function sourceLabel(source: string, t: Translate) {
  const key = source.toLowerCase();
  return key === "tenhou"
    ? t("source.tenhou")
    : key === "mjai"
      ? t("source.mjai")
      : key === "majsoul" || key === "mahjongsoul"
        ? t("source.majsoul")
        : source;
}
export const playerName = (name: string, seat: number, t: Translate) =>
  name || t("common.player", { number: seat + 1 });
const windKeys = ["east", "south", "west", "north"] as const;
export const windName = (wind: number, t: Translate, short = false) =>
  t(`mahjong.${windKeys[wind % 4]}${short ? "Short" : ""}`);
export const seatName = (seat: number, dealer: number, t: Translate) =>
  t("mahjong.seat", { wind: windName((seat - dealer + 4) % 4, t) });
export const roundName = (round: ReviewRound, t: Translate, short = false) =>
  t(short ? "mahjong.roundShort" : "mahjong.round", {
    wind: windName(Math.floor(round.roundIndex / 4), t, short),
    number: (round.roundIndex % 4) + 1,
  });
export function dateLabel(date: string, locale: string, t: Translate) {
  const value = new Date(date);
  return Number.isNaN(value.getTime())
    ? t("common.unknownDate")
    : new Intl.DateTimeFormat(locale, {
        month: "numeric",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }).format(value);
}
