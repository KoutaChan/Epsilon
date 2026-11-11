import type {
  ModelInfo,
  ModelKey,
  RecordSource,
} from "../../../protocol/model.ts";
import type { Translate } from "./catalog.ts";

const WINDS = ["east", "south", "west", "north"] as const;
const DATE_OPTIONS: Intl.DateTimeFormatOptions = {
  month: "numeric",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
};
const dateFormats = new Map<string, Intl.DateTimeFormat>();

/** モデルとリビジョンを区別するキー（select の値や比較に使う）。 */
export function modelKey(model: ModelKey): string {
  return JSON.stringify([model.modelId, model.revision]);
}

export function modelLabel(model: ModelInfo): string {
  return `${model.displayName} ${model.version}`.trim();
}

export function sourceLabel(source: RecordSource, t: Translate): string {
  return t(`source.${source}`);
}

/** 対局者名を返す。名前の無い席は「プレイヤー n」（n は席の番号 + 1）。 */
export function playerName(name: string, seat: number, t: Translate): string {
  return name || t("common.player", { number: seat + 1 });
}

/** 風の名前。風の番号は 0 が東。 */
export function windName(wind: number, t: Translate, short = false): string {
  const name = WINDS[wind % 4]!;
  return t(short ? `mahjong.${name}Short` : `mahjong.${name}`);
}

/** 親から数えた席の風（東家〜北家。三人打ちは西家まで）。 */
export function seatName(
  seat: number,
  round: { dealer: number; seats: number },
  t: Translate,
): string {
  const wind = (seat - round.dealer + round.seats) % round.seats;
  return t("mahjong.seat", { wind: windName(wind, t) });
}

/** 局の名前。局の番号（`roundIndex`）は (場風) × 席の数 + (局 − 1)。 */
export function roundName(
  round: { roundIndex: number; seats: number },
  t: Translate,
  short = false,
): string {
  return t(short ? "mahjong.roundShort" : "mahjong.round", {
    wind: windName(Math.floor(round.roundIndex / round.seats), t, short),
    number: (round.roundIndex % round.seats) + 1,
  });
}

/** 月日と時刻の表示。読めない日時は common.unknownDate の文言にする。書式はロケールごとに 1 つ作って使い回す。 */
export function dateLabel(date: string, locale: string, t: Translate): string {
  const value = new Date(date);
  if (Number.isNaN(value.getTime())) return t("common.unknownDate");
  let format = dateFormats.get(locale);
  if (format === undefined) {
    format = new Intl.DateTimeFormat(locale, DATE_OPTIONS);
    dateFormats.set(locale, format);
  }
  return format.format(value);
}
