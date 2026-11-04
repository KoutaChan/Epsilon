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

/** モデルの版の鍵（select の値・比べ方）。 */
export function modelKey(model: ModelKey): string {
  return JSON.stringify([model.modelId, model.revision]);
}

export function modelLabel(model: ModelInfo): string {
  return `${model.displayName} ${model.version}`.trim();
}

export function sourceLabel(source: RecordSource, t: Translate): string {
  return t(`source.${source}`);
}

/** 名前の無い席は「プレイヤー n」。 */
export function playerName(name: string, seat: number, t: Translate): string {
  return name || t("common.player", { number: seat + 1 });
}

/** 風の名前。wind は 0 が東。 */
export function windName(wind: number, t: Translate, short = false): string {
  const name = WINDS[wind % 4]!;
  return t(short ? `mahjong.${name}Short` : `mahjong.${name}`);
}

/** 親から数えた席の風（東家…北家）。 */
export function seatName(seat: number, dealer: number, t: Translate): string {
  return t("mahjong.seat", { wind: windName((seat - dealer + 4) % 4, t) });
}

/** 局の名前。roundIndex は (場風) × 4 + (局 − 1)。 */
export function roundName(
  round: { roundIndex: number },
  t: Translate,
  short = false,
): string {
  return t(short ? "mahjong.roundShort" : "mahjong.round", {
    wind: windName(Math.floor(round.roundIndex / 4), t, short),
    number: (round.roundIndex % 4) + 1,
  });
}

/** 月日と時刻。読めない日時は common.unknownDate。書式は言語ごとに 1 つ作り置く。 */
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
