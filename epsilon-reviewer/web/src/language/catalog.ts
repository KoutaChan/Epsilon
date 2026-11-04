import type errors from "../../../language/ja/errors.json";
import type mahjong from "../../../language/ja/mahjong.json";
import type replay from "../../../language/ja/replay.json";
import type shell from "../../../language/ja/shell.json";
import { isErrorCode } from "../../../protocol/errors.ts";

// 画面の文言は language/<言語>/ の shell・replay・mahjong・errors（desktop.json はメインプロセスだけが読む）。
// 鍵の型は ja のファイルで、どの言語も同じ鍵を持つことは tools/check-language.mjs が確かめる。

export type MessageKey =
  | keyof typeof shell
  | keyof typeof replay
  | keyof typeof mahjong
  | keyof typeof errors;

/** 文言の {名前} を values で置き換える。値の無い {名前} はそのまま残す。 */
export type Translate = (
  key: MessageKey,
  values?: Record<string, string | number>,
) => string;

const PLACEHOLDER = /\{(\w+)\}/g;

const files = import.meta.glob<Record<string, string>>(
  "../../../language/*/{shell,replay,mahjong,errors}.json",
  { eager: true, import: "default" },
);
const catalogs = new Map<string, Record<string, string>>();
for (const [path, messages] of Object.entries(files)) {
  const id = path.split("/").at(-2)!;
  catalogs.set(id, Object.assign(catalogs.get(id) ?? {}, messages));
}

/** 選べる言語（language/ のフォルダーの順）。name は language.name の文言。 */
export const languages: readonly { id: string; name: string }[] = [
  ...catalogs,
].map(([id, messages]) => ({ id, name: messages["language.name"]! }));

const translators = new Map(
  [...catalogs].map(([id, messages]) => [id, translatorOf(messages)]),
);

export function isLanguage(id: string): boolean {
  return translators.has(id);
}

/** 言語ごとに 1 回だけ作った Translate。 */
export function translator(language: string): Translate {
  const translate = translators.get(language);
  if (translate === undefined)
    throw new RangeError(`Unknown language: ${language}`);
  return translate;
}

function translatorOf(messages: Readonly<Record<string, string>>): Translate {
  return (key, values) => {
    // 鍵は型で決まるが、牌譜の値から作る鍵が文言に無くても画面を落とさない。
    const message = messages[key] ?? key;
    if (values === undefined) return message;
    return message.replace(PLACEHOLDER, (token, name: string) =>
      Object.hasOwn(values, name) ? String(values[name]) : token,
    );
  };
}

/** 失敗の文言の鍵。符号を持たない・知らない符号は errors.operation_failed。 */
export function errorKey(error: unknown): MessageKey {
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? error.code
      : undefined;
  return isErrorCode(code) ? `errors.${code}` : "errors.operation_failed";
}
