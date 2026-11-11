import type errors from "../../../language/ja/errors.json";
import type mahjong from "../../../language/ja/mahjong.json";
import type replay from "../../../language/ja/replay.json";
import type shell from "../../../language/ja/shell.json";
import { isErrorCode } from "../../../protocol/errors.ts";

// 画面の文言は language/<言語>/ の shell・replay・mahjong・errors にある（desktop.json はメインプロセスだけが読む）。
// キーの型は ja のファイルから作る。どの言語も同じキーを持つことは tools/check-language.mjs が確かめる。

export type MessageKey =
  | keyof typeof shell
  | keyof typeof replay
  | keyof typeof mahjong
  | keyof typeof errors;

/** 文言の中の {名前} を、渡した値で置き換える。値の無い {名前} はそのまま残す。 */
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

/** 選べる言語（language/ のフォルダーの順）。名前には、その言語の language.name の文言を使う。 */
export const languages: readonly { id: string; name: string }[] = [
  ...catalogs,
].map(([id, messages]) => ({ id, name: messages["language.name"]! }));

const translators = new Map(
  [...catalogs].map(([id, messages]) => [id, translatorOf(messages)]),
);

export function isLanguage(id: string): boolean {
  return translators.has(id);
}

/** その言語の Translate を返す。Translate は言語ごとに 1 回だけ作る。 */
export function translator(language: string): Translate {
  const translate = translators.get(language);
  if (translate === undefined)
    throw new RangeError(`Unknown language: ${language}`);
  return translate;
}

function translatorOf(messages: Readonly<Record<string, string>>): Translate {
  return (key, values) => {
    // キーは型で決まるが、牌譜の値から作ったキーは文言に無いことがある。そのときも画面を落とさず、キーをそのまま出す。
    const message = messages[key] ?? key;
    if (values === undefined) return message;
    return message.replace(PLACEHOLDER, (token, name: string) =>
      Object.hasOwn(values, name) ? String(values[name]) : token,
    );
  };
}

/** エラーを表す文言のキーを返す。エラーコードが無いか、知らないコードなら errors.operation_failed。 */
export function errorKey(error: unknown): MessageKey {
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? error.code
      : undefined;
  return isErrorCode(code) ? `errors.${code}` : "errors.operation_failed";
}
