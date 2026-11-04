import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type desktopMessages from "../../language/ja/desktop.json";
import type shellMessages from "../../language/ja/shell.json";

/** メインプロセスの文言の鍵（ja のファイルの鍵）。 */
export type MessageKey =
  keyof typeof shellMessages | keyof typeof desktopMessages;

/** メインプロセスが読む言語のファイル（窓の題とネイティブのダイアログ）。 */
const FILES = ["shell.json", "desktop.json"] as const;

/** 元の言語。どの言語にも無い鍵はこの言語の文言を使う。 */
const SOURCE_LANGUAGE = "ja";

/** language/<id>/ の文言。id はフォルダーの名前で、ファイルの無い言語の鍵は ja へ戻る。 */
export class Languages {
  readonly #messages: ReadonlyMap<string, ReadonlyMap<string, string>>;

  private constructor(
    messages: ReadonlyMap<string, ReadonlyMap<string, string>>,
  ) {
    this.#messages = messages;
  }

  /** @throws Error フォルダーが無い、ja が無い、または JSON が壊れている */
  static load(directory: string): Languages {
    const messages = new Map<string, ReadonlyMap<string, string>>();
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const merged = new Map<string, string>();
      for (const file of FILES) {
        const source = path.join(directory, entry.name, file);
        let text: string;
        try {
          text = readFileSync(source, "utf8");
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
          throw error;
        }
        for (const [key, value] of Object.entries(
          JSON.parse(text) as Record<string, string>,
        ))
          merged.set(key, value);
      }
      messages.set(entry.name, merged);
    }
    if (!messages.has(SOURCE_LANGUAGE))
      throw new Error(`The language folder ${SOURCE_LANGUAGE} is missing.`);
    return new Languages(messages);
  }

  has(language: string): boolean {
    return this.#messages.has(language);
  }

  /** OS の言語（例 en-US）に合う id。同じ id、無ければ - の前、それも無ければ ja。 */
  select(locale: string): string {
    if (this.has(locale)) return locale;
    const base = locale.split("-")[0] ?? "";
    return this.has(base) ? base : SOURCE_LANGUAGE;
  }

  /** @throws Error ja にも鍵が無い */
  message(language: string, key: MessageKey): string {
    const text =
      this.#messages.get(language)?.get(key) ??
      this.#messages.get(SOURCE_LANGUAGE)?.get(key);
    if (text === undefined) throw new Error(`Missing language key: ${key}`);
    return text;
  }
}
