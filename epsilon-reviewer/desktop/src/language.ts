import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import type desktopMessages from "../../language/ja/desktop.json";
import type shellMessages from "../../language/ja/shell.json";

/** メインプロセスで使う文言のキー。ja のファイルにあるキーと同じ。 */
export type MessageKey =
  keyof typeof shellMessages | keyof typeof desktopMessages;

/** メインプロセスが読む言語のファイル。ウィンドウのタイトルとネイティブのダイアログの文言が入る。 */
const FILES = ["shell.json", "desktop.json"] as const;

/** 基準の言語。ほかの言語のファイルに無いキーは、この言語の文言で補う。 */
const SOURCE_LANGUAGE = "ja";

/** language/<id>/ に置いた、言語ごとの文言。id はフォルダーの名前。その言語のファイルに無いキーは ja の文言を使う。 */
export class Languages {
  readonly #messages: ReadonlyMap<string, ReadonlyMap<string, string>>;

  private constructor(
    messages: ReadonlyMap<string, ReadonlyMap<string, string>>,
  ) {
    this.#messages = messages;
  }

  /** @throws Error フォルダーが無いとき、ja が無いとき、または JSON が壊れているとき */
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

  /**
   * OS の言語（en-US など）に合う言語の id を返す。同じ id があればそれを、無ければ "-" より前の部分を、
   * それも無ければ ja を返す。
   */
  select(locale: string): string {
    if (this.has(locale)) return locale;
    const base = locale.split("-")[0] ?? "";
    return this.has(base) ? base : SOURCE_LANGUAGE;
  }

  /** @throws Error ja にもキーが無いとき */
  message(language: string, key: MessageKey): string {
    const text =
      this.#messages.get(language)?.get(key) ??
      this.#messages.get(SOURCE_LANGUAGE)?.get(key);
    if (text === undefined) throw new Error(`Missing language key: ${key}`);
    return text;
  }
}
