import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { DesktopError } from "./errors";

/** Web と同じカタログを使う。native dialog の表示直前に現在の言語を参照する。 */
export class Languages {
  private readonly catalogs = new Map<string, Record<string, string>>();

  constructor(directory: string) {
    for (const file of readdirSync(directory)) {
      if (file.endsWith(".json")) {
        this.catalogs.set(
          path.basename(file, ".json"),
          JSON.parse(readFileSync(path.join(directory, file), "utf8")),
        );
      }
    }
    if (!this.catalogs.has("ja"))
      throw new Error("The Japanese language catalog is missing.");
  }

  selectLanguageForLocale(locale: string): string {
    if (this.catalogs.has(locale)) return locale;
    const base = locale.split("-")[0]!;
    return this.catalogs.has(base) ? base : "ja";
  }

  requireSupportedLanguage(language: string): string {
    if (!this.catalogs.has(language))
      throw new DesktopError(
        "invalid_language",
        "Unknown language: " + language,
      );
    return language;
  }

  getMessage(language: string, key: string): string {
    const text =
      this.catalogs.get(language)![key] ?? this.catalogs.get("ja")![key];
    if (text === undefined) throw new Error("Missing language key: " + key);
    return text;
  }
}
