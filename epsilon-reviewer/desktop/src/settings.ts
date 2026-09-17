import { readFile } from "node:fs/promises";
import path from "node:path";
import { writeJsonAtomically } from "./result-store";
import { DesktopError } from "./errors";
import type { Languages } from "./language";

export interface Settings {
  serverUrl: string;
  resultsDirectory: string;
  language: string;
}

export function parseServerOrigin(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new DesktopError("invalid_server_url", "The server URL is invalid.");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.pathname !== "/"
  ) {
    throw new DesktopError(
      "invalid_server_url",
      "The server URL must be an HTTP(S) origin without credentials.",
    );
  }
  return url.origin;
}

/** 読み込み時に検証し、変更時は保存を確定してから参照を置き換える。 */
export class SettingsStore {
  private tail: Promise<unknown> = Promise.resolve();

  private constructor(
    private readonly file: string,
    private readonly languages: Languages,
    private current: Settings,
  ) {}

  static async open(
    file: string,
    defaults: Settings,
    languages: Languages,
  ): Promise<SettingsStore> {
    let current = defaults;
    try {
      const saved = JSON.parse(await readFile(file, "utf8")) as Settings;
      current = {
        serverUrl: parseServerOrigin(saved.serverUrl),
        resultsDirectory: saved.resultsDirectory,
        language: languages.requireSupportedLanguage(
          saved.language ?? defaults.language,
        ),
      };
      if (!path.isAbsolute(current.resultsDirectory))
        throw new DesktopError(
          "invalid_settings",
          "The results directory must be an absolute path.",
        );
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    return new SettingsStore(file, languages, current);
  }

  get value(): Readonly<Settings> {
    return this.current;
  }

  setServerUrl(value: string): Promise<Settings> {
    return this.update("serverUrl", parseServerOrigin(value));
  }

  setLanguage(value: string): Promise<Settings> {
    return this.update(
      "language",
      this.languages.requireSupportedLanguage(value),
    );
  }

  setResultsDirectory(value: string): Promise<Settings> {
    // OS のディレクトリ選択ダイアログから渡された絶対パス。
    return this.update("resultsDirectory", value);
  }

  private update<K extends keyof Settings>(
    key: K,
    value: Settings[K],
  ): Promise<Settings> {
    const operation = this.tail.then(async () => {
      const next = { ...this.current, [key]: value };
      await writeJsonAtomically(this.file, next);
      this.current = next;
      return next;
    });
    this.tail = operation.catch(() => undefined);
    return operation;
  }
}
