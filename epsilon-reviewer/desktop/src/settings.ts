import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import path from "node:path";
import type { DesktopSettings } from "../../protocol/bridge.ts";
import { isFields } from "./values.ts";

/**
 * <userData>/settings.json。無ければ既定の値を使い（書かない）、読めない・形が違えば既定の値で書き直す。
 * 変更は 1 つずつ行い、ファイルへ確定してから記憶の値を替える。
 */
export class SettingsStore {
  readonly #file: string;
  #value: DesktopSettings;
  #tail: Promise<unknown> = Promise.resolve();

  private constructor(file: string, value: DesktopSettings) {
    this.#file = file;
    this.#value = value;
  }

  /** @param isLanguage 言語の id が language/ のフォルダーにあるか */
  static async open(
    file: string,
    defaults: DesktopSettings,
    isLanguage: (language: string) => boolean,
  ): Promise<SettingsStore> {
    const text = await readFile(file, "utf8").catch((error: unknown) =>
      isMissing(error) ? null : "",
    );
    if (text === null) return new SettingsStore(file, defaults);
    const saved = parseSettings(text, isLanguage);
    if (saved !== null) return new SettingsStore(file, saved);
    await writeJsonAtomically(file, defaults);
    return new SettingsStore(file, defaults);
  }

  get value(): DesktopSettings {
    return this.#value;
  }

  update(change: Partial<DesktopSettings>): Promise<DesktopSettings> {
    const committed = this.#tail.then(async () => {
      const next = { ...this.#value, ...change };
      await writeJsonAtomically(this.#file, next);
      this.#value = next;
      return next;
    });
    this.#tail = committed.catch(() => undefined);
    return committed;
  }
}

/** 形が正しければその値、違えば null。欄は resultsDirectory（絶対パス）と language だけ。 */
function parseSettings(
  text: string,
  isLanguage: (language: string) => boolean,
): DesktopSettings | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isFields(value)) return null;
  const { resultsDirectory, language, ...rest } = value;
  return typeof resultsDirectory === "string" &&
    path.isAbsolute(resultsDirectory) &&
    typeof language === "string" &&
    isLanguage(language) &&
    Object.keys(rest).length === 0
    ? { resultsDirectory, language }
    : null;
}

function isMissing(error: unknown): boolean {
  return (error as NodeJS.ErrnoException).code === "ENOENT";
}

/** 同じフォルダーの一時ファイルへ書いて fsync し、置き換える。 */
async function writeJsonAtomically(
  file: string,
  value: unknown,
): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${randomUUID()}.tmp`;
  try {
    const handle = await open(temporary, "wx", 0o600);
    try {
      await handle.writeFile(JSON.stringify(value) + "\n");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(temporary, file);
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
}
