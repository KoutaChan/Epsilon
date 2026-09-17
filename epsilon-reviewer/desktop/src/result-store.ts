import { DesktopError } from "./errors";
import { randomUUID } from "node:crypto";
import {
  access,
  mkdir,
  open,
  readFile,
  rename,
  unlink,
} from "node:fs/promises";
import path from "node:path";
import { gunzip } from "node:zlib";
import { promisify } from "node:util";

export const CURRENT_RESULT_FORMAT_VERSION = 4;

export interface ModelSnapshot {
  modelId: string;
  revision: string;
  displayName: string;
  version: string;
  series: string;
}

export interface ResultMetadata {
  source: string;
  fileName?: string;
  names: string[];
  finalScores?: number[] | null;
  finalScoresSource?: string | null;
  roundCount: number;
}

export interface StoredResult {
  formatVersion: typeof CURRENT_RESULT_FORMAT_VERSION;
  resultId: string;
  createdAt: string;
  metadata: ResultMetadata;
  model: ModelSnapshot;
  rounds: { states: unknown[]; steps: unknown[] }[];
}

export interface SourceInput {
  recordSource?: string;
  fileName: string;
  base64: string;
}

export interface ResultSummary {
  resultId: string;
  createdAt: string;
  metadata: ResultMetadata;
  model: ModelSnapshot;
}

interface IndexEntry {
  summary: ResultSummary;
  file: string;
  sourceFile?: string;
}

interface HistoryIndex {
  version: 1;
  entries: IndexEntry[];
}

interface PendingChange {
  version: 1;
  transactionId: string;
  operation: "save" | "remove";
  entry: IndexEntry;
}

export function requireFilenameSafeResultId(id: string): string {
  if (!/^[a-zA-Z0-9_-]{1,128}$/.test(id))
    throw new DesktopError("invalid_result_id", "Invalid result ID.");
  return id;
}

/** 書き込みを flush してから同じディレクトリ内で置き換える。 */
export async function writeBytesAtomically(
  file: string,
  bytes: string | Uint8Array,
  transactionId = randomUUID(),
): Promise<void> {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.${transactionId}.tmp`;
  try {
    const handle = await open(temporary, "wx", 0o600);
    try {
      await handle.writeFile(bytes);
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

export function writeJsonAtomically(
  file: string,
  value: unknown,
  transactionId = randomUUID(),
): Promise<void> {
  return writeBytesAtomically(
    file,
    JSON.stringify(value) + "\n",
    transactionId,
  );
}

const decompressArchive = promisify(gunzip);

/** 圧縮ファイルを受け取る境界で、形式と要求された結果 ID を確認する。 */
async function readResultDocument(
  bytes: Uint8Array,
  id: string,
): Promise<StoredResult> {
  let result: StoredResult;
  try {
    result = JSON.parse((await decompressArchive(bytes)).toString("utf8"));
  } catch (cause) {
    throw new DesktopError(
      "invalid_result",
      "The result archive is not valid gzip JSON.",
      422,
      { cause },
    );
  }
  if (
    !result ||
    result.formatVersion !== CURRENT_RESULT_FORMAT_VERSION ||
    !result.metadata ||
    !result.model ||
    !Array.isArray(result.rounds) ||
    !result.rounds.every(
      (round) =>
        round && Array.isArray(round.states) && Array.isArray(round.steps),
    )
  ) {
    throw new DesktopError(
      "invalid_result",
      "Unsupported or invalid result format.",
      422,
    );
  }
  if (result.resultId !== id)
    throw new DesktopError(
      "result_mismatch",
      "The result ID does not match the requested ID.",
      422,
    );
  return result;
}

function requireMatchingHistoryPaths(entry: IndexEntry): void {
  requireFilenameSafeResultId(entry.summary.resultId);
  if (
    !path.isAbsolute(entry.file) ||
    path.basename(entry.file) !==
      `${entry.summary.resultId}.epsilon-reviewer.json.gz`
  ) {
    throw new DesktopError(
      "invalid_history",
      "Invalid result path in the history index.",
    );
  }
  if (entry.sourceFile && entry.sourceFile !== `${entry.file}.source.json`) {
    throw new DesktopError(
      "invalid_history",
      "Invalid original-record path in the history index.",
    );
  }
}

async function removeFile(file: string): Promise<void> {
  try {
    await unlink(file);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}

/** サーバーの結果だけを登録する。索引の置換を確定点にして未完了操作を回収する。 */
export class ResultStore {
  private tail: Promise<unknown> = Promise.resolve();

  constructor(private readonly indexDirectory: string) {}

  private get indexFile(): string {
    return path.join(this.indexDirectory, "history-v4.json");
  }
  private get pendingFile(): string {
    return path.join(this.indexDirectory, "pending-v4.json");
  }

  private async index(): Promise<HistoryIndex> {
    let json: string;
    try {
      json = await readFile(this.indexFile, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT")
        return { version: 1, entries: [] };
      throw error;
    }
    const data = JSON.parse(json) as HistoryIndex;
    if (data.version !== 1 || !Array.isArray(data.entries)) {
      throw new DesktopError(
        "invalid_history",
        "Unsupported history index format.",
      );
    }
    for (const entry of data.entries) requireMatchingHistoryPaths(entry);
    return data;
  }

  private exclusive<T>(
    operation: (index: HistoryIndex) => Promise<T>,
  ): Promise<T> {
    const result = this.tail.then(async () => {
      const index = await this.index();
      await this.recover(index);
      return operation(index);
    });
    this.tail = result.catch(() => undefined);
    return result;
  }

  /** 索引に登録済みの save は保持し、未登録の save と確定済み remove は回収する。 */
  private async recover(index: HistoryIndex): Promise<void> {
    let text: string;
    try {
      text = await readFile(this.pendingFile, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
    const pending = JSON.parse(text) as PendingChange;
    if (
      pending.version !== 1 ||
      !["save", "remove"].includes(pending.operation) ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
        pending.transactionId,
      )
    ) {
      throw new DesktopError(
        "invalid_history",
        "Invalid pending storage transaction.",
      );
    }
    requireMatchingHistoryPaths(pending.entry);
    const committed = index.entries.find(
      (entry) => entry.summary.resultId === pending.entry.summary.resultId,
    );
    if (
      committed &&
      (committed.file !== pending.entry.file ||
        committed.sourceFile !== pending.entry.sourceFile)
    ) {
      throw new DesktopError(
        "invalid_history",
        "Pending transaction paths do not match the history index.",
      );
    }
    const files = [pending.entry.file, pending.entry.sourceFile].filter(
      (file): file is string => Boolean(file),
    );
    if (!committed) {
      for (const file of files) await removeFile(file);
    }
    for (const file of files)
      await removeFile(`${file}.${pending.transactionId}.tmp`);
    await removeFile(`${this.indexFile}.${pending.transactionId}.tmp`);
    await removeFile(this.pendingFile);
  }

  async list(): Promise<ResultSummary[]> {
    return this.exclusive(async (index) => {
      return index.entries
        .map((entry) => entry.summary)
        .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
    });
  }

  async has(id: string): Promise<boolean> {
    requireFilenameSafeResultId(id);
    return this.exclusive(async (index) =>
      index.entries.some((entry) => entry.summary.resultId === id),
    );
  }

  async get(id: string): Promise<StoredResult> {
    return readResultDocument(await this.readArchive(id), id);
  }

  async readArchive(id: string): Promise<Buffer> {
    requireFilenameSafeResultId(id);
    return this.exclusive(async (index) => {
      const entry = index.entries.find(
        (entry) => entry.summary.resultId === id,
      );
      if (!entry)
        throw new DesktopError(
          "result_not_found",
          "The result is not in local history.",
        );
      return readFile(entry.file);
    });
  }

  async getSource(id: string): Promise<SourceInput> {
    requireFilenameSafeResultId(id);
    return this.exclusive(async (index) => {
      const entry = index.entries.find(
        (entry) => entry.summary.resultId === id,
      );
      if (!entry?.sourceFile)
        throw new DesktopError(
          "source_missing",
          "The original record is not stored locally.",
        );
      const source = JSON.parse(
        await readFile(entry.sourceFile, "utf8"),
      ) as SourceInput;
      if (
        typeof source.fileName !== "string" ||
        typeof source.base64 !== "string"
      )
        throw new DesktopError(
          "source_missing",
          "The stored original-record file is invalid.",
        );
      return source;
    });
  }

  async saveArchive(
    id: string,
    bytes: Uint8Array,
    directory: string,
    source?: SourceInput,
  ): Promise<void> {
    requireFilenameSafeResultId(id);
    const result = await readResultDocument(bytes, id);
    await this.exclusive(async (index) => {
      const existing = index.entries.find(
        (entry) => entry.summary.resultId === result.resultId,
      );
      if (existing) {
        const stored = await readFile(existing.file);
        if (!stored.equals(bytes))
          throw new DesktopError(
            "storage_conflict",
            "Cannot overwrite a different result with the same ID.",
          );
        if (
          source &&
          (!existing.sourceFile ||
            JSON.stringify(
              JSON.parse(await readFile(existing.sourceFile, "utf8")),
            ) !== JSON.stringify(source))
        ) {
          throw new DesktopError(
            "storage_conflict",
            "The original record does not match the stored record.",
          );
        }
        return;
      }
      const file = path.resolve(
        directory,
        `${result.resultId}.epsilon-reviewer.json.gz`,
      );
      const sourceFile = source ? `${file}.source.json` : undefined;
      for (const candidate of [file, sourceFile].filter(
        (value): value is string => Boolean(value),
      )) {
        try {
          await access(candidate);
        } catch (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") continue;
          throw error;
        }
        throw new DesktopError(
          "storage_conflict",
          "Cannot overwrite an existing file outside the history index.",
        );
      }
      const { resultId, createdAt, metadata, model } = result;
      const entry: IndexEntry = {
        summary: { resultId, createdAt, metadata, model },
        file,
        sourceFile,
      };
      const transactionId = randomUUID();
      const pending: PendingChange = {
        version: 1,
        transactionId,
        operation: "save",
        entry,
      };
      await writeJsonAtomically(this.pendingFile, pending);
      if (source) await writeJsonAtomically(sourceFile!, source, transactionId);
      await writeBytesAtomically(file, bytes, transactionId);
      index.entries.unshift(entry);
      await writeJsonAtomically(this.indexFile, index, transactionId);
      await removeFile(this.pendingFile);
    });
  }

  async remove(id: string): Promise<void> {
    requireFilenameSafeResultId(id);
    await this.exclusive(async (index) => {
      const entry = index.entries.find(
        (entry) => entry.summary.resultId === id,
      );
      if (!entry) return;
      const transactionId = randomUUID();
      const pending: PendingChange = {
        version: 1,
        transactionId,
        operation: "remove",
        entry,
      };
      await writeJsonAtomically(this.pendingFile, pending);
      index.entries = index.entries.filter(
        (item) => item.summary.resultId !== id,
      );
      await writeJsonAtomically(this.indexFile, index, transactionId);
      for (const file of [entry.file, entry.sourceFile].filter(
        (file): file is string => Boolean(file),
      ))
        await removeFile(file);
      await removeFile(this.pendingFile);
    });
  }
}
