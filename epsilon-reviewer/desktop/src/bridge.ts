import path from "node:path";
import type { IpcMainInvokeEvent } from "electron";
import {
  BRIDGE_CHANNELS,
  type BridgeReply,
  type DesktopSettings,
  type InvokeReply,
} from "../../protocol/bridge.ts";
import {
  COMMANDS,
  type CommandName,
  type RendererCommandName,
  type StdioCommandName,
} from "../../protocol/commands.ts";
import type { ErrorBody } from "../../protocol/errors.ts";
import type { JobStatus } from "../../protocol/model.ts";
import type { Languages } from "./language.ts";
import type { SettingsStore } from "./settings.ts";
import { errorMessage, isFields, type Fields } from "./values.ts";

/** ブリッジが使う Java の口（EngineProcess）。 */
export interface Engine {
  request(
    command: StdioCommandName,
    request: object,
    body?: ArrayBuffer,
  ): Promise<InvokeReply>;
  /** 保存先を変えた後に、その保存先で開き直す。 */
  restart(): Promise<void>;
}

/** 窓とダイアログ（main.ts が Electron で作る）。 */
export interface BridgeHost {
  /** 信用できない送り手なら失敗の中身、信用できれば null。 */
  checkSender(event: IpcMainInvokeEvent): ErrorBody | null;
  /** 保存先のフォルダーを選ぶ。やめたら null。 */
  chooseDirectory(current: string): Promise<string | null>;
  /** 書き出すファイルを選ぶ。やめたら null。 */
  chooseExportPath(resultId: string): Promise<string | null>;
  setTitle(title: string): void;
}

export interface BridgeServices {
  engine: Engine;
  settings: SettingsStore;
  languages: Languages;
  host: BridgeHost;
}

type Listener = (
  event: IpcMainInvokeEvent,
  ...args: unknown[]
) => Promise<unknown>;

/** ipcMain の handle（試験は偽の表を渡す）。 */
export interface IpcHandlers {
  handle(channel: string, listener: Listener): void;
}

type Refusal = { ok: false; error: ErrorBody };

const FINISHED: ReadonlySet<string> = new Set<JobStatus>([
  "completed",
  "failed",
  "cancelled",
]);

/**
 * BRIDGE_CHANNELS の手を登録する。どの手も送り手を確かめ、BridgeReply か InvokeReply を返す（例外は IPC の
 * 向こうへ投げると符号を失うので、予期しない失敗も operation_failed の答えにする）。
 */
export function registerBridge(
  ipc: IpcHandlers,
  { engine, settings, languages, host }: BridgeServices,
): void {
  const analyses = new ActiveAnalyses();
  const handle = (
    channel: string,
    run: (...args: unknown[]) => Promise<unknown>,
  ): void =>
    ipc.handle(channel, async (event, ...args) => {
      try {
        const untrusted = host.checkSender(event);
        return untrusted === null ? await run(...args) : refuse(untrusted);
      } catch (error) {
        console.error(`The bridge call ${channel} failed:`, error);
        return refuse({
          code: "operation_failed",
          message: errorMessage(error),
        });
      }
    });

  handle(
    BRIDGE_CHANNELS.invoke,
    async (command, request, body): Promise<InvokeReply> => {
      if (!isRendererCommand(command))
        return invalid(`The renderer cannot call ${String(command)}.`);
      if (!isFields(request)) return invalid("The request must be an object.");
      if (command === "importFile" && !isAbsolutePath(request.path))
        return invalid("importFile needs an absolute path.");
      let bytes: ArrayBuffer | undefined;
      if (command === "importBytes") {
        if (!(body instanceof ArrayBuffer))
          return invalid("importBytes needs the record bytes.");
        bytes = body;
      } else if (body !== undefined)
        return invalid(`${command} takes no body.`);
      const reply = await engine.request(command, request, bytes);
      analyses.observe(command, request, reply);
      return reply;
    },
  );

  handle(
    BRIDGE_CHANNELS.settings,
    async (): Promise<BridgeReply<DesktopSettings>> => accept(settings.value),
  );

  handle(
    BRIDGE_CHANNELS.language,
    async (language): Promise<BridgeReply<DesktopSettings>> => {
      if (typeof language !== "string" || !languages.has(language))
        return invalid(`Unknown language: ${String(language)}.`);
      const next = await settings.update({ language });
      host.setTitle(languages.message(language, "app.title"));
      return accept(next);
    },
  );

  handle(
    BRIDGE_CHANNELS.directory,
    async (): Promise<BridgeReply<DesktopSettings>> => {
      if (await analyses.running(engine))
        return refuse({
          code: "analysis_active",
          message: "An analysis is already queued or running.",
        });
      const directory = await host.chooseDirectory(
        settings.value.resultsDirectory,
      );
      if (directory === null) return accept(settings.value);
      const next = await settings.update({ resultsDirectory: directory });
      await engine.restart();
      return accept(next);
    },
  );

  handle(
    BRIDGE_CHANNELS.export,
    async (resultId): Promise<BridgeReply<{ saved: boolean }>> => {
      if (typeof resultId !== "string")
        return invalid("The result ID must be a string.");
      const target = await host.chooseExportPath(resultId);
      if (target === null) return accept({ saved: false });
      const reply = await engine.request("exportResult", {
        resultId,
        path: target,
      });
      return reply.ok ? accept({ saved: true }) : reply;
    },
  );
}

/**
 * 画面が始めた解析のうち、終わりの状態（completed・failed・cancelled）をまだ見ていない仕事。保存先の変更は
 * Java を起こし直すので、走る解析がある間は断る。
 */
class ActiveAnalyses {
  readonly #jobs = new Set<string>();

  /** 画面の呼び出しの答えを見る。 */
  observe(
    command: RendererCommandName,
    request: Fields,
    reply: InvokeReply,
  ): void {
    if (command === "startAnalysis") {
      if (
        reply.ok &&
        isFields(reply.response) &&
        typeof reply.response.jobId === "string"
      )
        this.#jobs.add(reply.response.jobId);
    } else if (
      (command === "job" || command === "cancelJob") &&
      typeof request.jobId === "string"
    ) {
      this.#settle(request.jobId, reply);
    }
  }

  /** 覚えている仕事を Java に聞き直し、終わっていない仕事があるか（画面が聞くのをやめた仕事も確かめる）。 */
  async running(engine: Engine): Promise<boolean> {
    for (const jobId of [...this.#jobs])
      this.#settle(jobId, await engine.request("job", { jobId }));
    return this.#jobs.size > 0;
  }

  /** 走っている仕事の答えでなければ忘れる。失敗の答えは、子プロセスが替わって仕事が無いこと。 */
  #settle(jobId: string, reply: InvokeReply): void {
    const status =
      reply.ok && isFields(reply.response) ? reply.response.status : undefined;
    if (typeof status !== "string" || FINISHED.has(status))
      this.#jobs.delete(jobId);
  }
}

function isRendererCommand(value: unknown): value is RendererCommandName {
  return (
    typeof value === "string" &&
    Object.hasOwn(COMMANDS, value) &&
    COMMANDS[value as CommandName].renderer
  );
}

function isAbsolutePath(value: unknown): boolean {
  return typeof value === "string" && path.isAbsolute(value);
}

function accept<T>(value: T): BridgeReply<T> {
  return { ok: true, value };
}

function refuse(error: ErrorBody): Refusal {
  return { ok: false, error };
}

function invalid(message: string): Refusal {
  return refuse({ code: "invalid_request", message });
}
