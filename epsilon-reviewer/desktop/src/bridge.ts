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

/** ブリッジから Java のエンジンを呼ぶためのインターフェース。実装は EngineProcess。 */
export interface Engine {
  request(
    command: StdioCommandName,
    request: object,
    body?: ArrayBuffer,
  ): Promise<InvokeReply>;
  /** 起動に失敗したままか（data_locked など）。true の間は、restart するまでどの要求も断る。 */
  readonly failed: boolean;
  /** 保存先を変えた後に呼び、新しい保存先でエンジンを起動し直す。 */
  restart(): Promise<void>;
}

/** ブリッジが使うウィンドウとダイアログ。main.ts が Electron で実装する。 */
export interface BridgeHost {
  /** IPC の送信元を確かめる。信用できれば null を、信用できなければ返すエラーの中身を返す。 */
  checkSender(event: IpcMainInvokeEvent): ErrorBody | null;
  /** 保存先のフォルダーを選ばせる。キャンセルされたら null。 */
  chooseDirectory(current: string): Promise<string | null>;
  /** 書き出し先のファイルを選ばせる。キャンセルされたら null。 */
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

/** ipcMain のうち、ブリッジが使う handle だけを抜き出したもの。テストでは偽物を渡す。 */
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
 * BRIDGE_CHANNELS の全てのチャンネルにハンドラーを登録する。どのハンドラーも送信元を確かめ、BridgeReply か
 * InvokeReply を返す。例外を IPC の向こうへ投げるとエラーコードが消えるので、予期しない失敗も
 * operation_failed の応答にして返す。
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
      const current = settings.value.resultsDirectory;
      const directory = await host.chooseDirectory(current);
      // 今と同じフォルダーを選び直したときは、起動に失敗したままのときだけ起動し直す。起動し直すと、取り込んだ牌譜が消える。
      if (directory === null || (directory === current && !engine.failed))
        return accept(settings.value);
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
 * 画面が始めた解析のジョブのうち、終わった状態（completed・failed・cancelled）をまだ受け取っていないものを覚える。
 * 保存先を変えると Java を起動し直すので、動いている解析がある間は保存先の変更を断る。
 */
class ActiveAnalyses {
  readonly #jobs = new Set<string>();

  /** 画面からの呼び出しとその応答を見て、ジョブを覚えるか忘れる。 */
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

  /**
   * 覚えているジョブの状態を Java に聞き直し、終わっていないジョブがあれば true を返す。
   * 画面が状態を聞かなくなったジョブも、ここで確かめる。
   */
  async running(engine: Engine): Promise<boolean> {
    for (const jobId of [...this.#jobs])
      this.#settle(jobId, await engine.request("job", { jobId }));
    return this.#jobs.size > 0;
  }

  /**
   * 応答が終わった状態か、状態の無い応答（失敗の応答など）なら、そのジョブを忘れる。失敗の応答は、子プロセスが
   * 起動し直してジョブが無くなったことを表す。
   */
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
