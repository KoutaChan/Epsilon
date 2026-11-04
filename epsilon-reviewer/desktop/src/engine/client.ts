import type { Readable, Writable } from "node:stream";
import type { InvokeReply } from "../../../protocol/bridge.ts";
import type { StdioCommandName } from "../../../protocol/commands.ts";
import { isErrorCode, type ErrorBody } from "../../../protocol/errors.ts";
import { errorMessage, isFields, type Fields } from "../values.ts";
import { encodeFrame, FrameDecoder, type Frame } from "./frames.ts";

/** 子プロセスが止まった・起こせない時に待ちの要求へ返す失敗。 */
export const ENGINE_STOPPED: ErrorBody = {
  code: "engine_failed",
  message: "The analysis engine stopped.",
};

/**
 * Java の子プロセスの stdin・stdout の上の要求と答え。id は 1 から増やし、答えは id で対応させる（順は問わない）。
 * failed の知らせの失敗を覚え、子プロセスが終われば待ちの要求をその失敗か engine_failed で断る。
 */
export class EngineClient {
  readonly #input: Writable;
  readonly #onBroken: () => void;
  readonly #decoder = new FrameDecoder();
  readonly #pending = new Map<number, (reply: InvokeReply) => void>();
  #nextId = 1;
  #failure: ErrorBody | null = null;
  #closed: ErrorBody | null = null;

  /** @param onBroken 壊れた枠を受けた時に呼ぶ（子プロセスを止める） */
  constructor(input: Writable, output: Readable, onBroken: () => void) {
    this.#input = input;
    this.#onBroken = onBroken;
    output.on("data", (chunk: Buffer) => this.#receive(chunk));
  }

  /** failed の知らせで受けた起動の失敗。 */
  get failure(): ErrorBody | null {
    return this.#failure;
  }

  /** 要求を送る。答えは失敗も含めて解決で返す（拒否しない）。 */
  request(
    command: StdioCommandName,
    request: object,
    body?: ArrayBuffer,
  ): Promise<InvokeReply> {
    if (this.#closed !== null)
      return Promise.resolve({ ok: false, error: this.#closed });
    const id = this.#nextId++;
    let chunks: Uint8Array[];
    try {
      chunks = encodeFrame({ id, command, request }, body);
    } catch (error) {
      return Promise.resolve({
        ok: false,
        error: { code: "invalid_request", message: errorMessage(error) },
      });
    }
    return new Promise((resolve) => {
      this.#pending.set(id, resolve);
      for (const chunk of chunks) this.#input.write(chunk);
    });
  }

  /** 子プロセスが終わった。待ちの要求と以後の要求を failed の失敗か engine_failed で断る。 */
  close(): void {
    if (this.#closed !== null) return;
    const error = this.#failure ?? ENGINE_STOPPED;
    this.#closed = error;
    for (const resolve of this.#pending.values()) resolve({ ok: false, error });
    this.#pending.clear();
  }

  #receive(chunk: Buffer): void {
    if (this.#closed !== null) return;
    try {
      for (const frame of this.#decoder.push(chunk)) this.#accept(frame);
    } catch (error) {
      console.error("The analysis engine sent an invalid frame:", error);
      this.close();
      this.#onBroken();
    }
  }

  #accept({ header, body }: Frame): void {
    if (!isFields(header)) throw new Error("The frame header is not an object.");
    if (header.event === "ready") return;
    if (header.event === "failed") {
      if (!isErrorBody(header.error)) throw new Error("The failed event has no error.");
      this.#failure = header.error;
      return;
    }
    const id = header.id;
    const resolve = typeof id === "number" ? this.#pending.get(id) : undefined;
    if (typeof id !== "number" || resolve === undefined)
      throw new Error(`No request waits for the reply ${String(id)}.`);
    const reply = replyOf(header, body);
    this.#pending.delete(id);
    resolve(reply);
  }
}

/** @throws Error 答えの形が違う */
function replyOf(header: Fields, body: ArrayBuffer | null): InvokeReply {
  if (header.ok === false) {
    if (!isErrorBody(header.error)) throw new Error("The failed reply has no error.");
    return { ok: false, error: header.error };
  }
  if (header.ok !== true) throw new Error("The reply has no ok flag.");
  const reply: { ok: true; response?: unknown; body?: ArrayBuffer } = { ok: true };
  if (header.response !== undefined) reply.response = header.response;
  if (body !== null) reply.body = body;
  return reply;
}

function isErrorBody(value: unknown): value is ErrorBody {
  return isFields(value) && isErrorCode(value.code) && typeof value.message === "string";
}
