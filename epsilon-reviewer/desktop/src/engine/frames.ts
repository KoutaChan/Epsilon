import {
  FRAME_BODY_LIMIT,
  FRAME_HEADER_LIMIT,
} from "../../../protocol/stdio.ts";

// フレームの形は [ヘッダーの長さ u32 BE][ヘッダー: UTF-8 の JSON][本文の長さ u32 BE][本文]（protocol/stdio.ts）。

/** 長さのフィールドのバイト数。 */
const LENGTH_BYTES = 4;

const UTF8 = new TextDecoder();

/** 読んだ 1 つのフレーム。body は本文と同じサイズの ArrayBuffer で、本文が無ければ null。 */
export interface Frame {
  header: unknown;
  body: ArrayBuffer | null;
}

/**
 * フレームを、書き出す順に並べたバイト列の配列にする。本文はコピーせず、配列の最後の要素としてそのまま渡す。
 *
 * @throws RangeError ヘッダーか本文が上限を超えるとき
 * @throws TypeError ヘッダーを JSON にできないとき
 */
export function encodeFrame(header: object, body?: ArrayBuffer): Uint8Array[] {
  const json = JSON.stringify(header);
  const headerLength = Buffer.byteLength(json);
  const bodyLength = body?.byteLength ?? 0;
  if (headerLength > FRAME_HEADER_LIMIT)
    throw new RangeError(
      `The frame header exceeds ${FRAME_HEADER_LIMIT} bytes.`,
    );
  if (bodyLength > FRAME_BODY_LIMIT)
    throw new RangeError(`The frame body exceeds ${FRAME_BODY_LIMIT} bytes.`);
  const prefix = Buffer.allocUnsafe(headerLength + 2 * LENGTH_BYTES);
  prefix.writeUInt32BE(headerLength, 0);
  prefix.write(json, LENGTH_BYTES, "utf8");
  prefix.writeUInt32BE(bodyLength, LENGTH_BYTES + headerLength);
  return body === undefined || bodyLength === 0
    ? [prefix]
    : [prefix, new Uint8Array(body)];
}

type Part = "headerLength" | "header" | "bodyLength" | "body";

/**
 * 子プロセスの出力のチャンクからフレームを組み立てる。チャンク同士は連結しない。1 つのチャンクに収まる長さと
 * ヘッダーはその場で読み、チャンクをまたぐ部分だけを、その部分のサイズのバッファへコピーする。本文は必ず、
 * 本文と同じサイズの ArrayBuffer へ 1 回だけコピーする（そのまま IPC へ渡せる）。
 * 壊れたフレームを受けると例外を投げ、その後は使えない。
 */
export class FrameDecoder {
  #part: Part = "headerLength";
  #size = LENGTH_BYTES;
  #filled = 0;
  /** チャンクをまたぐ部分を貯めるバッファ。 */
  #pending: Uint8Array | null = null;
  readonly #length = new Uint8Array(LENGTH_BYTES);
  #header: unknown = null;
  #body: ArrayBuffer | null = null;

  /**
   * チャンクを読み、そろったフレームを順に返す。
   *
   * @throws Error 長さが上限を超えるとき、またはヘッダーが JSON でないとき
   */
  push(chunk: Uint8Array): Frame[] {
    const frames: Frame[] = [];
    let offset = 0;
    while (offset < chunk.length) {
      const take = Math.min(this.#size - this.#filled, chunk.length - offset);
      let bytes: Uint8Array;
      if (this.#filled === 0 && take === this.#size && this.#part !== "body") {
        bytes = chunk.subarray(offset, offset + take);
      } else {
        this.#pending ??=
          this.#part === "header" ? new Uint8Array(this.#size) : this.#length;
        this.#pending.set(chunk.subarray(offset, offset + take), this.#filled);
        this.#filled += take;
        if (this.#filled < this.#size) break;
        bytes = this.#pending;
      }
      offset += take;
      const frame = this.#complete(bytes);
      if (frame !== null) frames.push(frame);
    }
    return frames;
  }

  /** そろった部分を読み、次の部分へ進む。フレームがそろえばそれを返し、まだなら null を返す。 */
  #complete(bytes: Uint8Array): Frame | null {
    this.#filled = 0;
    this.#pending = null;
    switch (this.#part) {
      case "headerLength": {
        const length = readLength(bytes);
        if (length > FRAME_HEADER_LIMIT)
          throw new Error(
            `The frame header of ${length} bytes exceeds the limit.`,
          );
        this.#expect("header", length);
        return null;
      }
      case "header":
        this.#header = JSON.parse(UTF8.decode(bytes));
        this.#expect("bodyLength", LENGTH_BYTES);
        return null;
      case "bodyLength": {
        const length = readLength(bytes);
        if (length > FRAME_BODY_LIMIT)
          throw new Error(
            `The frame body of ${length} bytes exceeds the limit.`,
          );
        if (length === 0) return this.#emit(null);
        this.#body = new ArrayBuffer(length);
        this.#pending = new Uint8Array(this.#body);
        this.#expect("body", length);
        return null;
      }
      case "body":
        return this.#emit(this.#body);
    }
  }

  #expect(part: Part, size: number): void {
    this.#part = part;
    this.#size = size;
  }

  #emit(body: ArrayBuffer | null): Frame {
    const frame = { header: this.#header, body };
    this.#header = null;
    this.#body = null;
    this.#expect("headerLength", LENGTH_BYTES);
    return frame;
  }
}

function readLength(bytes: Uint8Array): number {
  return new DataView(bytes.buffer, bytes.byteOffset, LENGTH_BYTES).getUint32(
    0,
  );
}
