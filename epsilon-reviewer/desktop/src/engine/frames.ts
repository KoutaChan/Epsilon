import {
  FRAME_BODY_LIMIT,
  FRAME_HEADER_LIMIT,
} from "../../../protocol/stdio.ts";

// 枠: [頭の長さ u32 BE][頭: UTF-8 の JSON][本文の長さ u32 BE][本文]（protocol/stdio.ts）。

/** 長さの欄のバイト数。 */
const LENGTH_BYTES = 4;

const UTF8 = new TextDecoder();

/** 読んだ 1 枠。body は本文（無ければ null）で、本文ちょうどの大きさの ArrayBuffer。 */
export interface Frame {
  header: unknown;
  body: ArrayBuffer | null;
}

/**
 * 枠を書く順のバイト列にする。本文は写さずに最後の 1 つとしてそのまま渡す。
 *
 * @throws RangeError 頭か本文が上限を超える
 * @throws TypeError 頭を JSON にできない
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
 * 子プロセスの出力の塊から枠を組む。塊はつながない。1 つの塊に収まる長さと頭はその場で読み、塊をまたぐ部分だけを
 * その部分の大きさの置き場へ写す。本文は常に本文ちょうどの ArrayBuffer へ 1 回だけ写す（そのまま IPC へ渡せる）。
 * 壊れた枠で例外を投げ、その後は使えない。
 */
export class FrameDecoder {
  #part: Part = "headerLength";
  #size = LENGTH_BYTES;
  #filled = 0;
  /** 塊をまたぐ部分の置き場。 */
  #pending: Uint8Array | null = null;
  readonly #length = new Uint8Array(LENGTH_BYTES);
  #header: unknown = null;
  #body: ArrayBuffer | null = null;

  /**
   * 塊を読み、揃った枠を順に返す。
   *
   * @throws Error 長さが上限を超える、または頭が JSON でない
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

  /** 揃った部分を読んで次の部分へ進む。枠が揃えば返す。 */
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
