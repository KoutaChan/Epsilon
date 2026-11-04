import type { StdioCommandName } from "./commands.ts";
import type { ErrorBody } from "./errors.ts";

// Electron のメインプロセスと Java の子プロセス（1 つ）の間の枠。HTTP もポートも使わない。
// 1 枠 = [頭の長さ u32 BE][頭: UTF-8 の JSON][本文の長さ u32 BE][本文のバイト列]。本文が無い枠は本文の長さ 0。
// stdin は Electron → Java、stdout は Java → Electron で、stdout には枠だけを流す（Java の記録はファイルへ）。
// Java は stdin の終わりで解析を止め、モデルを閉じて終わる（終了の状態 0）。

/** 頭の JSON の上限（バイト）。history の答えは結果の数だけ伸びる。 */
export const FRAME_HEADER_LIMIT = 16 * 1024 * 1024;

/** 本文の上限（バイト）。牌譜は 16 MiB まで。 */
export const FRAME_BODY_LIMIT = 32 * 1024 * 1024;

/**
 * Electron → Java の要求。id は Electron が付ける正の整数で、子プロセスの間は重ならない。本文は importBytes の
 * 牌譜のバイト列だけ。
 */
export interface StdioRequest {
  id: number;
  command: StdioCommandName;
  request: object;
}

/**
 * Java → Electron の答え。答えの順は要求の順と限らない。document の操作（result）は response を持たず、
 * 本文が保存した gzip の文書。empty の操作は response を持たない。
 */
export type StdioReply =
  | { id: number; ok: true; response?: unknown }
  | { id: number; ok: false; error: ErrorBody };

/**
 * Java → Electron の知らせ。ready は保存先とカタログを開いた後に 1 回（モデルはまだ開かない）。failed は起動に
 * 失敗した時に 1 回だけ送り、状態 1 で終わる。
 */
export type StdioEvent =
  { event: "ready" } | { event: "failed"; error: ErrorBody };

export type StdioMessage = StdioReply | StdioEvent;

/**
 * 子プロセスへ渡す環境変数の名前。パスは日本語を含み得るので引数でなく環境変数で渡し、引数は ASCII に保つ。
 */
export const ENGINE_ENV = {
  /** 解析結果の保存先のフォルダー（デスクトップの「保存先」）。 */
  data: "EPSILON_REVIEWER_DATA",
  /** 同梱の models.json。 */
  models: "EPSILON_REVIEWER_MODELS",
  /** Java の記録のフォルダー。 */
  logs: "EPSILON_REVIEWER_LOGS",
} as const;
