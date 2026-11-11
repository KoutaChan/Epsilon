import type { ErrorBody } from "./errors.ts";

// Electron のメインプロセスと Java の子プロセス（1 つ）がやり取りするフレームの形。HTTP もポートも使わない。
// 1 フレーム = [ヘッダーの長さ u32 BE][ヘッダー: UTF-8 の JSON][本文の長さ u32 BE][本文のバイト列]。
// 本文が無いフレームは、本文の長さを 0 にする。
// Electron から Java へは stdin、Java から Electron へは stdout で送る。stdout にはフレームしか書かない（Java のログはファイルへ書く）。
// Java は stdin が閉じたら解析を止め、モデルを閉じて終了する（終了コード 0）。

/** ヘッダーの JSON の上限（バイト）。履歴の応答は、解析結果が多いほど長くなる。 */
export const FRAME_HEADER_LIMIT = 16 * 1024 * 1024;

/** 本文の上限（バイト）。牌譜は 16 MiB まで。 */
export const FRAME_BODY_LIMIT = 32 * 1024 * 1024;

/**
 * Java から Electron への応答。応答は要求の順に返るとは限らない。本文の種類が document の操作（result）は、
 * 応答に response が無く、保存した gzip の文書が本文になる。本文の種類が empty の操作も、応答に response が無い。
 */
export type StdioReply =
  | { id: number; ok: true; response?: unknown }
  | { id: number; ok: false; error: ErrorBody };

/**
 * Java から Electron へのイベント。ready は、保存先とカタログを開いた後に 1 回だけ送る（モデルはまだ開いていない）。
 * failed は、起動に失敗したときに 1 回だけ送り、その後は終了コード 1 で終了する。
 */
export type StdioEvent =
  { event: "ready" } | { event: "failed"; error: ErrorBody };

/**
 * 子プロセスへ渡す環境変数の名前。パスには日本語が入りうるので、パスはコマンドラインの引数ではなく環境変数で渡し、
 * 引数は ASCII だけにする。
 */
export const ENGINE_ENV = {
  /** 解析結果を保存するフォルダー（デスクトップ版の「保存先」）。 */
  data: "EPSILON_REVIEWER_DATA",
  /** 同梱の models.json のパス。 */
  models: "EPSILON_REVIEWER_MODELS",
  /** Java がログを書くフォルダー。 */
  logs: "EPSILON_REVIEWER_LOGS",
} as const;
