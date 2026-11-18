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
