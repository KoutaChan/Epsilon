import type { CommandRequest, RendererCommandName } from "./commands.ts";
import type { ErrorBody } from "./errors.ts";

// デスクトップ版の画面とメインプロセスをつなぐブリッジ（preload が window.epsilonReviewer に置く）。
// どの呼び出しも失敗を ok: false で返し、IPC の向こうへ例外を投げない（例外にするとエラーコードが消えるため）。

export interface DesktopSettings {
  /** 解析結果の保存先のフォルダー（絶対パス）。 */
  resultsDirectory: string;
  /** 言語の id（language/<id>/ のフォルダー名）。 */
  language: string;
}

export type BridgeReply<T> =
  { ok: true; value: T } | { ok: false; error: ErrorBody };

/** Java の操作の応答。body は、本文を返す操作（result）が返す gzip の文書。 */
export type InvokeReply =
  | { ok: true; response?: unknown; body?: ArrayBuffer }
  | { ok: false; error: ErrorBody };

export interface ReviewerBridge {
  /** Java の操作を呼ぶ。body は importBytes で渡す牌譜のバイト列。 */
  invoke<N extends RendererCommandName>(
    command: N,
    request: CommandRequest<N>,
    body?: ArrayBuffer,
  ): Promise<InvokeReply>;
  /** 選んだファイルやドロップしたファイルの絶対パスを返す。パスの無いファイルなら空の文字列。 */
  pathForFile(file: File): string;
  settings(): Promise<BridgeReply<DesktopSettings>>;
  /** 言語を保存し、ウィンドウのタイトルもその言語に変える。 */
  setLanguage(language: string): Promise<BridgeReply<DesktopSettings>>;
  /**
   * フォルダーを選ぶダイアログを出す。選ばれたら保存先を変えて Java を起動し直す。解析が待ちか実行中なら
   * analysis_active で断る。
   */
  chooseResultsDirectory(): Promise<BridgeReply<DesktopSettings>>;
  /** 保存のダイアログを出し、解析結果をファイルに書き出す。キャンセルされたら saved: false。 */
  exportResult(resultId: string): Promise<BridgeReply<{ saved: boolean }>>;
}

/** IPC のチャンネルの名前。 */
export const BRIDGE_CHANNELS = {
  invoke: "reviewer:invoke",
  settings: "reviewer:settings",
  language: "reviewer:language",
  directory: "reviewer:directory",
  export: "reviewer:export",
} as const;

declare global {
  interface Window {
    epsilonReviewer?: ReviewerBridge;
  }
}
