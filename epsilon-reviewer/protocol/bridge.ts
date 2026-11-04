import type { CommandRequest, RendererCommandName } from "./commands.ts";
import type { ErrorBody } from "./errors.ts";

// デスクトップの画面とメインプロセスの橋渡し（preload が window.epsilonReviewer に出す）。
// どの呼び出しも失敗を ok: false で返し、IPC の向こうへ例外を投げない（例外は符号を失うため）。

export interface DesktopSettings {
  /** 解析結果の保存先のフォルダー（絶対パス）。 */
  resultsDirectory: string;
  /** 言語の id（language/<id>/）。 */
  language: string;
}

export type BridgeReply<T> =
  | { ok: true; value: T }
  | { ok: false; error: ErrorBody };

/** Java の操作の答え。body は document の操作（result）の gzip の文書。 */
export type InvokeReply =
  | { ok: true; response?: unknown; body?: ArrayBuffer }
  | { ok: false; error: ErrorBody };

export interface ReviewerBridge {
  /** Java の操作を呼ぶ。body は importBytes の牌譜。 */
  invoke<N extends RendererCommandName>(
    command: N,
    request: CommandRequest<N>,
    body?: ArrayBuffer,
  ): Promise<InvokeReply>;
  /** 選んだ・落としたファイルの絶対パス。パスの無いファイルは空の文字列。 */
  pathForFile(file: File): string;
  settings(): Promise<BridgeReply<DesktopSettings>>;
  /** 言語を保存し、窓の題も変える。 */
  setLanguage(language: string): Promise<BridgeReply<DesktopSettings>>;
  /** フォルダーを選ぶ画面を出し、選べば保存先を変えて Java を開き直す。解析の途中は analysis_active。 */
  chooseResultsDirectory(): Promise<BridgeReply<DesktopSettings>>;
  /** 保存の画面を出して解析結果のファイルを書く。やめたら saved: false。 */
  exportResult(resultId: string): Promise<BridgeReply<{ saved: boolean }>>;
}

/** IPC の道の名前。 */
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
