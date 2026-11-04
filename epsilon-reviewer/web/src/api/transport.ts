import type { DesktopSettings } from "../../../protocol/bridge.ts";
import type {
  CommandName,
  CommandRequest,
  CommandResponse,
} from "../../../protocol/commands.ts";

export interface CallOptions {
  /** importBytes の牌譜。 */
  body?: Blob | ArrayBuffer;
  signal?: AbortSignal;
}

/** 操作の表（COMMANDS）の 1 操作を送る。失敗は ApiError、中止は signal の理由のまま投げる。 */
export interface CommandCaller {
  call<N extends CommandName>(
    command: N,
    request: CommandRequest<N>,
    options?: CallOptions,
  ): Promise<CommandResponse<N>>;
}

/** ブラウザーから HTTP の道へ送る。 */
export interface WebTransport extends CommandCaller {
  readonly platform: "web";
}

/** デスクトップの画面から橋渡し（window.epsilonReviewer）へ送る。橋渡しだけの手もここに持つ。 */
export interface DesktopTransport extends CommandCaller {
  readonly platform: "desktop";
  /** 選んだ・落としたファイルの絶対パス。パスの無いファイルは空の文字列。 */
  pathForFile(file: File): string;
  settings(): Promise<DesktopSettings>;
  setLanguage(language: string): Promise<DesktopSettings>;
  chooseResultsDirectory(): Promise<DesktopSettings>;
  /** 保存の画面を出して書き出す。やめたら false。 */
  exportResult(resultId: string): Promise<boolean>;
}

export type Transport = WebTransport | DesktopTransport;
