import type { DesktopSettings } from "../../../protocol/bridge.ts";
import type {
  CommandName,
  CommandRequest,
  CommandResponse,
} from "../../../protocol/commands.ts";

export interface CallOptions {
  /** importBytes で送る牌譜。 */
  body?: Blob | ArrayBuffer;
  signal?: AbortSignal;
}

/** 操作の表（COMMANDS）にある操作を 1 つ送る。失敗したら ApiError を投げ、中止されたら signal の理由をそのまま投げる。 */
export interface CommandCaller {
  call<N extends CommandName>(
    command: N,
    request: CommandRequest<N>,
    options?: CallOptions,
  ): Promise<CommandResponse<N>>;
}

/** ブラウザーから HTTP のパスへ送る。 */
export interface WebTransport extends CommandCaller {
  readonly platform: "web";
}

/** デスクトップの画面からブリッジ（window.epsilonReviewer）へ送る。ブリッジでしかできない操作もここに置く。 */
export interface DesktopTransport extends CommandCaller {
  readonly platform: "desktop";
  /** 選んだファイルやドロップしたファイルの絶対パス。パスの無いファイルは空文字列。 */
  pathForFile(file: File): string;
  settings(): Promise<DesktopSettings>;
  setLanguage(language: string): Promise<DesktopSettings>;
  chooseResultsDirectory(): Promise<DesktopSettings>;
  /** 保存ダイアログを出して書き出す。保存したら true、取り消したら false。 */
  exportResult(resultId: string): Promise<boolean>;
}

export type Transport = WebTransport | DesktopTransport;
