export interface ApiResponse<T = unknown> {
  status: number;
  body: T;
}

export class DesktopError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status = 400,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "DesktopError";
  }
}

/** 診断文は英語、利用者向けの翻訳は renderer の言語カタログが所有する。 */
export function errorResponse(error: unknown): ApiResponse {
  const known = error instanceof DesktopError;
  return {
    status: known ? error.status : 500,
    body: {
      error: {
        code: known ? error.code : "storage_failed",
        message:
          error instanceof Error ? error.message : "Desktop operation failed.",
      },
    },
  };
}
