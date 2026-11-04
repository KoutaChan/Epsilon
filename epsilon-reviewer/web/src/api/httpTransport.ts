import { COMMANDS, httpTarget } from "../../../protocol/commands.ts";
import type {
  CommandName,
  CommandRequest,
  CommandResponse,
  CommandSpec,
  ResponseBody,
} from "../../../protocol/commands.ts";
import { isErrorCode } from "../../../protocol/errors.ts";
import { ApiError } from "./errors.ts";
import type { CallOptions, WebTransport } from "./transport.ts";

/** COMMANDS の HTTP の道へ fetch で送る。共有リンクの操作だけ Cookie を送らない。 */
export function httpTransport(): WebTransport {
  return {
    platform: "web",
    async call<N extends CommandName>(
      command: N,
      request: CommandRequest<N>,
      options: CallOptions = {},
    ): Promise<CommandResponse<N>> {
      const spec: CommandSpec = COMMANDS[command];
      if (spec.http === null)
        throw new TypeError(`${command} has no HTTP route.`);
      const target = httpTarget(spec.http, request);
      const init: RequestInit = {
        method: spec.http.method,
        credentials: spec.access.includes("share") ? "omit" : "same-origin",
        signal: options.signal,
      };
      if (spec.request === "json") {
        init.headers = { "Content-Type": "application/json" };
        init.body = JSON.stringify(target.body ?? {});
      } else if (spec.request === "binary") {
        init.headers = { "Content-Type": "application/octet-stream" };
        init.body = options.body;
      }
      let response: Response;
      try {
        response = await fetch(target.path, init);
      } catch (error) {
        if (options.signal?.aborted) throw error;
        throw new ApiError(
          "connection_failed",
          "Could not connect to the reviewer.",
        );
      }
      if (!response.ok) throw await failure(response, options.signal);
      const value = await decode(spec.response, response, options.signal);
      return value as CommandResponse<N>;
    },
  };
}

/** 2xx の本文。document はブラウザーが Content-Encoding の gzip を解き、ETag の 304 も扱う。 */
async function decode(
  kind: ResponseBody,
  response: Response,
  signal: AbortSignal | undefined,
): Promise<unknown> {
  if (kind === "empty") return undefined;
  try {
    return kind === "attachment"
      ? await response.blob()
      : await response.json();
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(
      "invalid_response",
      "The response body could not be read.",
      response.status,
    );
  }
}

/** 2xx 以外の本文 {error:{code,message}}。形が違えば invalid_response。 */
async function failure(
  response: Response,
  signal: AbortSignal | undefined,
): Promise<ApiError> {
  let body: unknown;
  try {
    body = await response.json();
  } catch (error) {
    if (signal?.aborted) throw error;
  }
  const error = isObject(body) ? body.error : undefined;
  if (isObject(error) && isErrorCode(error.code))
    return new ApiError(
      error.code,
      typeof error.message === "string" ? error.message : "",
      response.status,
    );
  return new ApiError(
    "invalid_response",
    `Unexpected HTTP status ${response.status}.`,
    response.status,
  );
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
