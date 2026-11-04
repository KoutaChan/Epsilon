import type {
  BridgeReply,
  InvokeReply,
  ReviewerBridge,
} from "../../../protocol/bridge.ts";
import { COMMANDS } from "../../../protocol/commands.ts";
import type {
  CommandName,
  CommandRequest,
  CommandResponse,
  RendererCommandName,
} from "../../../protocol/commands.ts";
import { ApiError } from "./errors.ts";
import type { CallOptions, DesktopTransport } from "./transport.ts";

/** 橋渡しの invoke を操作の名前ごとの型でなく 1 つの形で呼ぶ（呼ぶ前に renderer の操作か確かめる）。 */
interface Invoker {
  invoke(
    command: RendererCommandName,
    request: object,
    body?: ArrayBuffer,
  ): Promise<InvokeReply>;
}

/** デスクトップの橋渡しで送る。IPC は ArrayBuffer だけを運ぶので Blob の本文はここで読む。 */
export function bridgeTransport(bridge: ReviewerBridge): DesktopTransport {
  const invoker: Invoker = bridge;
  return {
    platform: "desktop",
    async call<N extends CommandName>(
      command: N,
      request: CommandRequest<N>,
      options: CallOptions = {},
    ): Promise<CommandResponse<N>> {
      if (!isRendererCommand(command))
        throw new TypeError(`${command} is not available on the desktop.`);
      const { body, signal } = options;
      signal?.throwIfAborted();
      const reply = await invoker.invoke(
        command,
        request,
        body instanceof Blob ? await body.arrayBuffer() : body,
      );
      signal?.throwIfAborted();
      if (!reply.ok) throw new ApiError(reply.error.code, reply.error.message);
      const value: unknown =
        COMMANDS[command].response === "document"
          ? await readDocument(reply.body)
          : reply.response;
      return value as CommandResponse<N>;
    },
    pathForFile: (file) => bridge.pathForFile(file),
    settings: async () => unwrap(await bridge.settings()),
    setLanguage: async (language) => unwrap(await bridge.setLanguage(language)),
    chooseResultsDirectory: async () =>
      unwrap(await bridge.chooseResultsDirectory()),
    exportResult: async (resultId) =>
      unwrap(await bridge.exportResult(resultId)).saved,
  };
}

function isRendererCommand(
  command: CommandName,
): command is RendererCommandName {
  return COMMANDS[command].renderer;
}

/** 保存の gzip の文書を画面の中で 1 回だけ解いて読む。 */
async function readDocument(body: ArrayBuffer | undefined): Promise<unknown> {
  if (body === undefined)
    throw new ApiError("invalid_response", "The document reply has no body.");
  try {
    return await new Response(
      new Blob([body]).stream().pipeThrough(new DecompressionStream("gzip")),
    ).json();
  } catch {
    throw new ApiError("invalid_response", "The document could not be read.");
  }
}

function unwrap<T>(reply: BridgeReply<T>): T {
  if (!reply.ok) throw new ApiError(reply.error.code, reply.error.message);
  return reply.value;
}
