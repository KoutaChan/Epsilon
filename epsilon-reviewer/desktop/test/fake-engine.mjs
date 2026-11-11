// テストで Java のエンジンの代わりに動く偽物。protocol/stdio.ts のフレームで応答する。FAKE_ENGINE=locked なら
// 起動に失敗し、FAKE_IGNORE_EOF を付けると stdin が閉じても終わらない（kill のテストに使う）。
import { encodeFrame, FrameDecoder } from "../src/engine/frames.ts";

const DOCUMENT = new Uint8Array([0x1f, 0x8b, 8, 0, 1, 2, 3]);

function send(header, body, done) {
  const chunks = encodeFrame(header, body);
  chunks.forEach((chunk, index) =>
    process.stdout.write(chunk, index === chunks.length - 1 ? done : undefined),
  );
}

function answer({ header, body }) {
  const { id, command, request } = header;
  switch (command) {
    case "models":
      return send({
        id,
        ok: true,
        response: {
          pid: process.pid,
          cwd: process.cwd(),
          data: process.env.EPSILON_REVIEWER_DATA,
          models: process.env.EPSILON_REVIEWER_MODELS,
          threads: process.env.OMP_NUM_THREADS,
        },
      });
    case "importBytes":
      return send({ id, ok: true, response: { bytes: body?.byteLength ?? 0 } });
    case "result":
      return send({ id, ok: true }, DOCUMENT.slice().buffer);
    case "job":
      if (request.jobId === "crash") process.exit(3);
      if (request.jobId === "slow")
        return setTimeout(
          () =>
            send({
              id,
              ok: true,
              response: { jobId: "slow", status: "running", progress: 1 },
            }),
          100,
        );
      return send({
        id,
        ok: false,
        error: {
          code: "job_not_found",
          message: "The analysis job was not found.",
        },
      });
    default:
      return send({
        id,
        ok: false,
        error: { code: "invalid_request", message: "The request is invalid." },
      });
  }
}

if (process.env.FAKE_ENGINE === "locked") {
  send(
    {
      event: "failed",
      error: {
        code: "data_locked",
        message: "The data directory is used by another reviewer.",
      },
    },
    undefined,
    () => process.exit(1),
  );
} else {
  process.stderr.write(`started ${process.pid}\n`);
  send({ event: "ready" });
  const decoder = new FrameDecoder();
  process.stdin.on("data", (chunk) => decoder.push(chunk).forEach(answer));
  if (process.env.FAKE_IGNORE_EOF)
    process.stdin.on("end", () => setInterval(() => {}, 1000));
}
