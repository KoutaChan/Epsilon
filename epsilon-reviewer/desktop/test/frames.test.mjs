import assert from "node:assert/strict";
import { test } from "node:test";
import { FRAME_BODY_LIMIT, FRAME_HEADER_LIMIT } from "../../protocol/stdio.ts";
import { encodeFrame, FrameDecoder } from "../src/engine/frames.ts";

const wire = (...frames) =>
  Buffer.concat(frames.flatMap(([header, body]) => encodeFrame(header, body)));

const FRAMES = [
  [{ id: 1, ok: true, response: { names: ["東家", "南家"] } }, new Uint8Array([1, 2, 250]).buffer],
  [{ event: "ready" }],
  [{ id: 2, ok: false, error: { code: "busy", message: "What-if analysis is busy." } }],
];

function assertFrames(frames) {
  assert.equal(frames.length, FRAMES.length);
  frames.forEach((frame, index) => {
    const [header, body] = FRAMES[index];
    assert.deepEqual(frame.header, header);
    if (body === undefined) assert.equal(frame.body, null);
    else {
      assert.ok(frame.body instanceof ArrayBuffer);
      assert.deepEqual(new Uint8Array(frame.body), new Uint8Array(body));
    }
  });
}

test("frames split at every byte boundary decode to the same headers and bodies", () => {
  const bytes = wire(...FRAMES);
  for (let cut = 0; cut <= bytes.length; cut++) {
    const decoder = new FrameDecoder();
    assertFrames([...decoder.push(bytes.subarray(0, cut)), ...decoder.push(bytes.subarray(cut))]);
  }
  const decoder = new FrameDecoder();
  const frames = [];
  for (let index = 0; index < bytes.length; index++)
    frames.push(...decoder.push(bytes.subarray(index, index + 1)));
  assertFrames(frames);
});

test("several frames in one chunk come out in order", () => {
  assertFrames(new FrameDecoder().push(wire(...FRAMES)));
});

test("the encoder keeps the body without copying and the decoder gives it its own buffer", () => {
  const body = new Uint8Array([9, 8, 7]).buffer;
  const chunks = encodeFrame({ id: 3 }, body);
  assert.equal(chunks.length, 2);
  assert.equal(chunks[1].buffer, body);
  const [frame] = new FrameDecoder().push(Buffer.concat(chunks));
  assert.equal(frame.body.byteLength, 3);
  assert.notEqual(frame.body, body);
});

test("headers and bodies up to the limits pass, larger ones are refused", () => {
  const header = { text: "x".repeat(FRAME_HEADER_LIMIT - 11) };
  assert.equal(Buffer.byteLength(JSON.stringify(header)), FRAME_HEADER_LIMIT);
  const [frame] = new FrameDecoder().push(wire([header]));
  assert.equal(frame.header.text.length, FRAME_HEADER_LIMIT - 11);
  assert.throws(() => encodeFrame({ text: "x".repeat(FRAME_HEADER_LIMIT - 10) }), RangeError);
  assert.equal(encodeFrame({}, new ArrayBuffer(FRAME_BODY_LIMIT)).length, 2);
  assert.throws(() => encodeFrame({}, new ArrayBuffer(FRAME_BODY_LIMIT + 1)), RangeError);
});

test("the decoder refuses an oversized length before the bytes arrive", () => {
  const headerLength = Buffer.alloc(4);
  headerLength.writeUInt32BE(FRAME_HEADER_LIMIT + 1);
  assert.throws(
    () => new FrameDecoder().push(headerLength),
    new RegExp(`header of ${FRAME_HEADER_LIMIT + 1} bytes`),
  );
  const bodyLength = wire([{ id: 4 }]);
  bodyLength.writeUInt32BE(FRAME_BODY_LIMIT + 1, bodyLength.length - 4);
  assert.throws(() => new FrameDecoder().push(bodyLength), /body of 33554433 bytes/);
});

test("a header that is not JSON is refused", () => {
  const bytes = Buffer.from([0, 0, 0, 3, 0x7b, 0x7b, 0x7b, 0, 0, 0, 0]);
  assert.throws(() => new FrameDecoder().push(bytes), SyntaxError);
});
