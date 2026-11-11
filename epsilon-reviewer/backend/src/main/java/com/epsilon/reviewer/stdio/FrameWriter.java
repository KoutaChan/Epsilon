package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.json.Json;
import com.google.gson.JsonObject;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/** フレームを書く（形は FrameReader と同じ）。応答は要求を処理するスレッドごとに届くので、書き込みは synchronized で 1 つずつ行う。 */
public final class FrameWriter {
  private static final byte[] NO_BODY = new byte[0];
  private final DataOutputStream out;

  public FrameWriter(OutputStream out) {
    this.out = new DataOutputStream(new BufferedOutputStream(out, 64 * 1024));
  }

  /**
   * @param body 本文。無ければ null
   * @throws IllegalArgumentException ヘッダーか本文が上限を超える。このときは何も書かない
   */
  public synchronized void write(JsonObject header, byte[] body) throws IOException {
    byte[] json = Json.write(header);
    byte[] payload = body == null ? NO_BODY : body;
    if (json.length > FrameReader.HEADER_LIMIT || payload.length > FrameReader.BODY_LIMIT)
      throw new IllegalArgumentException(
          "A frame of " + json.length + " + " + payload.length + " bytes exceeds the limits.");
    out.writeInt(json.length);
    out.write(json);
    out.writeInt(payload.length);
    out.write(payload);
    out.flush();
  }
}
