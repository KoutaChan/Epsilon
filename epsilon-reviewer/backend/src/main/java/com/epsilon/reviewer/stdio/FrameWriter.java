package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.json.Json;
import com.google.gson.JsonObject;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;

/** 枠を書く（FrameReader と同じ形）。答えは仮想スレッドごとに来るので 1 つの書き手で直列にする。 */
public final class FrameWriter {
  private static final byte[] NO_BODY = new byte[0];
  private final DataOutputStream out;

  public FrameWriter(OutputStream out) {
    this.out = new DataOutputStream(new BufferedOutputStream(out, 64 * 1024));
  }

  /**
   * @param body 本文。無ければ null
   * @throws IllegalArgumentException 頭か本文が上限を超える（何も書かない）
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
