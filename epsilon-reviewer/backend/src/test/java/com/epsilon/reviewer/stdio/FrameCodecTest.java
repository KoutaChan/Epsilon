package com.epsilon.reviewer.stdio;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.Test;

/** フレームを書いて読み戻せること、壊れたフレームで読むのが止まること、サイズの上限を確かめる。フレームの形は [ヘッダーの長さ u32 BE][ヘッダー][本文の長さ u32 BE][本文]。 */
public class FrameCodecTest {
  @Test
  public void framesRoundTripWithTheirBodies() throws IOException {
    var bytes = new ByteArrayOutputStream();
    var writer = new FrameWriter(bytes);
    JsonObject first =
        JsonParser.parseString("{\"id\":1,\"name\":\"みなと\",\"draw\":null}").getAsJsonObject();
    writer.write(first, new byte[] {1, 2, 3});
    writer.write(JsonParser.parseString("{\"event\":\"ready\"}").getAsJsonObject(), null);
    var reader = new FrameReader(new ByteArrayInputStream(bytes.toByteArray()));
    Frame frame = reader.read().orElseThrow();
    Assert.assertEquals(frame.header(), first);
    Assert.assertTrue(frame.header().get("draw").isJsonNull());
    Assert.assertEquals(frame.body(), new byte[] {1, 2, 3});
    Frame ready = reader.read().orElseThrow();
    Assert.assertEquals(ready.header().get("event").getAsString(), "ready");
    Assert.assertEquals(ready.body().length, 0);
    Assert.assertEquals(reader.read(), Optional.empty());
  }

  @Test
  public void brokenFramesStopTheReader() {
    Assert.expectThrows(
        EOFException.class, () -> new FrameReader(stream(raw("{\"id\":1}", 10, 4))).read());
    Assert.expectThrows(IOException.class, () -> new FrameReader(stream(raw("[1]", 0, 0))).read());
    Assert.expectThrows(
        IOException.class,
        () -> new FrameReader(stream(lengths(FrameReader.HEADER_LIMIT + 1))).read());
    Assert.expectThrows(
        IOException.class,
        () -> new FrameReader(stream(raw("{\"id\":1}", 32 * 1024 * 1024 + 1, 0))).read());
    Assert.expectThrows(
        EOFException.class, () -> new FrameReader(stream(new byte[] {0, 0})).read());
  }

  @Test
  public void oversizedRepliesAreNotWritten() throws IOException {
    var bytes = new ByteArrayOutputStream();
    var writer = new FrameWriter(bytes);
    JsonObject header = new JsonObject();
    header.addProperty("text", "x".repeat(FrameReader.HEADER_LIMIT));
    Assert.expectThrows(IllegalArgumentException.class, () -> writer.write(header, null));
    Assert.assertEquals(bytes.size(), 0);
  }

  private static ByteArrayInputStream stream(byte[] bytes) {
    return new ByteArrayInputStream(bytes);
  }

  /** ヘッダー、宣言する本文の長さ、実際の本文のバイト数を別々に指定してフレームを作る。 */
  private static byte[] raw(String header, int declaredBody, int body) throws IOException {
    var bytes = new ByteArrayOutputStream();
    var out = new DataOutputStream(bytes);
    byte[] json = header.getBytes(StandardCharsets.UTF_8);
    out.writeInt(json.length);
    out.write(json);
    out.writeInt(declaredBody);
    out.write(new byte[body]);
    return bytes.toByteArray();
  }

  private static byte[] lengths(int header) throws IOException {
    var bytes = new ByteArrayOutputStream();
    new DataOutputStream(bytes).writeInt(header);
    return bytes.toByteArray();
  }
}
