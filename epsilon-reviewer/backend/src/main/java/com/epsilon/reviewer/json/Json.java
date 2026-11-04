package com.epsilon.reviewer.json;

import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/** Gson の設定（record・null の欄を書かない・HTML を逃がさない）と、厳密な読み。 */
public final class Json {
  public static final Gson GSON =
      new GsonBuilder()
          .disableHtmlEscaping()
          .registerTypeAdapter(ScenarioAnswer.class, new ScenarioAnswerJson())
          .create();

  private Json() {}

  /** UTF-8 の JSON。JsonElement の木は null の欄もそのまま書く（局面の draw）。 */
  public static byte[] write(Object value) {
    var bytes = new ByteArrayOutputStream();
    try (JsonWriter writer =
        GSON.newJsonWriter(new OutputStreamWriter(bytes, StandardCharsets.UTF_8))) {
      if (value instanceof JsonElement tree) {
        writer.setSerializeNulls(true);
        GSON.getAdapter(JsonElement.class).write(writer, tree);
      } else GSON.toJson(value, value.getClass(), writer);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return bytes.toByteArray();
  }

  /**
   * 1 つの JSON の物を厳密に読む（RFC 8259 のとおり。後ろに続く値も断る）。
   *
   * @param limit 受け付けるバイト数
   * @throws ReviewerException invalid_request
   */
  public static JsonObject readObject(byte[] utf8, int limit) {
    if (utf8.length > limit)
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST, "The JSON body exceeds " + limit + " bytes.");
    try (var reader =
        new JsonReader(
            new InputStreamReader(new ByteArrayInputStream(utf8), StandardCharsets.UTF_8))) {
      reader.setStrictness(Strictness.STRICT);
      JsonElement element = JsonParser.parseReader(reader);
      if (reader.peek() != JsonToken.END_DOCUMENT || !element.isJsonObject())
        throw new ReviewerException(ErrorCode.INVALID_REQUEST, "The body must be one JSON object.");
      return element.getAsJsonObject();
    } catch (JsonParseException | IOException failure) {
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, "The body is not valid JSON.");
    }
  }
}
