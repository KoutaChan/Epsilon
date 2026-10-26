package com.epsilon.prism.policy;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 版の係数（policy.json）。網の出力 s(o, a) から σ への写像は貪欲（s の argmax、同点は p0。{@link NetPolicyService}）だけなので、版が持つのは番号だけ。
 *
 * @param version 方策の版。反復番号と同じ
 */
public record PolicyParams(int version) {
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  public static PolicyParams of(int version) {
    return new PolicyParams(version);
  }

  /** policy.json を読む。 */
  public static PolicyParams read(Path file) throws IOException {
    PolicyParams params;
    try {
      params = GSON.fromJson(Files.readString(file), PolicyParams.class);
    } catch (JsonParseException invalid) {
      throw new IOException("Invalid policy parameters " + file, invalid);
    }
    if (params == null || params.version() < 1)
      throw new IOException("Invalid policy parameters " + file);
    return params;
  }

  /** policy.json を書く。 */
  public void write(Path file) throws IOException {
    Files.writeString(file, GSON.toJson(this));
  }
}
