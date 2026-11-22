package com.epsilon.prism.policy;

import com.epsilon.prism.core.AgariOrder;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * 方策のバージョンごとの係数（policy.json）。ネットワークの出力 s(o, a) から σ を作る方法は、貪欲の 1 つだけ（s の argmax。同点は
 * p0 で決める。{@link GreedyAnswer}）。推論で証明できる和了の順序クランプを掛けるかもここで記録する（欠けていれば {@link
 * AgariOrder.Mode#LEGACY}、未知のモードは例外）。
 *
 * @param version 方策のバージョン。反復番号と同じ
 * @param agariOrder 推論時の和了順序クランプのモード
 */
public record PolicyParams(int version, AgariOrder.Mode agariOrder) {
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  public PolicyParams {
    Objects.requireNonNull(agariOrder, "agariOrder");
  }

  public PolicyParams(int version) {
    this(version, AgariOrder.Mode.LEGACY);
  }

  public static PolicyParams of(int version) {
    return new PolicyParams(version, AgariOrder.Mode.LEGACY);
  }

  public static PolicyParams of(int version, AgariOrder.Mode agariOrder) {
    return new PolicyParams(version, agariOrder);
  }

  /** policy.json を読む。{@code agariOrder} が欠けていれば {@link AgariOrder.Mode#LEGACY}、未知の値は例外。 */
  public static PolicyParams read(Path file) throws IOException {
    JsonElement root;
    try {
      root = JsonParser.parseString(Files.readString(file));
    } catch (JsonParseException invalid) {
      throw new IOException("Invalid policy parameters " + file, invalid);
    }
    if (root == null || !root.isJsonObject()) {
      throw new IOException("Invalid policy parameters " + file);
    }
    JsonObject object = root.getAsJsonObject();
    JsonElement versionElement = object.get("version");
    if (versionElement == null
        || !versionElement.isJsonPrimitive()
        || !versionElement.getAsJsonPrimitive().isNumber()) {
      throw new IOException("Invalid policy parameters " + file);
    }
    int version;
    try {
      double numeric = versionElement.getAsDouble();
      version = versionElement.getAsInt();
      if (numeric != version || version < 1) {
        throw new IOException("Invalid policy parameters " + file);
      }
    } catch (NumberFormatException invalid) {
      throw new IOException("Invalid policy parameters " + file, invalid);
    }
    AgariOrder.Mode mode = AgariOrder.Mode.LEGACY;
    if (object.has("agariOrder")) {
      JsonElement modeElement = object.get("agariOrder");
      if (!modeElement.isJsonPrimitive() || !modeElement.getAsJsonPrimitive().isString()) {
        throw new IOException("Invalid agariOrder in " + file);
      }
      String name = modeElement.getAsString();
      try {
        mode = AgariOrder.Mode.valueOf(name);
      } catch (IllegalArgumentException unknown) {
        throw new IOException("Unknown agariOrder mode " + name + " in " + file, unknown);
      }
    }
    return new PolicyParams(version, mode);
  }

  /** policy.json を書く。{@link AgariOrder.Mode#LEGACY} では従来と同じ形式（version のみ）で書く。 */
  public void write(Path file) throws IOException {
    JsonObject object = new JsonObject();
    object.add("version", new JsonPrimitive(version));
    if (agariOrder != AgariOrder.Mode.LEGACY) {
      object.add("agariOrder", new JsonPrimitive(agariOrder.name()));
    }
    Files.writeString(file, GSON.toJson(object));
  }
}
