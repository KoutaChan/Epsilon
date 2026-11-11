package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.Map;
import java.util.Set;

/**
 * GRP のチェックポイント。1 つのディレクトリに grp.json と weights.f32（float32、リトルエンディアン）を置く。grp.json には
 * モデルの種類とルールを書く。ほかに、GrpNetwork ならフィールドの名前と幅、隠れ層の幅、段階ごとの出力ヘッドを持つかを書く。
 * GruSinkhorn なら隠れ層の幅と層の数を、OutcomeNetwork ならフィールドの名前と幅、隠れ層の幅を書く。weights.f32 には、重みを
 * モデルに応じて GrpTensor・GruTensor・OutcomeTensor の宣言の順に並べる。チェックポイントは、grp.json と weights.f32 をこの順に
 * つないだバイト列の SHA-256（小文字の 16 進）で見分ける。形式のバージョン、マジックナンバー、最適化器の状態は持たない。読むときは
 * フィールドの幅を今のコードの幅と比べ、違えば読まない。
 *
 * <p>OutcomeNetwork は Φ を出すモデル（GrpModel）ではないので、{@code readOutcomes} で読む。FinalRound は、葉のネットワークと
 * OutcomeNetwork のチェックポイント 2 つと、打ち切りの値から作り直せる。
 */
public final class GrpCheckpoint {

  public static final String JSON_FILE = "grp.json";
  public static final String WEIGHTS_FILE = "weights.f32";

  /** grp.json に書くモデルの種類の名前。 */
  private static final String NETWORK = "GrpNetwork";
  private static final String GRU_SINKHORN = "GruSinkhorn";
  private static final String OUTCOME_NETWORK = "OutcomeNetwork";

  private static final String MODEL = "model";
  private static final String VARIANT = "variant";
  private static final String FIELDS = "fields";
  private static final String HIDDEN = "hidden";
  private static final String PHASE_HEADS = "phaseHeads";
  private static final String LAYERS = "layers";

  private static final Set<String> NETWORK_KEYS = Set.of(MODEL, VARIANT, FIELDS, HIDDEN, PHASE_HEADS);
  private static final Set<String> GRU_KEYS = Set.of(MODEL, VARIANT, HIDDEN, LAYERS);
  private static final Set<String> OUTCOME_KEYS = Set.of(MODEL, VARIANT, FIELDS, HIDDEN);

  private GrpCheckpoint() {}

  /**
   * grp.json と weights.f32 を書く。2 つのファイルは、書き先と同じ親にある一時ディレクトリに書いてから、ディレクトリごとリネーム
   * する。途中で落ちても、半端なチェックポイントは残らない。チェックポイントは上書きしないので、書き先が既にあれば IOException を
   * 投げる。
   *
   * @param directory 書き先のディレクトリ。まだ無いこと
   * @return SHA-256（小文字の 16 進）
   */
  public static String write(GrpModel model, Path directory) throws IOException {
    return switch (model) {
      case GrpNetwork network -> store(describe(network.shape()), floats(network.weights()), directory);
      case GruSinkhorn gru -> store(describe(gru.shape()), floats(gru.weights()), directory);
    };
  }

  /**
   * OutcomeNetwork を {@link #write(GrpModel, Path)} と同じやり方で書く。
   *
   * @return SHA-256（小文字の 16 進）
   */
  public static String write(OutcomeNetwork network, Path directory) throws IOException {
    return store(describe(network.shape()), floats(network.weights()), directory);
  }

  /**
   * Φ を出すモデルを読む。読めないとき、grp.json が今のコードの形に合わないとき、weights.f32 の大きさが形に合わないときは
   * IOException を投げる。OutcomeNetwork のチェックポイントでも IOException を投げるので、そちらは
   * {@link #readOutcomes} で読む。
   */
  public static GrpModel read(Path directory) throws IOException {
    JsonObject root = parse(directory);
    String model = string(root, MODEL, directory);
    return switch (model) {
      case NETWORK -> network(root, directory);
      case GRU_SINKHORN -> gru(root, directory);
      case OUTCOME_NETWORK -> throw new IOException(directory + " holds an " + OUTCOME_NETWORK + ", not a Φ model");
      default -> throw new IOException("Unknown GRP model " + model + " in " + directory);
    };
  }

  /**
   * OutcomeNetwork を読む。読めないとき、ほかの種類のモデルのとき、grp.json か weights.f32 が今のコードの形に合わないときは
   * IOException を投げる。
   */
  public static OutcomeNetwork readOutcomes(Path directory) throws IOException {
    JsonObject root = parse(directory);
    String model = string(root, MODEL, directory);
    if (!model.equals(OUTCOME_NETWORK)) {
      throw new IOException(directory + " holds a " + model + ", not an " + OUTCOME_NETWORK);
    }
    requireKeys(root, OUTCOME_KEYS, directory);
    OutcomeShape shape;
    try {
      Variant variant = Variant.valueOf(string(root, VARIANT, directory));
      shape = new OutcomeShape(variant, fields(root, variant, directory), root.get(HIDDEN).getAsInt());
    } catch (RuntimeException malformed) {
      throw new IOException("Malformed " + JSON_FILE + " in " + directory + ": " + malformed.getMessage(), malformed);
    }
    float[] weights = weights(directory, OutcomeTensor.starts(shape)[OutcomeTensor.values().length]);
    try {
      return OutcomeNetwork.of(shape, weights);
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid weights in " + directory + ": " + invalid.getMessage(), invalid);
    }
  }

  /** ディレクトリにあるチェックポイントの SHA-256 を求める。 */
  public static String sha256(Path directory) throws IOException {
    byte[] description = Files.readAllBytes(directory.resolve(JSON_FILE));
    return sha256(description, Files.readAllBytes(directory.resolve(WEIGHTS_FILE)));
  }

  private static GrpNetwork network(JsonObject root, Path directory) throws IOException {
    requireKeys(root, NETWORK_KEYS, directory);
    GrpShape shape = shape(root, directory);
    float[] weights = weights(directory, GrpTensor.starts(shape)[GrpTensor.values().length]);
    try {
      return GrpNetwork.of(shape, weights);
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid weights in " + directory + ": " + invalid.getMessage(), invalid);
    }
  }

  private static GruSinkhorn gru(JsonObject root, Path directory) throws IOException {
    requireKeys(root, GRU_KEYS, directory);
    GruShape shape;
    try {
      Variant variant = Variant.valueOf(string(root, VARIANT, directory));
      shape = new GruShape(variant, root.get(HIDDEN).getAsInt(), root.get(LAYERS).getAsInt());
    } catch (RuntimeException malformed) {
      throw new IOException("Malformed " + JSON_FILE + " in " + directory + ": " + malformed.getMessage(), malformed);
    }
    float[] weights = weights(directory, GruTensor.count(shape));
    try {
      return GruSinkhorn.of(shape, weights);
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid weights in " + directory + ": " + invalid.getMessage(), invalid);
    }
  }

  /** weights.f32 から、形で決まる数の値を読む。ファイルの大きさが合わなければ IOException を投げる。 */
  private static float[] weights(Path directory, int count) throws IOException {
    byte[] bytes = Files.readAllBytes(directory.resolve(WEIGHTS_FILE));
    if (bytes.length != Float.BYTES * count) {
      throw new IOException(
          WEIGHTS_FILE + " in " + directory + " has " + bytes.length + " bytes, the shape needs " + Float.BYTES * count);
    }
    float[] weights = new float[count];
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(weights);
    return weights;
  }

  private static String store(byte[] description, byte[] weights, Path directory) throws IOException {
    Path target = directory.toAbsolutePath();
    if (Files.exists(target)) {
      throw new FileAlreadyExistsException(target + " already holds a checkpoint");
    }
    Path staging = target.resolveSibling("." + target.getFileName() + ".writing");
    deleteStaging(staging);
    Files.createDirectories(staging);
    try {
      DurableFiles.write(staging.resolve(JSON_FILE), description);
      DurableFiles.write(staging.resolve(WEIGHTS_FILE), weights);
      DurableFiles.syncDirectory(staging);
      Files.move(staging, target, StandardCopyOption.ATOMIC_MOVE);
      DurableFiles.syncDirectory(target.getParent());
    } finally {
      deleteStaging(staging);
    }
    return sha256(description, weights);
  }

  /** 一時ディレクトリを消す。前に落ちた書き込みの残りも、リネームの前に失敗した今回の書き込みの残りも消す。 */
  private static void deleteStaging(Path staging) throws IOException {
    Files.deleteIfExists(staging.resolve(JSON_FILE));
    Files.deleteIfExists(staging.resolve(WEIGHTS_FILE));
    Files.deleteIfExists(staging);
  }

  private static byte[] describe(GrpShape shape) {
    JsonObject root = new JsonObject();
    root.addProperty(MODEL, NETWORK);
    root.addProperty(VARIANT, shape.variant().name());
    root.add(FIELDS, widths(shape.variant(), shape.fields()));
    root.addProperty(HIDDEN, shape.hidden());
    root.addProperty(PHASE_HEADS, shape.phaseHeads());
    return text(root);
  }

  private static byte[] describe(OutcomeShape shape) {
    JsonObject root = new JsonObject();
    root.addProperty(MODEL, OUTCOME_NETWORK);
    root.addProperty(VARIANT, shape.variant().name());
    root.add(FIELDS, widths(shape.variant(), shape.fields()));
    root.addProperty(HIDDEN, shape.hidden());
    return text(root);
  }

  /** フィールドの名前から幅への対応を作る。並びは宣言の順。 */
  private static JsonObject widths(Variant variant, Set<GrpField> fields) {
    JsonObject widths = new JsonObject();
    for (GrpField field : fields) {
      widths.addProperty(field.name(), field.width(variant));
    }
    return widths;
  }

  private static byte[] describe(GruShape shape) {
    JsonObject root = new JsonObject();
    root.addProperty(MODEL, GRU_SINKHORN);
    root.addProperty(VARIANT, shape.variant().name());
    root.addProperty(HIDDEN, shape.hidden());
    root.addProperty(LAYERS, shape.layers());
    return text(root);
  }

  private static byte[] text(JsonObject root) {
    String text = new GsonBuilder().setPrettyPrinting().create().toJson(root) + "\n";
    return text.getBytes(StandardCharsets.UTF_8);
  }

  private static byte[] floats(float[] values) {
    ByteBuffer buffer = ByteBuffer.allocate(Float.BYTES * values.length).order(ByteOrder.LITTLE_ENDIAN);
    buffer.asFloatBuffer().put(values);
    return buffer.array();
  }

  private static JsonObject parse(Path directory) throws IOException {
    String text = Files.readString(directory.resolve(JSON_FILE), StandardCharsets.UTF_8);
    try {
      return JsonParser.parseString(text).getAsJsonObject();
    } catch (RuntimeException malformed) {
      throw new IOException("Malformed " + JSON_FILE + " in " + directory, malformed);
    }
  }

  private static void requireKeys(JsonObject root, Set<String> keys, Path directory) throws IOException {
    if (!root.keySet().equals(keys)) {
      throw new IOException(JSON_FILE + " in " + directory + " has keys " + root.keySet() + ", expected " + keys);
    }
  }

  private static GrpShape shape(JsonObject root, Path directory) throws IOException {
    try {
      Variant variant = Variant.valueOf(string(root, VARIANT, directory));
      return new GrpShape(
          variant, fields(root, variant, directory), root.get(HIDDEN).getAsInt(), root.get(PHASE_HEADS).getAsBoolean());
    } catch (RuntimeException malformed) {
      throw new IOException("Malformed " + JSON_FILE + " in " + directory + ": " + malformed.getMessage(), malformed);
    }
  }

  /** grp.json に書かれたフィールドを読む。幅が今のコードの幅と違えば IOException を投げる。 */
  private static Set<GrpField> fields(JsonObject root, Variant variant, Path directory) throws IOException {
    Set<GrpField> fields = EnumSet.noneOf(GrpField.class);
    for (Map.Entry<String, JsonElement> entry : root.getAsJsonObject(FIELDS).entrySet()) {
      GrpField field = GrpField.valueOf(entry.getKey());
      int width = entry.getValue().getAsInt();
      if (width != field.width(variant)) {
        throw new IOException(
            field + " is " + width + " wide in " + directory + " but " + field.width(variant) + " in this code");
      }
      fields.add(field);
    }
    return fields;
  }

  private static String string(JsonObject root, String key, Path directory) throws IOException {
    JsonElement value = root.get(key);
    if (value == null || !value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString()) {
      throw new IOException(JSON_FILE + " in " + directory + " lacks the string " + key);
    }
    return value.getAsString();
  }

  private static String sha256(byte[] description, byte[] weights) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      digest.update(description);
      digest.update(weights);
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException("SHA-256 is not available", missing);
    }
  }
}
