package com.epsilon.reviewer.model;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.json.Json;
import com.epsilon.spi.ReviewEvaluation;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 起動時に models.json を厳密に読んだモデルの一覧。使えるかは中身のハッシュを作らずに決め、画面のカタログは作り置きして使えるかが変わった
 * 時だけ作り直す。
 *
 * <p>models.json は {"models":[{modelId, revision, displayName, version, series, checkpoint,
 * options?, sha256?}], "defaultModel"?:{modelId, revision}}。checkpoint は models.json のフォルダーからの
 * 相対か絶対。知らない欄・重複・無い既定のモデルは起動の失敗。
 */
public final class ModelCatalog {
  private static final Logger LOG = LoggerFactory.getLogger(ModelCatalog.class);

  /** models.json の大きさの上限（バイト）。 */
  private static final int FILE_LIMIT = 1024 * 1024;

  private static final Set<String> CATALOG_FIELDS = Set.of("models", "defaultModel");
  private static final Set<String> MODEL_FIELDS =
      Set.of(
          "modelId",
          "revision",
          "displayName",
          "version",
          "series",
          "checkpoint",
          "options",
          "sha256");
  private static final Set<String> KEY_FIELDS = Set.of("modelId", "revision");
  private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");

  /** models.json の順。markChanged が値を替える。 */
  private final Map<ModelKey, ModelEntry> entries;

  private final Map<ModelKey, CheckpointFingerprint> fingerprints;
  private final ModelKey defaultModel;
  private volatile CatalogView view;

  private ModelCatalog(
      Map<ModelKey, ModelEntry> entries,
      Map<ModelKey, CheckpointFingerprint> fingerprints,
      ModelKey defaultModel) {
    this.entries = entries;
    this.fingerprints = fingerprints;
    this.defaultModel = defaultModel;
    view = buildView();
  }

  /** @throws IOException models.json が読めないか形が違う（起動の失敗） */
  public static ModelCatalog load(Path modelsJson, PolicyProviders providers) throws IOException {
    JsonObject document;
    try {
      document = Json.readObject(Files.readAllBytes(modelsJson), FILE_LIMIT);
    } catch (ReviewerException failure) {
      throw new IOException("Invalid model catalog " + modelsJson + ": " + failure.getMessage());
    }
    requireFields(document, CATALOG_FIELDS, "The model catalog");
    if (!(document.get("models") instanceof JsonArray models))
      throw invalid("The model catalog requires a models array.");
    Path base = modelsJson.toAbsolutePath().getParent();
    Map<ModelKey, ModelEntry> entries = new LinkedHashMap<>();
    Map<ModelKey, CheckpointFingerprint> fingerprints = new HashMap<>();
    for (int i = 0; i < models.size(); i++) {
      String name = "models[" + i + "]";
      JsonObject row = object(models.get(i), name);
      requireFields(row, MODEL_FIELDS, name);
      var key = new ModelKey(text(row, "modelId", name), text(row, "revision", name));
      if (entries.containsKey(key)) throw invalid(name + " repeats a model ID and revision.");
      String series = text(row, "series", name);
      Path checkpoint = base.resolve(text(row, "checkpoint", name)).normalize();
      Map<String, String> options = options(row, name);
      String sha256 = row.has("sha256") ? text(row, "sha256", name) : null;
      if (sha256 != null && !SHA256.matcher(sha256).matches())
        throw invalid(name + ".sha256 must be 64 lowercase hexadecimal digits.");
      Availability availability = Availability.AVAILABLE;
      if (providers.find(series).isEmpty()) {
        availability = Availability.UNSUPPORTED_SERIES;
        LOG.warn(
            "Model {} {} has no provider for series {}.", key.modelId(), key.revision(), series);
      } else
        try {
          fingerprints.put(key, CheckpointFingerprint.of(checkpoint));
        } catch (IOException | UncheckedIOException failure) {
          availability = Availability.CHECKPOINT_UNREADABLE;
          LOG.warn(
              "Model {} {} has an unreadable checkpoint: {}",
              key.modelId(),
              key.revision(),
              failure.toString());
        }
      entries.put(
          key,
          new ModelEntry(
              key,
              text(row, "displayName", name),
              text(row, "version", name),
              series,
              checkpoint,
              options,
              sha256,
              temperature(options, name),
              availability));
    }
    ModelKey defaultModel = null;
    if (document.has("defaultModel")) {
      JsonObject selected = object(document.get("defaultModel"), "defaultModel");
      requireFields(selected, KEY_FIELDS, "defaultModel");
      defaultModel =
          new ModelKey(
              text(selected, "modelId", "defaultModel"),
              text(selected, "revision", "defaultModel"));
      if (!entries.containsKey(defaultModel))
        throw invalid("The default model is not in the catalog.");
    }
    return new ModelCatalog(entries, fingerprints, defaultModel);
  }

  /** models.json の無いサーバの空のカタログ。 */
  public static ModelCatalog empty() {
    return new ModelCatalog(new LinkedHashMap<>(), Map.of(), null);
  }

  /** 作り置きのカタログ（ハッシュを作らない）。 */
  public CatalogView view() {
    return view;
  }

  /** @throws ReviewerException model_unavailable（カタログに無いか使えない） */
  public ModelEntry require(ModelKey key) {
    ModelEntry entry;
    synchronized (this) {
      entry = entries.get(key);
    }
    if (entry == null || entry.availability() != Availability.AVAILABLE)
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    return entry;
  }

  /** チェックポイントが登録の時と違うと分かったモデルを、以後使えなくする。 */
  void markChanged(ModelKey key) {
    synchronized (this) {
      entries.computeIfPresent(key, (ignored, entry) -> entry.with(Availability.MODEL_CHANGED));
      view = buildView();
    }
    LOG.warn("Model {} {} changed after it was registered.", key.modelId(), key.revision());
  }

  /** 起動時の印。使えるモデルだけが持つ。 */
  CheckpointFingerprint fingerprint(ModelKey key) {
    return fingerprints.get(key);
  }

  private CatalogView buildView() {
    List<CatalogModel> models = new ArrayList<>(entries.size());
    for (ModelEntry entry : entries.values())
      models.add(
          new CatalogModel(
              entry.key().modelId(),
              entry.key().revision(),
              entry.displayName(),
              entry.version(),
              entry.availability() == Availability.AVAILABLE));
    return new CatalogView(List.copyOf(models), defaultModel);
  }

  private static Map<String, String> options(JsonObject row, String name) throws IOException {
    if (!row.has("options")) return Map.of();
    JsonObject options = object(row.get("options"), name + ".options");
    Map<String, String> values = new HashMap<>();
    for (String key : options.keySet()) values.put(key, text(options, key, name + ".options"));
    return Map.copyOf(values);
  }

  /** options.temperature（正の有限の数）。無ければ ReviewEvaluation の既定。 */
  private static double temperature(Map<String, String> options, String name) throws IOException {
    String text = options.get("temperature");
    if (text == null) return ReviewEvaluation.DISPLAY_TEMPERATURE;
    try {
      double temperature = Double.parseDouble(text);
      if (temperature > 0 && Double.isFinite(temperature)) return temperature;
    } catch (NumberFormatException ignored) {
      // 下で断る。
    }
    throw invalid(name + ".options.temperature must be a positive number.");
  }

  private static void requireFields(JsonObject object, Set<String> fields, String name)
      throws IOException {
    for (String key : object.keySet())
      if (!fields.contains(key)) throw invalid(name + " has an unknown field: " + key + ".");
  }

  private static JsonObject object(JsonElement element, String name) throws IOException {
    if (!(element instanceof JsonObject object)) throw invalid(name + " must be an object.");
    return object;
  }

  private static String text(JsonObject object, String key, String name) throws IOException {
    JsonElement value = object.get(key);
    if (value == null
        || !value.isJsonPrimitive()
        || !value.getAsJsonPrimitive().isString()
        || value.getAsString().isBlank())
      throw invalid(name + "." + key + " must be a non-blank string.");
    return value.getAsString();
  }

  private static IOException invalid(String message) {
    return new IOException("Invalid model catalog: " + message);
  }
}
