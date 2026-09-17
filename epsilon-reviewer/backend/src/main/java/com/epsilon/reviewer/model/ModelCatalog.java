package com.epsilon.reviewer.model;

import com.epsilon.reviewer.ApiError;
import com.epsilon.reviewer.ApiException;
import com.google.gson.*;
import java.io.*;
import java.nio.file.*;
import java.security.*;
import java.util.*;

/** 起動時に読み込んだモデルの識別情報とチェックポイントを固定して保持する一覧。 */
public final class ModelCatalog {
  private final List<Entry> entries;
  private final ModelKey defaultModel;

  public ModelCatalog(Path config) throws IOException {
    if (config == null) {
      entries = List.of();
      defaultModel = null;
      return;
    }
    JsonObject document;
    try (Reader reader = Files.newBufferedReader(config)) {
      document = JsonParser.parseReader(reader).getAsJsonObject();
    } catch (RuntimeException e) {
      throw new IOException("Invalid model catalog JSON.", e);
    }
    var loaded = new ArrayList<Entry>();
    var keys = new HashSet<String>();
    for (JsonElement item : document.getAsJsonArray("models")) {
      JsonObject row = item.getAsJsonObject();
      String id = readRequiredPrimitiveText(row, "modelId");
      String revision = readRequiredPrimitiveText(row, "revision");
      if (!keys.add(id + "\n" + revision))
        throw new IOException("Duplicate model ID and revision.");
      Path checkpoint =
          config
              .toAbsolutePath()
              .getParent()
              .resolve(readRequiredPrimitiveText(row, "checkpoint"))
              .normalize();
      Map<String, String> options = new LinkedHashMap<>();
      if (row.has("options"))
        row.getAsJsonObject("options")
            .entrySet()
            .forEach(e -> options.put(e.getKey(), e.getValue().getAsString()));
      var definition =
          new ModelDefinition(
              id,
              revision,
              readRequiredPrimitiveText(row, "displayName"),
              readRequiredPrimitiveText(row, "version"),
              readRequiredPrimitiveText(row, "series"),
              checkpoint,
              options);
      ApiError unavailable = null;
      if (!Set.of("epsilon", "epsilon-nano", "epsilon-pico").contains(definition.series()))
        unavailable = new ApiError("unsupported_series", "Model series is not supported.");
      if (!Files.isDirectory(checkpoint) || !Files.isReadable(checkpoint))
        unavailable = new ApiError("model_unavailable", "Checkpoint is unavailable.");
      byte[] digest = unavailable == null ? digest(checkpoint) : null;
      loaded.add(new Entry(definition, digest, unavailable));
    }
    entries = loaded;
    if (document.has("defaultModel")) {
      JsonObject selected = document.getAsJsonObject("defaultModel");
      defaultModel =
          new ModelKey(
              readRequiredPrimitiveText(selected, "modelId"),
              readRequiredPrimitiveText(selected, "revision"));
      if (entries.stream()
          .noneMatch(
              entry ->
                  entry.definition.modelId().equals(defaultModel.modelId())
                      && entry.definition.revision().equals(defaultModel.revision())))
        throw new IOException("Default model is not present in the catalog.");
    } else defaultModel = null;
  }

  public Catalog publicCatalog() {
    var models = new ArrayList<AvailableModel>(entries.size());
    for (Entry entry : entries) {
      ModelDefinition model = entry.definition;
      ApiError unavailable = unavailableReason(entry);
      models.add(
          new AvailableModel(
              model.modelId(),
              model.revision(),
              model.displayName(),
              model.version(),
              model.series(),
              unavailable == null ? "available" : "unavailable",
              unavailable == null ? null : unavailable.code(),
              unavailable == null ? null : unavailable.message()));
    }
    return new Catalog(models, defaultModel);
  }

  public ModelDefinition requireAvailableModelRevision(String id, String revision) {
    Entry entry = requireCatalogEntry(id, revision);
    ApiError unavailable = unavailableReason(entry);
    if (unavailable != null) throw new ApiException(409, unavailable.code(), unavailable.message());
    return entry.definition;
  }

  public Lease acquire(String id, String revision, Path temporaryDirectory) throws IOException {
    Entry entry = requireCatalogEntry(id, revision);
    if (entry.unavailable != null)
      throw new ApiException(409, entry.unavailable.code(), entry.unavailable.message());
    Files.createDirectories(temporaryDirectory);
    Path copy = Files.createTempDirectory(temporaryDirectory, "model-");
    try {
      for (Path file : files(entry.definition.checkpoint())) {
        Path target = copy.resolve(entry.definition.checkpoint().relativize(file));
        Files.createDirectories(target.getParent());
        Files.copy(file, target);
      }
      if (!MessageDigest.isEqual(entry.digest, digest(copy)))
        throw new ApiException(
            409, "model_changed", "Checkpoint changed; register it with a new revision.");
      ModelDefinition d = entry.definition;
      return new Lease(
          new ModelDefinition(
              d.modelId(),
              d.revision(),
              d.displayName(),
              d.version(),
              d.series(),
              copy,
              d.options()));
    } catch (Throwable e) {
      removeTree(copy);
      throw e;
    }
  }

  private Entry requireCatalogEntry(String id, String revision) {
    return entries.stream()
        .filter(e -> e.definition.modelId().equals(id) && e.definition.revision().equals(revision))
        .findFirst()
        .orElseThrow(
            () ->
                new ApiException(
                    409, "model_unavailable", "Selected model ID and revision were not found."));
  }

  private ApiError unavailableReason(Entry entry) {
    if (entry.unavailable != null) return entry.unavailable;
    try {
      if (!MessageDigest.isEqual(entry.digest, digest(entry.definition.checkpoint())))
        return new ApiError("model_changed", "Checkpoint changed after catalog registration.");
      return null;
    } catch (IOException failure) {
      return new ApiError("model_unavailable", "Checkpoint is unavailable.");
    }
  }

  private static byte[] digest(Path path) throws IOException {
    try {
      var digest = MessageDigest.getInstance("SHA-256");
      for (Path file : files(path)) {
        digest.update(
            path.relativize(file)
                .toString()
                .replace('\\', '/')
                .getBytes(java.nio.charset.StandardCharsets.UTF_8));
        digest.update((byte) 0);
        digest.update(java.nio.ByteBuffer.allocate(Long.BYTES).putLong(Files.size(file)).array());
        try (InputStream in = Files.newInputStream(file);
            var stream = new DigestInputStream(in, digest)) {
          stream.transferTo(OutputStream.nullOutputStream());
        }
      }
      return digest.digest();
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }

  private static List<Path> files(Path directory) throws IOException {
    if (Files.isSymbolicLink(directory) || !Files.isDirectory(directory))
      throw new IOException("Checkpoint must be a regular directory.");
    try (var paths = Files.walk(directory)) {
      var all = paths.sorted().toList();
      if (all.stream().anyMatch(Files::isSymbolicLink))
        throw new IOException("Symbolic links are not allowed inside a checkpoint.");
      var files = all.stream().filter(Files::isRegularFile).toList();
      if (files.isEmpty()) throw new IOException("Checkpoint contains no files.");
      return files;
    }
  }

  public static void cleanSnapshots(Path temporaryDirectory) throws IOException {
    if (!Files.isDirectory(temporaryDirectory)) return;
    try (var entries = Files.list(temporaryDirectory)) {
      for (Path entry :
          entries.filter(p -> p.getFileName().toString().startsWith("model-")).toList())
        removeTree(entry);
    }
  }

  private static void removeTree(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(path);
    }
  }

  private static String readRequiredPrimitiveText(JsonObject value, String key) throws IOException {
    if (!value.has(key)
        || !value.get(key).isJsonPrimitive()
        || value.get(key).getAsString().isBlank())
      throw new IOException("Model catalog requires field: " + key + ".");
    return value.get(key).getAsString();
  }

  public record ModelKey(String modelId, String revision) {}

  public record AvailableModel(
      String modelId,
      String revision,
      String displayName,
      String version,
      String series,
      String availability,
      String reasonCode,
      String reason) {}

  public record Catalog(List<AvailableModel> models, ModelKey defaultModel) {}

  private record Entry(ModelDefinition definition, byte[] digest, ApiError unavailable) {}

  public record Lease(ModelDefinition definition) implements AutoCloseable {
    @Override
    public void close() throws IOException {
      removeTree(definition.checkpoint());
    }
  }
}
