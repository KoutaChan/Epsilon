package com.epsilon.reviewer.model;

import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 起動時に models.json から読んだモデルの一覧。ファイルの形は {@link ModelsFile} で、設定ファイルと同じ仕組みで読む。知らないフィールド、
 * 型や範囲の誤り、知らない系列やルール、重複、一覧に無い既定のモデルがあれば、誤りを全て集めてから起動を失敗させる。使えるかどうかは、
 * 中身のハッシュを作らずに決める。画面に出すカタログは作っておき、使えるかどうかが変わったときだけ作り直す。チェックポイントの
 * 相対パスは、models.json のあるフォルダーを起点にする。
 */
public final class ModelCatalog {
  private static final Logger LOG = LoggerFactory.getLogger(ModelCatalog.class);

  /** models.json の大きさの上限（バイト）。 */
  private static final int FILE_LIMIT = 1024 * 1024;

  /** models.json の順に並べたモデル。チェックポイントが変わったと分かったら、値を差し替える。 */
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

  /**
   * models.json を読む。チェックポイントを読めないモデルと、その系列を扱うクラスがクラスパスに無いモデルは、起動を止めずに使えないモデルとして載せる。
   *
   * @throws SettingsException models.json の形が正しくないとき（誤り 1 つを 1 行に書き、キーのパスと書いた場所を添える）
   * @throws IOException models.json が読めないとき
   */
  public static ModelCatalog load(Path modelsJson, ModelFamilies families) throws IOException {
    if (Files.size(modelsJson) > FILE_LIMIT)
      throw new SettingsException(
          List.of("Model catalog " + modelsJson + " exceeds the limit of " + FILE_LIMIT + " bytes"));
    String origin = modelsJson.toString();
    Node.Table tree = Json.read(Files.readString(modelsJson, StandardCharsets.UTF_8), origin);
    ModelsFile file =
        Settings.bind(ModelsFile.class, List.of(new Layer(origin, Layer.Source.FILE, tree, List.of()))).value();
    Path base = modelsJson.toAbsolutePath().getParent();
    Map<ModelKey, ModelEntry> entries = new LinkedHashMap<>();
    Map<ModelKey, CheckpointFingerprint> fingerprints = new HashMap<>();
    for (ModelsFile.Model model : file.models()) {
      ModelKey key = model.key();
      Series series = Series.ofId(model.series());
      Path checkpoint = model.checkpoint() == null ? null : base.resolve(model.checkpoint()).normalize();
      boolean available = true;
      if (families.find(series).isEmpty()) {
        available = false;
        LOG.warn(
            "Model {} {} has no model family for series {}.",
            key.modelId(),
            key.revision(),
            series.id());
      } else if (checkpoint != null)
        try {
          fingerprints.put(key, CheckpointFingerprint.of(checkpoint));
        } catch (IOException | UncheckedIOException failure) {
          available = false;
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
              model.displayName(),
              model.version(),
              series,
              checkpoint,
              model.options().policy(),
              model.variant(),
              model.sha256(),
              model.options().temperature(),
              available));
    }
    ModelKey defaultModel = file.defaultModel() == null ? null : file.defaultModel().key();
    return new ModelCatalog(entries, fingerprints, defaultModel);
  }

  /** models.json を持たないサーバの、空のカタログ。 */
  public static ModelCatalog empty() {
    return new ModelCatalog(new LinkedHashMap<>(), Map.of(), null);
  }

  /** 作っておいたカタログを返す（ハッシュは作らない）。 */
  public CatalogView view() {
    return view;
  }

  /** @throws ReviewerException model_unavailable（カタログに無いか、使えないとき） */
  public ModelEntry require(ModelKey key) {
    ModelEntry entry;
    synchronized (this) {
      entry = entries.get(key);
    }
    if (entry == null || !entry.available())
      throw new ReviewerException(ErrorCode.MODEL_UNAVAILABLE);
    return entry;
  }

  /**
   * モデルの今の revision を返す（models.json で最後に書いた revision）。使えるかどうかは見ない。
   *
   * @return 今の revision を指すキー。カタログにそのモデルが無ければ空
   */
  public synchronized Optional<ModelKey> current(String modelId) {
    ModelKey found = null;
    for (ModelKey key : entries.keySet()) if (key.modelId().equals(modelId)) found = key;
    return Optional.ofNullable(found);
  }

  /**
   * もしもで使うバージョンを選ぶ。解析を作ったバージョンが使えればそれを、使えなければそのモデルの今のバージョンを返す。同じモデルの
   * 複数のバージョンをカタログに並べたとき、古いバージョンの解析へ新しいバージョンで答えないようにするため。
   *
   * @param written 解析を作ったモデルのバージョン
   * @return もしもで使うモデルのキー
   * @throws ReviewerException model_unavailable（カタログにそのモデルが無いとき）
   */
  public ModelKey scenarioModel(ModelKey written) {
    ModelEntry entry;
    synchronized (this) {
      entry = entries.get(written);
    }
    if (entry != null && entry.available()) return written;
    return current(written.modelId())
        .orElseThrow(() -> new ReviewerException(ErrorCode.MODEL_UNAVAILABLE));
  }

  /** チェックポイントが登録したときと違うと分かったモデルを、以後使えなくする。 */
  void markChanged(ModelKey key) {
    synchronized (this) {
      entries.computeIfPresent(key, (ignored, entry) -> entry.unavailable());
      view = buildView();
    }
    LOG.warn("Model {} {} changed after it was registered.", key.modelId(), key.revision());
  }

  /** 起動時に取ったフィンガープリント。チェックポイントを持ち、使えるモデルにだけある。 */
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
              entry.variant().seats(),
              entry.available()));
    return new CatalogView(List.copyOf(models), defaultModel);
  }
}
