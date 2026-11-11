package com.epsilon.reviewer.model;

import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * models.json の中身。起動時に設定ファイルと同じ仕組みで読み、知らないフィールド、型や範囲の誤りを全て集めてから起動を失敗させる。
 * 同じモデルのバージョンが 2 回出たときと、既定のモデルが一覧に無いときも誤りにする。
 *
 * @param models モデルの一覧。画面に出す順
 * @param defaultModel 画面で最初に選ぶモデル。無ければ null
 */
public record ModelsFile(
    @Key(doc = "モデルの一覧（画面に出す順）", presence = Presence.REQUIRED, change = Change.MACHINE) List<Model> models,
    @Key(doc = "画面で最初に選んでおくモデル", presence = Presence.ABSENT, change = Change.MACHINE) Reference defaultModel) {

  private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");

  public ModelsFile {
    Set<ModelKey> keys = new HashSet<>();
    for (int index = 0; index < models.size(); index++) {
      ModelKey key = models.get(index).key();
      if (!keys.add(key)) {
        throw new InvalidSetting(
            "models", "models[" + index + "] repeats the model " + key.modelId() + " " + key.revision());
      }
    }
    if (defaultModel != null && !keys.contains(defaultModel.key())) {
      throw new InvalidSetting("defaultModel", "names no model of the catalog");
    }
    models = List.copyOf(models);
  }

  /**
   * モデル 1 つ。
   *
   * @param series 系列の名前（{@link Series#id}）
   * @param variant モデルが読む牌譜のルール
   * @param checkpoint チェックポイントのフォルダー（models.json のあるフォルダーからの相対パスか、絶対パス）。チェックポイントを使わない系列は null
   * @param options 方策を開く引数と表示の温度
   * @param sha256 チェックポイントのフォルダーの SHA-256。初めて開くときに、これで中身を確かめる。無ければ null
   */
  public record Model(
      @Key(doc = "モデルの名前", presence = Presence.REQUIRED) String modelId,
      @Key(doc = "重みを見分ける名前（重みを替えたら変える）", presence = Presence.REQUIRED) String revision,
      @Key(doc = "画面に出す名前", presence = Presence.REQUIRED) String displayName,
      @Key(doc = "画面に出すバージョン", presence = Presence.REQUIRED) String version,
      @Key(doc = "系列（epsilon-prism・epsilon-nano・epsilon・rulebot）", presence = Presence.REQUIRED) String series,
      @Key(doc = "モデルが読む牌譜のルール", presence = Presence.REQUIRED) Variant variant,
      @Key(doc = "チェックポイントのフォルダー（models.json のあるフォルダーからの相対パスか、絶対パス）", presence = Presence.ABSENT)
          Path checkpoint,
      @Key(doc = "方策を開く引数と表示の温度") ModelOptions options,
      @Key(doc = "チェックポイントのフォルダーの SHA-256", presence = Presence.ABSENT) String sha256) {

    public Model {
      requireText("modelId", modelId);
      requireText("revision", revision);
      requireText("displayName", displayName);
      requireText("version", version);
      Series kind;
      try {
        kind = Series.ofId(series);
      } catch (IllegalArgumentException unknown) {
        throw new InvalidSetting(
            "series",
            "expected one of " + Arrays.stream(Series.values()).map(Series::id).toList());
      }
      if ((checkpoint != null) != kind.checkpointed()) {
        throw new InvalidSetting(
            "checkpoint", "series " + series + (kind.checkpointed() ? " needs a checkpoint" : " takes no checkpoint"));
      }
      if (sha256 != null && !SHA256.matcher(sha256).matches()) {
        throw new InvalidSetting("sha256", "expected 64 lowercase hexadecimal digits");
      }
      if (sha256 != null && checkpoint == null) {
        throw new InvalidSetting("sha256", "needs a checkpoint");
      }
    }

    private static void requireText(String field, String text) {
      if (text.isBlank()) {
        throw new InvalidSetting(field, "must not be blank");
      }
    }

    /** モデルのキー。 */
    ModelKey key() {
      return new ModelKey(modelId, revision);
    }
  }

  /**
   * カタログの中のモデルを指す値。
   *
   * @param revision 重みを見分ける名前
   */
  public record Reference(
      @Key(doc = "モデルの名前", presence = Presence.REQUIRED) String modelId,
      @Key(doc = "重みを見分ける名前", presence = Presence.REQUIRED) String revision) {

    /** モデルのキー。 */
    ModelKey key() {
      return new ModelKey(modelId, revision);
    }
  }
}
