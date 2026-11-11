package com.epsilon.agent.settings;

import com.google.gson.JsonElement;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * 設定を重ねるときの 1 層。設定の木と、その層の出どころを持つ。
 *
 * @param origin 層の出どころ（ファイルのパス、{@code preset production}、{@code --set}）
 * @param source 層の種類
 * @param tree 設定の木
 * @param presets 設定ファイルの先頭の {@value #PRESETS} に並べたプリセットの名前。ほかの層では空
 */
public record Layer(String origin, Source source, Node.Table tree, List<String> presets) {

  /** 設定ファイルのいちばん外のキーで、プリセットの名前を並べる。層を作るときに木から取り除く。 */
  public static final String PRESETS = "presets";

  /** 層の種類。 */
  public enum Source {
    /** jar に同梱した TOML（四人打ち・三人打ちごとの既定値とプリセット）。 */
    BUNDLED,
    /** 人が書いた設定ファイル。秘密のキーを受け付けるのはこの層だけ。 */
    FILE,
    /** {@code --set} とコマンドの引数。 */
    COMMAND_LINE,
    /** プログラムが書いた木（設定の行）。 */
    RECORD
  }

  public Layer {
    Objects.requireNonNull(origin, "origin");
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(tree, "tree");
    presets = List.copyOf(presets);
  }

  /**
   * 設定ファイルを読む。各ノードの書かれた場所は {@code <パス>:<行>} になる。
   *
   * @throws SettingsException 読めない書き方があった
   */
  public static Layer toml(Path file) throws IOException {
    return parse(file.toString(), Source.FILE, Files.readString(file, StandardCharsets.UTF_8));
  }

  /**
   * 同梱の TOML（四人打ち・三人打ちごとの既定値、プリセット）を読む。
   *
   * @param name クラスパスの中の名前（prism/presets/production.toml）
   * @param origin 層の出どころ（{@code preset production}）
   */
  public static Layer resource(String name, String origin) {
    try (InputStream stream = Layer.class.getClassLoader().getResourceAsStream(name)) {
      if (stream == null) {
        throw new IllegalStateException("Bundled settings not found: " + name);
      }
      return parse(origin, Source.BUNDLED, new String(stream.readAllBytes(), StandardCharsets.UTF_8));
    } catch (IOException failure) {
      throw new UncheckedIOException("Cannot read the bundled settings " + name, failure);
    }
  }

  /**
   * TOML の中身を読む。設定ファイルの層なら、先頭の {@value #PRESETS} を木から取り除き、プリセットの名前として持つ。
   *
   * @throws SettingsException 読めない書き方があったか、プリセットの名前が文字列の配列でない
   */
  public static Layer parse(String origin, Source source, String text) {
    Node.Table tree = Toml.read(text, origin);
    Node presets = tree.get(PRESETS);
    if (source != Source.FILE || presets == null) {
      return new Layer(origin, source, tree, List.of());
    }
    if (!(presets instanceof Node.Array array) || !array.items().stream().allMatch(Node.Text.class::isInstance)) {
      throw new SettingsException(
          List.of("Setting " + PRESETS + " at " + presets.origin() + " must be a list of preset names"));
    }
    List<String> names = array.items().stream().map(item -> ((Node.Text) item).value()).toList();
    return new Layer(origin, source, tree.without(PRESETS), names);
  }

  /**
   * Gson で読んだ JSON の木（Reviewer の models.json など）から層を作る。
   *
   * @throws SettingsException JSON がオブジェクトでないか、配列の中に null がある
   */
  public static Layer json(String origin, JsonElement tree) {
    if (!(Json.tree(tree, origin) instanceof Node.Table table)) {
      throw new SettingsException(List.of("Settings at " + origin + " must be a JSON object"));
    }
    return new Layer(origin, Source.RECORD, table, List.of());
  }

  /** プログラムが書いた設定の木（設定の行の中身）から層を作る。全てのノードの書かれた場所を、層の出どころにする。 */
  public static Layer tree(String origin, Node.Table tree) {
    return new Layer(origin, Source.RECORD, tree.relabel(origin), List.of());
  }

  /**
   * 「パス=値」の並び（{@code --set} と、コマンドの引数をキーに直したもの）から層を作る。値はコマンドラインの書き方（{@link Toml#value}）で読む。
   *
   * @throws SettingsException 同じキーが 2 回あったか、読めない値があった
   */
  public static Layer assignments(String origin, List<String> assignments) {
    return new Layer(origin, Source.COMMAND_LINE, Toml.assignments(assignments, origin), List.of());
  }

  /** いちばん外のキーを 1 つ除いた層。全体の record に結び付けないセクション（コマンドのセクションなど）を除くのに使う。 */
  public Layer without(String key) {
    return new Layer(origin, source, tree.without(key), presets);
  }
}
