package com.epsilon.agent.app;

import com.epsilon.agent.run.RunKind;
import com.epsilon.agent.settings.BaseLayer;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * アプリの宣言。入口（{@link App}）は、これを見て引数を読み、コマンドを選んで実行する。
 *
 * <p>プリセットは、クラスパスの {@code <アプリの名前>/presets/<プリセットの名前>.toml} に同梱する。
 *
 * @param name アプリの名前（{@code prism} など）。使い方の表示と、プリセットの場所に使う
 * @param schema アプリ全体の設定の record
 * @param commands コマンドの sealed インターフェース。アプリのものと、あればルールごとのもの。アプリ全体の設定を読まない
 *     コマンドのインターフェース（{@code Command<Record>}）も渡せる。使い方の表示はこの順に並び、組み込みのコマンドが先に来る
 * @param base ルールごとの既定の層の選び方。ルールで既定値が変わらないアプリでは null
 * @param check 設定が参照するファイルの確かめ。{@code settings check} と {@code settings change} が呼ぶ
 * @param runKind アプリの run の種類（{@code settings change} が、変更の前と後で、次に公開するバージョンのジョブを比べる）。run を
 *     持たないアプリでは null
 */
public record Application<R extends Record>(
    String name,
    Class<R> schema,
    List<? extends Class<? extends Command<? super R>>> commands,
    BaseLayer base,
    Check<R> check,
    RunKind<R, ?, ?, ?> runKind) {

  private static final Pattern PRESET = Pattern.compile("[a-z0-9][a-z0-9-]*");

  /** 設定が参照するファイル（checkpoint と sha、Φ の項目など）を確かめる。設定をバインドするときには見ない点を見る。 */
  @FunctionalInterface
  public interface Check<R extends Record> {
    /** @return 見つけた問題。英語で 1 つにつき 1 行。無ければ空 */
    List<String> problems(R settings) throws IOException;
  }

  public Application {
    if (name == null || !PRESET.matcher(name).matches()) {
      throw new IllegalArgumentException("Invalid application name: " + name);
    }
    Objects.requireNonNull(schema, "schema");
    commands = List.copyOf(commands);
    Objects.requireNonNull(check, "check");
  }

  /** ルールごとの既定の層と run の種類が無く、参照するファイルも確かめないアプリを作る。 */
  public static <R extends Record> Application<R> of(
      String name, Class<R> schema, List<? extends Class<? extends Command<? super R>>> commands) {
    return new Application<>(name, schema, commands, null, settings -> List.of(), null);
  }

  public Application<R> withCheck(Check<R> check) {
    return new Application<>(name, schema, commands, base, check, runKind);
  }

  /**
   * 同梱のプリセットを読む。
   *
   * @throws SettingsException そのプリセットが無い
   */
  public Layer preset(String preset) {
    String resource = name + "/presets/" + preset + ".toml";
    if (!PRESET.matcher(preset).matches() || schema.getClassLoader().getResource(resource) == null) {
      throw new SettingsException(List.of("Unknown preset " + preset));
    }
    return Layer.resource(resource, "preset " + preset);
  }

  /**
   * 層のいちばん下に、ルールごとの既定の層を入れる。
   *
   * @param layers 下から順に並べた層
   * @throws SettingsException ルールを選ぶキーが無いか、その値に合う層が無い
   */
  public List<Layer> stack(List<Layer> layers) {
    return base == null ? layers : Settings.withBase(schema, layers, base);
  }

  /** どれかの層がルールを選ぶキーを書いていれば、ルールごとの既定の層を入れる。どの層も書いていなければそのまま返す。ひな形で使う。 */
  List<Layer> stackIfChosen(List<Layer> layers) {
    if (base == null) {
      return layers;
    }
    String[] keys = base.key().split("\\.");
    for (Layer layer : layers) {
      Node node = layer.tree();
      for (String key : keys) {
        node = node instanceof Node.Table table ? table.get(key) : null;
      }
      if (node != null) {
        return stack(layers);
      }
    }
    return layers;
  }
}
