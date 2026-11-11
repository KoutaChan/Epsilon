package com.epsilon.prism.app;

import com.epsilon.agent.app.Application;
import com.epsilon.agent.app.Command;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.BaseLayer;
import com.epsilon.agent.settings.Layer;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.grp.PhiControl;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.settings.PrismSettings;
import com.epsilon.prism.variant.PrismVariant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** PRISM のアプリの宣言。 */
public final class PrismApp {

  /** ルールの層を選ぶキー。 */
  private static final String VARIANT_KEY = "run.variant";

  private PrismApp() {}

  /**
   * PRISM のアプリを組み立てる。アプリは、コマンドの表（PRISM のコマンドと、jar にあるルールごとのコマンド）、全体の設定、ルールの層、設定が
   * 参照するファイルの確かめ、系譜の run の種類を持つ。
   */
  public static Application<PrismSettings> application() {
    List<Class<? extends Command<? super PrismSettings>>> commands = new ArrayList<>();
    commands.add(PrismCommand.class);
    for (PrismVariant variant : PrismVariant.all()) {
      commands.addAll(variant.commands());
    }
    return new Application<>(
        "prism",
        PrismSettings.class,
        commands,
        new BaseLayer(VARIANT_KEY, PrismApp::variantLayer),
        PrismApp::problems,
        PrismRun.LINEAGE);
  }

  /** run の最新の設定の行から設定を読む。 */
  public static PrismSettings latest(Path run) throws IOException {
    return new SettingsRows(new RunDirectory(run)).latest().bind(PrismSettings.class).value();
  }

  /** 名前で選んだルールの層を返す。 */
  private static Layer variantLayer(String name) {
    Variant variant = Variant.valueOf(name);
    try {
      return PrismVariant.of(variant).settings();
    } catch (IllegalStateException missing) {
      throw new IllegalArgumentException(missing.getMessage(), missing);
    }
  }

  /** 設定が参照するファイルを確かめる。確かめるのは、Φ の GRP と表、評価の卓で相手にするチェックポイント。 */
  private static List<String> problems(PrismSettings settings) {
    List<String> problems =
        new ArrayList<>(PhiControl.problems(new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed())));
    for (Map.Entry<String, TableSpec> entry : settings.evaluation().tables().entrySet()) {
      if (TableSpec.kind(entry.getValue().opponent()) == TableSpec.Opponent.CHECKPOINT) {
        Series.Spec spec = Series.parse(entry.getValue().opponent());
        if (!Files.isDirectory(spec.checkpoint())) {
          problems.add("The opponent checkpoint of evaluation table " + entry.getKey() + " does not exist: " + spec.checkpoint());
        }
      }
    }
    return problems;
  }
}
