package com.epsilon.prism;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * テストで使う三人打ちの PRISM の設定。三人打ちの層と注釈の既定値の上に、テストの run の乱数の種、プリセット、「キーのパス=値」の上書きを
 * この順で重ねる（{@code settings create} と同じ重ね方）。上書きは 1 つずつ別の層にし、同じキーは後の上書きが勝つ。
 */
public final class ThreePlayerSettings {

  /** テストの run の乱数の種。 */
  public static final long SEED = 20261010L;

  private ThreePlayerSettings() {}

  /** 上書きを重ねた設定。 */
  public static PrismSettings of(String... assignments) {
    return bind(List.of(), assignments).value();
  }

  /**
   * プリセットと上書きを重ねた設定。
   *
   * @param presets 同梱のプリセットの名前（重ねる順）
   * @param assignments 「キーのパス=値」の上書き
   */
  public static Settings<PrismSettings> bind(List<String> presets, String... assignments) {
    List<Layer> layers = new ArrayList<>();
    layers.add(Layer.assignments("test run", List.of("run.variant=TENHOU_3P", "run.seed=" + SEED)));
    for (String preset : presets) {
      layers.add(PrismApp.application().preset(preset));
    }
    for (String assignment : assignments) {
      layers.add(Layer.assignments("test", List.of(assignment)));
    }
    return Settings.bind(PrismSettings.class, PrismApp.application().stack(layers));
  }

  /** run のディレクトリに、プリセットと上書きを重ねた設定の 1 行目を書く（{@code settings create} と同じ行）。 */
  public static PrismSettings create(Path run, List<String> presets, String... assignments) throws IOException {
    Settings<PrismSettings> settings = bind(presets, assignments);
    try (SettingsRows.Locked locked = new SettingsRows(new RunDirectory(run)).lock()) {
      locked.append(1, "test", SettingsRow.CREATED, List.of(), settings.tree());
    }
    return settings.value();
  }

  /** run の設定に、プリセットと上書きを重ねた設定の行を 1 つ足す（前の行の値は引き継がない）。 */
  public static PrismSettings append(Path run, List<String> presets, String... assignments) throws IOException {
    Settings<PrismSettings> settings = bind(presets, assignments);
    try (SettingsRows.Locked locked = new SettingsRows(new RunDirectory(run)).lock()) {
      locked.append(1, "test", "test", List.of(), settings.tree());
    }
    return settings.value();
  }
}
