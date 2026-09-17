package com.epsilon.major.config.settings;

import com.epsilon.config.settings.SettingsLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** この系列の同梱既定値を選び、起動時の設定スナップショットを作る。 */
public final class EpsilonSettings {
  /** この系列に同梱する既定設定ファイルの名前。 */
  public static final String DEFAULT_RESOURCE = "major/settings.toml";

  private EpsilonSettings() {}

  /** 実行環境に依存しない、この系列の同梱既定値。 */
  public static SettingsLoader defaults() {
    return Defaults.VALUE;
  }

  /** 同梱既定値に、明示された設定値を上書きする。 */
  public static SettingsLoader of(Map<String, String> overrides) {
    return defaults().withOverrides(overrides);
  }

  /** 同梱既定値に、指定ファイルの設定と明示された設定値を順に上書きする。 */
  public static SettingsLoader load(Path path, Map<String, String> overrides) throws IOException {
    return SettingsLoader.load(defaults(), path, overrides);
  }

  private static final class Defaults {
    private static final SettingsLoader VALUE = SettingsLoader.fromResource(DEFAULT_RESOURCE);
  }
}
