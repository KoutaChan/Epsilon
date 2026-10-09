package com.epsilon.cfr.config.settings;

import com.epsilon.config.settings.SettingsLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/** CFRの同梱既定値、選択した設定ファイル、明示的な上書き値を共通ローダーで合成する。 */
public final class EpsilonSettings {
  public static final String DEFAULT_RESOURCE = "cfr/settings.toml";

  private EpsilonSettings() {}

  public static SettingsLoader defaults() {
    return Defaults.VALUE;
  }

  public static SettingsLoader of(Map<String, String> overrides) {
    return defaults().withOverrides(overrides);
  }

  public static SettingsLoader load(Path path, Map<String, String> overrides) throws IOException {
    return SettingsLoader.load(defaults(), path, overrides);
  }

  private static final class Defaults {
    private static final SettingsLoader VALUE = SettingsLoader.fromResource(DEFAULT_RESOURCE);
  }
}
