package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsFiles;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/** 起動時に読み取った設定ファイルのパス、上書き設定、コマンド引数を保持する。 */
public record StartupSettings(Path path, Map<String, String> overrides, String[] arguments) {

  /** 明示指定がなければ、同梱設定からローカル設定を初回だけ生成する。 */
  public static StartupSettings read(String series, String defaultResource, String[] arguments)
      throws IOException {
    String configured = System.getProperty("epsilon.settings.path");
    if (configured == null || configured.isBlank()) configured = System.getenv("EPSILON_SETTINGS");
    if (arguments.length > 0 && arguments[0].equals("--settings")) {
      if (arguments.length < 2) throw new IllegalArgumentException("--settings requires a path");
      configured = arguments[1];
      arguments = Arrays.copyOfRange(arguments, 2, arguments.length);
    }
    Path selected;
    if (configured != null && !configured.isBlank()) {
      selected = Path.of(configured);
    } else {
      selected = Path.of("config", series + ".toml");
      SettingsFiles.createDefault(selected, defaultResource);
    }
    Map<String, String> overrides = new LinkedHashMap<>();
    for (String key : System.getProperties().stringPropertyNames()) {
      if (key.startsWith("epsilon.") && !key.equals("epsilon.settings.path")) {
        overrides.put(key, System.getProperty(key));
      }
    }
    return new StartupSettings(selected, Map.copyOf(overrides), arguments);
  }
}
