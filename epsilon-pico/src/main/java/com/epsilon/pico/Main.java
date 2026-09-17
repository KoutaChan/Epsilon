package com.epsilon.pico;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.workflow.CommandLine;
import com.epsilon.workflow.StartupSettings;

/** epsilon-pico を起動し、今回の実行で使用する設定を読み込んで保持する。 */
public final class Main {
  private Main() {}

  public static int run(String[] arguments) throws java.io.IOException {
    StartupSettings startup =
        StartupSettings.read("epsilon-pico", EpsilonSettings.DEFAULT_RESOURCE, arguments);
    SettingsLoader settings = EpsilonSettings.load(startup.path(), startup.overrides());
    return CommandLine.run(startup.arguments(), Workflows.create(), settings);
  }

  public static void main(String[] arguments) throws java.io.IOException {
    int status = run(arguments);
    if (status != 0) System.exit(status);
  }
}
