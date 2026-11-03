package com.epsilon.prism;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.workflow.CommandLine;
import com.epsilon.workflow.StartupSettings;
import java.io.IOException;

/** epsilon-prism を起動し、今回の実行で使用する設定を読み込んで保持する。 */
public final class Main {
  private Main() {}

  public static int run(String[] arguments) throws IOException {
    StartupSettings startup =
        StartupSettings.read("epsilon-prism", EpsilonSettings.DEFAULT_RESOURCE, arguments);
    SettingsLoader settings = EpsilonSettings.load(startup.path(), startup.overrides());
    return CommandLine.run(startup.arguments(), Workflows.create(), settings);
  }

  /** 終わりに必ず System.exit を呼ぶ（残ったスレッドで JVM が終われなくならないように）。 */
  public static void main(String[] arguments) throws IOException {
    System.exit(run(arguments));
  }
}
