package com.epsilon.cfr;

import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.workflow.CommandLine;
import com.epsilon.workflow.StartupSettings;
import java.io.IOException;
import java.io.PrintStream;

/** LCFRのコマンドライン入口。 */
public final class Main {
  private Main() {}

  public static void main(String[] args) throws IOException {
    int status = run(args);
    if (status != 0) System.exit(status);
  }

  public static int run(String[] args) throws IOException {
    return run(args, System.out);
  }

  public static int run(String[] args, PrintStream output) throws IOException {
    StartupSettings startup =
        StartupSettings.read("epsilon-cfr", EpsilonSettings.DEFAULT_RESOURCE, args);
    SettingsLoader settings = EpsilonSettings.load(startup.path(), startup.overrides());
    return CommandLine.run(startup.arguments(), Workflows.create(output), settings, output);
  }
}
