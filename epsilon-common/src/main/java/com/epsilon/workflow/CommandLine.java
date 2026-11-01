package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsLoader;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** モデル系列に共通するコマンドライン引数の処理と、コマンドの呼び出し・終了コードの判定を行う。 */
public final class CommandLine {
  private static final Logger log = LoggerFactory.getLogger(CommandLine.class);

  private CommandLine() {}

  public static int run(String[] args, WorkflowRegistry workflows, SettingsLoader settings) {
    return run(args, workflows, settings, log::info, true);
  }

  /** 設定スナップショットを維持し、ヘルプの出力先を明示する。 */
  public static int run(
      String[] args, WorkflowRegistry workflows, SettingsLoader settings, PrintStream output) {
    Objects.requireNonNull(output, "output");
    return run(args, workflows, settings, output::println, true);
  }

  /** 系列固有の設定を各ワークフローで読む CLI を、設定ファイルの初期化なしで実行する。 */
  public static int run(String[] args, WorkflowRegistry workflows, PrintStream output) {
    Objects.requireNonNull(output, "output");
    return run(args, workflows, SettingsLoader.of(Map.of()), output::println, false);
  }

  private static int run(
      String[] args,
      WorkflowRegistry workflows,
      SettingsLoader settings,
      Consumer<String> usage,
      boolean commonSettings) {
    String command = args.length == 0 ? "help" : args[0];
    boolean help = command.equals("--help") || command.equals("-h") || command.equals("help");
    if (help || !workflows.handles(command)) {
      if (commonSettings) {
        usage.accept("Common option: --settings <file> <command> [arguments]");
      }
      workflows.usageLines().forEach(usage);
      return help ? 0 : 2;
    }
    try {
      String[] arguments = args.length == 0 ? args : Arrays.copyOfRange(args, 1, args.length);
      return workflows.execute(command, arguments, settings) ? 0 : 1;
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      log.info("Command interrupted: {}", command);
      return 130;
    } catch (Exception failure) {
      log.error("Command failed: {}", command, failure);
      return 1;
    }
  }
}
