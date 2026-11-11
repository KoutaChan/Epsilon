package com.epsilon.agent.app;

import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.NoSuchFileException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * アプリの入口。引数を読んでコマンドを 1 つ実行し、終了コードを返す。標準出力と {@code System.exit} を使うのはこのクラスだけで、
 * ほかの出力は全て SLF4J に書く。
 *
 * <p>ログには、コマンドごとに始めと終わりの 2 行を書く。始めの行には引数を書き、秘密の値は伏せる。終わりの行には終了の種類、
 * 経過時間、要約を書く。設定と引数の誤りは 1 行に 1 つずつ書いて {@link Exit#USAGE} で終わる。例外では {@link Exit#FAILED}、
 * 割り込みでは {@link Exit#INTERRUPTED} で終わる。
 */
public final class App {

  private static final Logger log = LoggerFactory.getLogger(App.class);

  private App() {}

  /** main から呼ぶ。シグナルのハンドラを入れてコマンドを実行し、終了コードで JVM を終える。 */
  public static <R extends Record> void main(Application<R> application, String[] args) {
    Signals signals = new Signals();
    signals.install(App::exit);
    int code;
    try {
      code = run(application, args, System.out, signals);
    } catch (Throwable failure) {
      log.error("Command failed", failure);
      code = Exit.FAILED.code();
    }
    exit(code);
  }

  /**
   * コマンドを 1 つ実行する。シグナルのハンドラは入れない。
   *
   * @return 終了コード
   */
  public static <R extends Record> int run(Application<R> application, String[] args) {
    return run(application, args, System.out, new Signals());
  }

  /**
   * コマンドを 1 つ実行し、結果を渡された出力先に書く。シグナルのハンドラは入れない。標準出力でフレームを送る子プロセスは、
   * {@code System.out} を標準エラーへ向けた後で、元の標準出力をここに渡す。
   *
   * @param out コマンドの結果と使い方の表示を書く先
   * @return 終了コード
   */
  public static <R extends Record> int run(Application<R> application, String[] args, PrintStream out) {
    return run(application, args, out, new Signals());
  }

  static <R extends Record> int run(Application<R> application, String[] args, PrintStream out, Signals signals) {
    Catalog catalog = Catalog.of(application);
    Invocation invocation;
    try {
      invocation = Invocation.parse(args);
    } catch (SettingsException invalid) {
      invalid.problems().forEach(log::error);
      return Exit.USAGE.code();
    }
    if (invocation.words().isEmpty()) {
      log.error("No command given");
      out.print(Usage.commands(application, catalog));
      out.flush();
      return Exit.USAGE.code();
    }
    Catalog.Match match = catalog.find(invocation.words());
    if (match == null) {
      String unknown = String.join(" ", invocation.words());
      log.error(
          "Unknown command {}{}; run '{} help' for the commands",
          unknown,
          Catalog.suggestion(unknown, catalog.entries().stream().map(Catalog.Entry::name).toList(), " (did you mean ", "?)"),
          application.name());
      return Exit.USAGE.code();
    }
    Catalog.Entry entry = match.entry();
    log.info("Command started: {} {}", application.name(), invocation.describe(match, Invocation.secretPatterns(application.schema(), entry)));
    long started = System.nanoTime();
    Outcome outcome = execute(application, catalog, match, invocation, out, signals);
    out.flush();
    log.info(
        "Command finished: {} {} {} in {} s: {}",
        application.name(),
        entry.name(),
        outcome.exit(),
        String.format(Locale.ROOT, "%.1f", (System.nanoTime() - started) / 1e9),
        outcome.summary());
    return outcome.exit().code();
  }

  private static <R extends Record> Outcome execute(
      Application<R> application,
      Catalog catalog,
      Catalog.Match match,
      Invocation invocation,
      PrintStream out,
      Signals signals) {
    Catalog.Entry entry = match.entry();
    try {
      Layers layers = layers(application, catalog, invocation, entry);
      Record arguments = invocation.command(match, layers.sections());
      @SuppressWarnings("unchecked")
      Command<R> command = (Command<R>) arguments;
      Context<R> context = new Invoked<>(application, entry.named().reads(), layers, out, signals);
      return Objects.requireNonNull(command.run(context), "outcome");
    } catch (SettingsException invalid) {
      invalid.problems().forEach(log::error);
      return new Outcome(Exit.USAGE, invalid.problems().size() + " problem(s) in the arguments or settings");
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      return new Outcome(Exit.INTERRUPTED, "interrupted");
    } catch (Exception failure) {
      log.error("Command {} failed", entry.name(), failure);
      return new Outcome(Exit.FAILED, String.valueOf(failure));
    }
  }

  /**
   * コマンドラインから組んだ設定の層。
   *
   * @param sections コマンドのセクションを含めた層。コマンドの引数をここからバインドする
   * @param settings コマンドのセクションを除いた層。全体の設定をここからバインドする
   * @param presets プリセットの名前
   */
  private record Layers(List<Layer> sections, List<Layer> settings, List<String> presets) {}

  private static Layers layers(Application<?> application, Catalog catalog, Invocation invocation, Catalog.Entry entry)
      throws IOException {
    if (entry.named().reads() == Reads.RUN) {
      if (invocation.hasLayers()) {
        throw new SettingsException(
            List.of(
                "Use '"
                    + application.name()
                    + " settings change --run DIR' to change a running lineage ("
                    + entry.name()
                    + " takes no --settings, --preset or --set)"));
      }
      return new Layers(List.of(), List.of(), List.of());
    }
    List<String> problems = new ArrayList<>();
    Layer file = null;
    if (invocation.settings() != null) {
      try {
        file = Layer.toml(invocation.settings());
      } catch (NoSuchFileException missing) {
        problems.add("Settings file " + invocation.settings() + " does not exist");
      } catch (SettingsException unreadable) {
        problems.addAll(unreadable.problems());
      }
    }
    List<String> presets = new ArrayList<>();
    if (file != null) {
      presets.addAll(file.presets());
    }
    presets.addAll(invocation.presets());
    List<Layer> layers = new ArrayList<>();
    for (String preset : presets) {
      try {
        layers.add(application.preset(preset));
      } catch (SettingsException unknown) {
        problems.addAll(unknown.problems());
      }
    }
    if (file != null) {
      layers.add(file);
    }
    if (!invocation.sets().isEmpty()) {
      try {
        layers.add(Layer.assignments("--set", invocation.sets()));
      } catch (SettingsException unreadable) {
        problems.addAll(unreadable.problems());
      }
    }
    for (Layer layer : layers) {
      problems.addAll(catalog.unknownSections(layer));
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    return new Layers(List.copyOf(layers), layers.stream().map(catalog::withoutSections).toList(), List.copyOf(presets));
  }

  private static void exit(int code) {
    System.out.flush();
    System.exit(code);
  }

  private static final class Invoked<R extends Record> implements Context<R> {
    private final Application<R> application;
    private final Reads reads;
    private final Layers layers;
    private final PrintStream out;
    private final Signals signals;
    private Settings<R> settings;

    Invoked(Application<R> application, Reads reads, Layers layers, PrintStream out, Signals signals) {
      this.application = application;
      this.reads = reads;
      this.layers = layers;
      this.out = out;
      this.signals = signals;
    }

    @Override
    public Application<R> application() {
      return application;
    }

    @Override
    public Settings<R> settings() {
      if (reads == Reads.RUN) {
        throw new IllegalStateException("A command that drives a run reads the run's settings");
      }
      if (settings == null) {
        settings = Settings.bind(application.schema(), application.stack(layers.settings()));
      }
      return settings;
    }

    @Override
    public List<Layer> layers() {
      return layers.settings();
    }

    @Override
    public List<String> presets() {
      return layers.presets();
    }

    @Override
    public PrintStream out() {
      return out;
    }

    @Override
    public Signals signals() {
      signals.claim();
      return signals;
    }
  }
}
