package com.epsilon.prism;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Application;
import com.epsilon.agent.app.Command;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.app.DuelSigma;
import com.epsilon.prism.app.Evaluate;
import com.epsilon.prism.app.FitCohorts;
import com.epsilon.prism.app.GrpSigma;
import com.epsilon.prism.app.GrpTrust;
import com.epsilon.prism.app.JobsCreate;
import com.epsilon.prism.app.JobsList;
import com.epsilon.prism.app.JobsRetry;
import com.epsilon.prism.app.JobsRun;
import com.epsilon.prism.app.PlayRiichi;
import com.epsilon.prism.app.PlayTenhou;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.app.Records;
import com.epsilon.prism.app.RefitGrp;
import com.epsilon.prism.app.Report;
import com.epsilon.prism.app.SwitchGrp;
import com.epsilon.prism.app.Train;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * コマンドの一覧と起動のテスト。
 *
 * <ul>
 *   <li>コマンドは重ならず、ヘルプの順に並ぶ。ルール固有のコマンドは PRISM のコマンドの後に来る。
 *   <li>必須の引数だけを書くと、ほかは既定値になる。
 *   <li>help は 0 で終わり、知らないコマンドと読めない引数は 2 で終わる。
 *   <li>train と jobs run は run に保存した設定だけを読む。{@code --set}、存在しない run、範囲の外の引数は使い方のエラーとして拒み、
 *       存在しない run を作らない。
 * </ul>
 */
public class CommandsTest {

  private static final Path R = Path.of("/r");
  private static final Path O = Path.of("/o");

  private static final List<String> COMMANDS =
      List.of(
          "train", "jobs list", "jobs run", "jobs retry", "jobs create", "report", "evaluate",
          "duel-sigma", "fit-cohorts", "records", "grp-refit", "grp-switch", "grp-trust", "grp-sigma", "play-tenhou",
          "play-riichi");

  /** 最後に実行した {@link Echo}。 */
  private static final AtomicReference<Echo> ECHOED = new AtomicReference<>();

  @Test
  public void commandsAreDistinctAndListedInHelpOrder() {
    List<String> names = new ArrayList<>();
    for (Class<?> group : PrismApp.application().commands())
      for (Class<?> command : group.getPermittedSubclasses()) names.add(command.getAnnotation(Named.class).value());
    Assert.assertEquals(names, COMMANDS);
  }

  @Test
  public void requiredArgumentsAloneBindTheDefaults() {
    Assert.assertEquals(section(Train.class, "train", "run = \"/r\""), new Train(R, null));
    Assert.assertEquals(section(JobsList.class, "jobs.list", "run = \"/r\""), new JobsList(R));
    Assert.assertEquals(section(JobsRun.class, "jobs.run", "run = \"/r\""), new JobsRun(R));
    Assert.assertEquals(
        section(JobsRetry.class, "jobs.retry", "run = \"/r\"\nname = \"v00002-x1n3\""), new JobsRetry(R, "v00002-x1n3"));
    Assert.assertEquals(section(Report.class, "report", "run = \"/r\""), new Report(R, null, false, null, null, null));
    Assert.assertEquals(
        section(JobsCreate.class, "jobs.create", "run = \"/r\"\ntable = \"x1n3\"\nversions = \"2..4\""),
        new JobsCreate(R, "x1n3", "2..4"));
    Assert.assertEquals(
        section(Evaluate.class, "evaluate", "model = \"prism:/m\"\ntable = \"x1n3\"\nout = \"/o\""),
        new Evaluate("prism:/m", null, null, "x1n3", O, null));
    Assert.assertEquals(
        section(Evaluate.class, "evaluate", "snapshots = \"/r\"\nversions = \"2..4\"\ntable = \"x1n3\"\nout = \"/o\""),
        new Evaluate(null, R, "2..4", "x1n3", O, null));
    Assert.assertEquals(section(DuelSigma.class, "duel-sigma", "run = \"/r\""), new DuelSigma(R, null, 16, 25000));
    Assert.assertEquals(
        section(FitCohorts.class, "fit-cohorts", "run = \"/r\"\ncohorts = \"3..5\"\nout = \"/o\""),
        new FitCohorts(R, "3..5", O, FitCohorts.Mode.RAW, null, null, null, null, null, 0));
    Assert.assertEquals(
        section(Records.class, "records", "model = \"prism:/m\"\nout = \"/o\""),
        new Records("prism:/m", O, EvaluationTable.HEURISTIC, null, 1024, 0, null));
    Assert.assertEquals(section(RefitGrp.class, "grp-refit", "run = \"/r\""), new RefitGrp(R, null));
    Assert.assertEquals(
        section(SwitchGrp.class, "grp-switch", "run = \"/r\"\nto = \"g002-a1\""),
        new SwitchGrp(R, "g002-a1", null, null, "grp-switch"));
    Assert.assertEquals(section(GrpTrust.class, "grp-trust", "run = \"/r\""), new GrpTrust(R, null, 2000));
    Assert.assertEquals(section(GrpSigma.class, "grp-sigma", "run = \"/r\""), new GrpSigma(R, null));
    Assert.assertEquals(
        section(PlayTenhou.class, "play-tenhou", "model = \"prism:/m\"\nroom = \"1234\""),
        new PlayTenhou("prism:/m", "1234", "NoName", 1, Variant.TENHOU_4P, PolicyOptions.defaults()));
    Assert.assertEquals(
        section(PlayRiichi.class, "play-riichi", "model = \"/m\""),
        new PlayRiichi(Path.of("/m"), 0, PolicyOptions.defaults(), null));
    Assert.assertEquals(
        PolicyOptions.defaults(), new PolicyOptions("cpu", Precision.FLOAT32, 8, Kernels.EAGER), "the policy defaults");
  }

  /**
   * ルール固有のコマンド（{@code PrismVariant.commands} が返すもの）は PRISM のコマンドの後に並び、PRISM の入口から呼べる。全体の設定は
   * 読み込まずに、引数だけを読む。テストの中だけで使うコマンドを、{@link PrismApp} と同じく PRISM のコマンドの後に足して確かめる。
   */
  @Test
  public void theVariantCommandsRunFromThePrismEntryPoint() {
    Application<PrismSettings> prism = PrismApp.application();
    List<Class<? extends Command<? super PrismSettings>>> commands = new ArrayList<>(prism.commands());
    commands.add(EchoCommand.class);
    Application<PrismSettings> application =
        new Application<>(prism.name(), prism.schema(), commands, prism.base(), prism.check(), prism.runKind());
    List<String> names = new ArrayList<>();
    for (Class<?> group : application.commands())
      for (Class<?> command : group.getPermittedSubclasses()) names.add(command.getAnnotation(Named.class).value());
    Assert.assertEquals(names.getLast(), "echo");
    Assert.assertEquals(App.run(application, new String[] {"help", "echo"}), Exit.OK.code());
    Assert.assertEquals(App.run(application, new String[] {"echo"}), Exit.USAGE.code());
    Assert.assertEquals(App.run(application, new String[] {"echo", "--path", "/o", "--count", "0"}), Exit.USAGE.code());
    ECHOED.set(null);
    Assert.assertEquals(App.run(application, new String[] {"echo", "--path", "/o"}), Exit.OK.code());
    Assert.assertEquals(ECHOED.get(), new Echo(O, 1));
  }

  @Test
  public void helpSucceedsAndUnknownCommandsAndArgumentsAreUsageErrors() throws Exception {
    Assert.assertEquals(Main.run(new String[] {"help"}), Exit.OK.code());
    Assert.assertEquals(Main.run(new String[] {"help", "train"}), Exit.OK.code());
    Assert.assertEquals(Main.run(new String[] {"audit-grp"}), Exit.USAGE.code());
    Assert.assertEquals(Main.run(new String[] {"check-config"}), Exit.USAGE.code());
    Assert.assertEquals(Main.run(new String[] {"train", "--run"}), Exit.USAGE.code());
    Path root = Files.createTempDirectory("prism-commands");
    try {
      Path run = root.resolve("run");
      // 存在しない run は、作らずに拒む
      Assert.assertEquals(Main.run(new String[] {"train", "--run", run.toString()}), Exit.USAGE.code());
      Assert.assertEquals(Main.run(new String[] {"jobs", "run", "--run", run.toString()}), Exit.USAGE.code());
      Assert.assertFalse(Files.exists(run), "a refused train creates no run");
      TestSettings.create(run, "phi.source=SCORE_RANK");
      // run を動かすコマンドは、設定を上書きする層を受け付けない
      Assert.assertEquals(
          Main.run(new String[] {"train", "--run", run.toString(), "--set", "fit.maxSteps=1"}), Exit.USAGE.code());
      Assert.assertEquals(
          Main.run(new String[] {"jobs", "run", "--run", run.toString(), "--set", "fit.maxSteps=1"}),
          Exit.USAGE.code());
      Assert.assertEquals(
          Main.run(new String[] {"train", "--run", run.toString(), "--stop-after", "0"}), Exit.USAGE.code());
      Assert.assertFalse(Files.exists(run.resolve("state.json")), "a refused train writes nothing");
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** コマンドのセクションに {@code body} だけを書いた設定ファイルから、コマンドを作る。 */
  private static <T extends Record> T section(Class<T> type, String path, String body) {
    String text = "[" + path + "]\n" + body + "\n";
    return Settings.section(type, path, List.of(Layer.parse("prism.toml", Layer.Source.FILE, text))).value();
  }

  /** ルール固有のコマンドと同じ型を持つ、テストの中だけで使うコマンドのインターフェース。 */
  sealed interface EchoCommand extends Command<Record> permits Echo {}

  /**
   * 受け取った引数を残すだけのコマンド。
   *
   * @param path 必須の引数
   * @param count 既定値のある 1 以上の引数
   */
  @Named(value = "echo", doc = "Keep the bound arguments")
  record Echo(
      @Key(doc = "Required argument", presence = Presence.REQUIRED) Path path,
      @Key(doc = "Argument with a default", value = "1") @Range(min = 1) int count)
      implements EchoCommand {
    @Override
    public Outcome run(Context<Record> context) {
      ECHOED.set(this);
      return Outcome.ok("echoed " + path);
    }
  }
}
