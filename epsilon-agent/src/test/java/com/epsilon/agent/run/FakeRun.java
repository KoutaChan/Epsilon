package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.Settings;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;

/**
 * 段の進め方のテストに使う小さな run。段は PHI（jar に依らない）、COLLECT（段の中で 1 回待つ）、PUBLISH の 3 つ。
 * どの段の値も、反復の番号と state と設定だけで決まる。公開したバージョンごとに卓のジョブを作る。ジョブは牌山ごとに games.tsv へ
 * 1 行を書き、全ての行の値の和を結果にする。
 *
 * <p>落とす場所（{@link Point}）ごとに、1 回だけ落ちる仕掛けを置ける。落ち方は 2 つある。{@link Crash} は JVM が落ちたのと同じで、
 * どこでも捕捉されずに抜ける。もう 1 つは、ジョブや段の中で投げる普通の例外。
 */
final class FakeRun {

  /** テスト用の run の設定。 */
  record Config(
      @Key(doc = "反復の回数", value = "2", change = Change.FIT) @Range(min = 1) int iterations,
      @Key(doc = "収集の段で掛ける数", value = "3", change = Change.COLLECT) int factor,
      @Key(doc = "乱数の種", value = "7", change = Change.FIXED) long seed,
      @Key(doc = "推論のバッチサイズ", value = "64", change = Change.MACHINE) int batch,
      @Key(doc = "評価に使う卓", value = "[\"x1n3\"]", change = Change.SCHEDULE) List<String> tables,
      @Key(doc = "ジョブ 1 つの牌山の数", value = "5", change = Change.SCHEDULE) @Range(min = 1) int walls,
      @Key(doc = "チャンク 1 つの牌山の数", value = "2", change = Change.SCHEDULE) @Range(min = 1) int chunk) {}

  /** state.json の中身。 @param value 最後に公開した値 */
  record State(int next, long value) implements Progress {}

  /** 反復の報告に書いて、次の反復へ持ち越す値。 */
  record Carry(long value) {}

  /** 段の報告に書く値。 */
  record Metrics(long output) {}

  /** reports/fake.jsonl の 1 行。値を読むときに、{@link Point#REPORT}（反復の報告を書いた後、state を書く前）の仕掛けがあれば落とす。 */
  record Line(long value, int closed) {
    @Override
    public long value() {
      FakeRun world = CURRENT.get();
      if (world != null) {
        world.fault(Point.REPORT, closed);
      }
      return value;
    }
  }

  /** ジョブの定義。 */
  record JobDef(int version, String table, int walls, int chunk) {}

  /** ジョブの計画。 */
  record Plan(long seed, int walls, int chunk) {}

  /** ジョブの結果。 @param games games.tsv の行の数 */
  record Result(long games, long sum) {}

  /** 1 回の反復。 */
  record It(
      int number, Path work, int settingsRow, String jar, int launchRow, Config settings, State state, FakeRun world)
      implements Iteration {}

  /** 段の一覧。 */
  enum Steps implements StepKind<It> {
    PHI(false),
    COLLECT(true),
    PUBLISH(true);

    private final boolean jarBound;

    Steps(boolean jarBound) {
      this.jarBound = jarBound;
    }

    @Override
    public Step<It> create(It iteration) {
      return new FakeStep(this);
    }

    @Override
    public Class<? extends Record> metrics() {
      return Metrics.class;
    }

    @Override
    public boolean jarBound() {
      return jarBound;
    }
  }

  /** 落とす場所。 */
  enum Point {
    /** 段の出力を書いた後、完了マーカーへの rename の前。 */
    BEFORE_SEAL,
    /** 完了マーカーへの rename の後、段から戻る前。 */
    AFTER_SEAL,
    /** 段の中で投げる普通の例外。 */
    STEP_FAIL,
    /** 反復を閉じるときの値を作るところ（報告を書く前）。 */
    CLOSE,
    /** 反復の報告を書いた後、state を書く前。 */
    REPORT,
    /** state を書いた後、ジョブを作るところ。 */
    JOBS,
    /** 作業ディレクトリを消した後、古い出力を消すところ。 */
    CLEAN,
    /** ジョブのチャンクの行を書いた後、job.json を書く前に JVM が落ちる。番号は、チャンクを処理した通しの回数。 */
    CHUNK_CRASH,
    /** ジョブのチャンクの行を書いた後に投げる普通の例外。 */
    CHUNK_FAIL,
    /** 結果を書いた後、その結果を反映するところ。番号はバージョン。 */
    SETTLE
  }

  /** 1 つの仕掛け。番号は反復の番号で、ジョブではチャンクの回数かバージョン。段を問わない場所では段は null。 */
  record Fault(Point point, int at, Steps step) {}

  /** JVM が落ちたことを真似るエラー。どこでも捕捉されない。 */
  static final class Crash extends Error {
    Crash(String where) {
      super("crash at " + where, null, false, false);
    }
  }

  /** 段の始めとチャンクの後に差し込む処理。 */
  @FunctionalInterface
  interface Hook {
    void run() throws Exception;
  }

  /** 今の呼び出しで動いている run。{@link Line} が落とす場所を確かめるのに使う。 */
  static final ThreadLocal<FakeRun> CURRENT = new ThreadLocal<>();

  static final String JAR = "test0001";

  final RunDirectory run;
  final List<Fault> faults = new ArrayList<>();
  final Map<String, Integer> runs = new TreeMap<>();
  final Map<String, Hook> stepHooks = new LinkedHashMap<>();
  final Map<Integer, Hook> chunkHooks = new LinkedHashMap<>();

  /** 計画したジョブの名前と、テストが段の始めに差し込んだ印を、起きた順に並べたもの。 */
  final List<String> trail = new ArrayList<>();

  int opens;
  int plans;
  int chunks;
  int settles;
  Launch<Config> launch;
  Point crashed;

  FakeRun(Path root) {
    this.run = new RunDirectory(root);
  }

  /** 設定の 1 行目を書いた run を作る。設定は、既定値を --set の書き方で上書きしたもの。 */
  static FakeRun create(Path root, String... sets) throws IOException {
    FakeRun world = new FakeRun(root);
    try (SettingsRows.Locked locked = new SettingsRows(world.run).lock()) {
      locked.append(1, JAR, SettingsRow.CREATED, List.of(), tree(sets));
    }
    return world;
  }

  /** 設定の木。 */
  static Node.Table tree(String... sets) {
    List<Layer> layers = sets.length == 0 ? List.of() : List.of(Layer.assignments("--set", List.of(sets)));
    return Settings.bind(Config.class, layers).tree();
  }

  void fail(Point point, int at) {
    faults.add(new Fault(point, at, null));
  }

  void fail(Point point, int at, Steps step) {
    faults.add(new Fault(point, at, step));
  }

  /** 仕掛けた場所なら落とす。1 つの仕掛けで落ちるのは 1 回だけ。 */
  void fault(Point point, int at) {
    fault(point, at, null);
  }

  void fault(Point point, int at, Steps step) {
    for (Fault fault : faults) {
      if (fault.point() == point && fault.at() == at && fault.step() == step) {
        faults.remove(fault);
        String where = point + " " + at + (step == null ? "" : " " + step);
        if (point == Point.STEP_FAIL || point == Point.CHUNK_FAIL) {
          throw new IllegalStateException("boom at " + where);
        }
        crashed = point;
        throw new Crash(where);
      }
    }
  }

  /** 起動の設定。jar は {@link #JAR} で、GPU は譲らない。 */
  Runner.Options options(BooleanSupplier signalled) {
    return new Runner.Options(JAR, null, null, signalled);
  }

  Runner<Config, State, It, Steps> runner(Runner.Options options) {
    return new Runner<>(run, new Kind(this), options, Duration.ofMillis(5), Duration.ofMillis(40));
  }

  /** run を 1 回起動して、終わるか止まるまで反復を回す。 */
  Ending train(Runner.Options options) throws Exception {
    CURRENT.set(this);
    try {
      return runner(options).train();
    } finally {
      CURRENT.remove();
    }
  }

  Ending train() throws Exception {
    return train(options(() -> false));
  }

  /**
   * run が終わるまで起動し直す。仕掛けた場所で落ちたら（{@link Crash} か、仕掛けた普通の例外）、次の起動で続ける。仕掛けていない
   * 場所で落ちたら、その例外をそのまま投げる。20 回で終わらなければ {@link AssertionError} を投げる。
   *
   * @return 起動の回数
   */
  int finish(Runner.Options options) throws Exception {
    for (int launches = 1; launches <= 20; launches++) {
      int before = faults.size();
      try {
        Ending ending = train(options);
        if (ending == Ending.FINISHED) {
          return launches;
        }
      } catch (Crash | IllegalStateException crash) {
        if (faults.size() == before) {
          throw crash;
        }
      }
    }
    throw new AssertionError("The run did not finish in 20 launches");
  }

  int finish() throws Exception {
    return finish(options(() -> false));
  }

  int ran(int t, Steps step) {
    return runs.getOrDefault(t + ":" + step, 0);
  }

  /** run の結果。比べられるように、時刻・秒数・リソースと、段を実行した（RAN）か前の結果を使った（REUSED）かは除く。 */
  record Outcome(
      String state,
      List<String> iterations,
      List<String> extra,
      List<String> evaluations,
      Map<String, String> outputs,
      Map<String, JobFile.State> jobs) {}

  Outcome outcome() throws IOException {
    Map<String, Class<? extends Record>> metrics = new LinkedHashMap<>();
    for (Steps step : Steps.values()) {
      metrics.put(step.name(), Metrics.class);
    }
    List<String> iterations = new ArrayList<>();
    for (IterationReport report :
        IterationReport.read(run.report(RunDirectory.ITERATIONS), metrics, Carry.class)) {
      StringBuilder line =
          new StringBuilder(
              report.iteration() + " v" + report.version() + " row " + report.settingsRow() + " " + report.carry());
      report
          .steps()
          .forEach(
              (name, step) ->
                  line.append(' ')
                      .append(name)
                      .append('=')
                      .append(step.status() == StepReport.Status.SKIPPED ? "SKIPPED" : "DONE")
                      .append('/')
                      .append(step.launchRow())
                      .append('/')
                      .append(step.jar())
                      .append('/')
                      .append(step.metrics()));
      iterations.add(line.toString());
    }
    Map<String, String> outputs = new TreeMap<>();
    Map<String, JobFile.State> jobs = new TreeMap<>();
    for (JobFile job : new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS)).list().jobs()) {
      jobs.put(job.name(), job.state());
      for (String file : List.of(JobOutputs.GAMES, JobOutputs.LOOKS, Jobs.RESULT)) {
        Path path = run.job(job.name()).resolve(file);
        outputs.put(job.name() + "/" + file, Files.exists(path) ? Files.readString(path, StandardCharsets.UTF_8) : "");
      }
    }
    return new Outcome(
        Files.readString(run.state(), StandardCharsets.UTF_8),
        iterations,
        DurableFiles.lines(run.report("fake.jsonl")),
        DurableFiles.lines(run.report(RunDirectory.EVALUATIONS)),
        outputs,
        jobs);
  }

  /** 作業ディレクトリが空である（閉じた反復の残りが無い）ことを確かめる。 */
  void assertNoWork() throws IOException {
    if (Files.isDirectory(run.work())) {
      try (Stream<Path> listed = Files.list(run.work())) {
        assertEquals(listed.toList(), List.of());
      }
    }
  }

  /** 段の実装。完了マーカーは work/t/&lt;段&gt;/ のディレクトリで、中に報告を置く。 */
  private record FakeStep(Steps kind) implements Step<It> {

    @Override
    public Optional<StepReport> done(It iteration) throws IOException {
      Path done = iteration.work().resolve(kind.name());
      return Files.isDirectory(done)
          ? Optional.of(StepReport.read(done.resolve("report.json"), Metrics.class))
          : Optional.empty();
    }

    @Override
    public StepReport run(It iteration) throws Exception {
      long started = System.nanoTime();
      FakeRun world = iteration.world();
      int t = iteration.number();
      Path part = iteration.work().resolve(kind.name() + ".part");
      DurableFiles.delete(part);
      world.runs.merge(t + ":" + kind, 1, Integer::sum);
      Hook hook = world.stepHooks.get(t + ":" + kind);
      if (hook != null) {
        hook.run();
      }
      world.fault(Point.STEP_FAIL, t, kind);
      long output =
          switch (kind) {
            case PHI -> iteration.state().value() + t;
            case COLLECT -> {
              int[] calls = {0};
              world.launch.control().await(SafePoint.waiting(kind.name()), timeout -> ++calls[0] > 1);
              yield input(iteration, Steps.PHI) * iteration.settings().factor() + iteration.settings().seed();
            }
            case PUBLISH -> input(iteration, Steps.COLLECT) + 1;
          };
      StepReport report = StepReport.ran(iteration, started, new Metrics(output));
      report.write(part.resolve("report.json"));
      world.fault(Point.BEFORE_SEAL, t, kind);
      DurableFiles.seal(part, iteration.work().resolve(kind.name()));
      world.fault(Point.AFTER_SEAL, t, kind);
      return report;
    }

    private static long input(It iteration, Steps step) throws IOException {
      Path report = iteration.work().resolve(step.name()).resolve("report.json");
      return ((Metrics) StepReport.read(report, Metrics.class).metrics()).output();
    }
  }

  /** run の種類。 */
  private record Kind(FakeRun world) implements RunKind<Config, State, It, Steps> {

    @Override
    public Class<Config> settings() {
      return Config.class;
    }

    @Override
    public Class<State> state() {
      return State.class;
    }

    @Override
    public Class<Steps> steps() {
      return Steps.class;
    }

    @Override
    public Class<? extends Record> carry() {
      return Carry.class;
    }

    @Override
    public int iterations(Config settings) {
      return settings.iterations();
    }

    /** 卓は反復が公開したバージョンから測る（始めの state と一緒に置くバージョン 1 は測らない）。 */
    @Override
    public List<JobSpec> jobs(int version, Config settings) {
      world.fault(Point.JOBS, version - 1);
      List<JobSpec> specs = new ArrayList<>();
      if (version < 2) {
        return specs;
      }
      for (String table : settings.tables()) {
        specs.add(
            new JobSpec(table, RecordFile.tree(new JobDef(version, table, settings.walls(), settings.chunk()))));
      }
      return specs;
    }

    @Override
    public RunSession<Config, State, It> open(Launch<Config> launch) {
      world.opens++;
      world.launch = launch;
      return new Session(world, launch);
    }
  }

  /** テスト用の run のセッション。 */
  private record Session(FakeRun world, Launch<Config> launch) implements RunSession<Config, State, It> {

    @Override
    public State begin(Config settings) {
      return new State(1, settings.seed());
    }

    @Override
    public It iteration(IterationFile fixed, Config settings, State state) {
      return new It(
          fixed.iteration(),
          world.run.work(fixed.iteration()),
          fixed.settingsRow(),
          fixed.jar(),
          launch.row(),
          settings,
          state,
          world);
    }

    @Override
    public Closing<State> close(It iteration, State state, Map<String, StepReport> steps) {
      int t = iteration.number();
      world.fault(Point.CLOSE, t);
      long value = ((Metrics) steps.get(Steps.PUBLISH.name()).metrics()).output();
      return new Closing<>(new State(t + 1, value), new Carry(value), Map.of("fake.jsonl", new Line(value, t)));
    }

    @Override
    public JobRunner jobRunner() {
      return new FakeJobs(world);
    }

    @Override
    public void clean(State state) {
      world.fault(Point.CLEAN, state.next() - 1);
    }

    @Override
    public void close() {}
  }

  /** テスト用のジョブの中身。 */
  record FakeJobs(FakeRun world) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) {
      world.plans++;
      world.trail.add(job.name());
      JobDef definition = RecordFile.read(JobDef.class, job.definition()).value();
      long seed = definition.version() * 1000L + definition.table().length();
      return RecordFile.tree(new Plan(seed, definition.walls(), definition.chunk()));
    }

    @Override
    public JobWork open(JobFile job, Path directory) {
      Plan plan = RecordFile.read(Plan.class, job.plan()).value();
      return new JobWork() {
        @Override
        public Chunk next(long walls, JobOutputs outputs) throws Exception {
          int call = ++world.chunks;
          long end = Math.min(walls + plan.chunk(), plan.walls());
          for (long wall = walls; wall < end; wall++) {
            outputs.game(wall + "\t" + score(plan.seed(), wall));
          }
          outputs.look(walls + "\t" + end);
          world.fault(Point.CHUNK_CRASH, call);
          world.fault(Point.CHUNK_FAIL, call);
          Hook hook = world.chunkHooks.get(call);
          if (hook != null) {
            hook.run();
          }
          if (end < plan.walls()) {
            return Chunk.more(end);
          }
          outputs.sync();
          long games = 0;
          long sum = 0;
          for (String line : DurableFiles.lines(directory.resolve(JobOutputs.GAMES))) {
            games++;
            sum += Long.parseLong(line.substring(line.indexOf('\t') + 1));
          }
          return Chunk.decided(end, new Result(games, sum));
        }

        @Override
        public void close() {}
      };
    }

    @Override
    public void settle(JobFile job, Node.Table result) {
      world.fault(Point.SETTLE, job.version());
      world.settles++;
    }

    private static long score(long seed, long wall) {
      return (seed * 31 + wall * 17) % 101;
    }
  }

  /** status.json を読む。 */
  RunStatus status() throws IOException {
    return RunStatus.read(run.status());
  }

  /** jsonl の 1 行から、キーの値を取り出す。 */
  static Node field(String line, String key) {
    return Json.read(line, "test").get(key);
  }
}
