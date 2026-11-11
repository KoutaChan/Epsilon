package com.epsilon.agent.run;

import com.epsilon.agent.run.IterationReport.Resources;
import com.epsilon.agent.run.RunStatus.State;
import com.epsilon.agent.run.SettingsRow.ChangedKey;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.agent.settings.Toml;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 段を順に回して run を進める。
 *
 * <p>1 回の呼び出しでは、学習のロックを取って status.json を書き始め、起動のときの設定の行と state を読んでから、起動のセーフポイントを
 * 通る（リソースを開く前）。リソースを開いた後は、待っているジョブを全て回してから、反復をくり返す。1 回の反復では、設定の行と jar を
 * 固定し、段を回し、反復を閉じ、公開したバージョンのジョブを回す。止まるのは、セーフポイントで止まると決めたとき、全ての反復を終えた
 * とき、この呼び出しの上限に達したとき。新しい run では、始めの state と一緒に、バージョン 1 のジョブを起動の行の卓から作る。
 *
 * <p>反復を閉じる順は、報告の行（同じ反復の行が既にあれば書かない）、state.json、公開したバージョンのジョブ（何度作っても同じ）、作業
 * ディレクトリと古い出力の削除。state.json を書いた時点で、バージョンを公開したとみなす。閉じる途中で落ちても、次の起動が残りを済ませる。
 *
 * <p>固定した jar と違う jar では、jar に依る段が済んだ反復を続けない。設定の行・state・反復の固定が読めないか、この jar の設定に
 * 結び付かないときは、status.json に FAILED と理由を書いて REFUSED で終わる。段かジョブが例外で落ちたときは、FAILED と理由を書いて
 * 例外を投げ直す。
 *
 * @param <R> run 全体の設定
 * @param <S> state.json
 * @param <I> 反復
 * @param <K> 段の宣言
 */
public final class Runner<
    R extends Record, S extends Record & Progress, I extends Iteration, K extends Enum<K> & StepKind<I>> {

  private static final Logger LOG = LoggerFactory.getLogger(Runner.class);

  /**
   * 呼び出しの設定。
   *
   * @param jar JVM の jar の ID（{@link Jars#running}）
   * @param stopAfter この呼び出しで閉じる反復の数の上限。無ければ null
   * @param gpu GPU の要求。GPU の使用権を持たない JVM では null（GPU を譲らない）
   * @param signalled 停止のシグナル（SIGTERM・SIGINT）が来たか
   */
  public record Options(String jar, Integer stopAfter, GpuRequests gpu, BooleanSupplier signalled) {

    public Options {
      Objects.requireNonNull(jar, "jar");
      Objects.requireNonNull(signalled, "signalled");
      if (stopAfter != null && stopAfter < 1) {
        throw new IllegalArgumentException("stopAfter must be at least 1: " + stopAfter);
      }
    }

    /** 呼び出しの設定を作る。GPU の使用権を示す環境変数があるときだけ、~/.epsilon/gpu/requests の要求に GPU を譲る。 */
    public static Options of(String jar, Integer stopAfter, BooleanSupplier signalled) {
      return new Options(jar, stopAfter, GpuRequests.leaseHeld() ? GpuRequests.standard() : null, signalled);
    }
  }

  /** 反復に固定したもの。 @param settings 固定した行を結び付けた設定 */
  private record Fixed<T>(IterationFile file, SettingsRow row, T settings) {}

  private final RunDirectory run;
  private final RunKind<R, S, I, K> kind;
  private final Options options;
  private final Duration checkInterval;
  private final Duration heartbeat;
  private final SettingsRows rows;
  private final Jobs jobs;
  private List<String> addedState = List.of();

  public Runner(RunDirectory run, RunKind<R, S, I, K> kind, Options options) {
    this(run, kind, options, RunControl.CHECK_INTERVAL, StatusFile.HEARTBEAT);
  }

  /**
   * @param checkInterval 一時停止している間と段の中で待つ間に、セーフポイントの確かめをやり直す間隔
   * @param heartbeat ハートビートの間隔
   */
  Runner(RunDirectory run, RunKind<R, S, I, K> kind, Options options, Duration checkInterval, Duration heartbeat) {
    this.run = run;
    this.kind = kind;
    this.options = options;
    this.checkInterval = checkInterval;
    this.heartbeat = heartbeat;
    this.rows = new SettingsRows(run);
    this.jobs = new Jobs(run.jobs(), run.report(RunDirectory.EVALUATIONS));
  }

  /** 反復を回す（待っているジョブを先に回す）。 */
  public Ending train() throws Exception {
    return held(true);
  }

  /** 止めた run の、待っているジョブだけを回す。 */
  public Ending runJobs() throws Exception {
    return held(false);
  }

  /** @throws SettingsException run のディレクトリが無い（作らずに止まる） */
  private Ending held(boolean iterate) throws Exception {
    if (!Files.isDirectory(run.root())) {
      throw new SettingsException(List.of("The run " + run.root() + " does not exist: create it with settings create"));
    }
    try (TrainLock lock = TrainLock.hold(run.trainLock());
        StatusFile status = new StatusFile(run, options.jar(), heartbeat)) {
      try {
        return launch(status, iterate);
      } catch (Exception | Error failure) {
        status.state(State.FAILED, null, oneLine(failure));
        throw failure;
      }
    }
  }

  private Ending launch(StatusFile status, boolean iterate) throws Exception {
    RunControl control =
        new RunControl(
            new ControlRequests(run), options.gpu(), status, stepNames(), options.signalled(), checkInterval);
    SettingsRow launchRow;
    R launchSettings;
    S state;
    try {
      launchRow = rows.latest();
      launchSettings = bind(launchRow);
      status.launched(launchRow.row());
      state = readState();
    } catch (SettingsException refused) {
      return refuse(status, refused);
    }
    if (state != null) {
      status.iteration(state.next(), null);
    }
    Optional<Halt> halt = control.check(SafePoint.launch());
    if (halt.isPresent()) {
      return halt.get().ending();
    }
    Launch<R> launch = new Launch<>(run, control, launchRow.row(), launchSettings, options.jar());
    try (RunSession<R, S, I> session = kind.open(launch)) {
      if (state == null && iterate) {
        state = session.begin(launchSettings);
        if (state.next() != 1) {
          throw new IllegalStateException("A new run starts at iteration 1, not " + state.next());
        }
        // バージョン 1 のジョブは state.json より先に作る（落ちても、次の起動が同じジョブを作り直す）
        createJobs(1, launchRow.row(), launchSettings);
        DurableFiles.write(run.state(), state);
        status.iteration(state.next(), null);
      }
      try {
        if (state != null) {
          repair(session, state);
        }
      } catch (SettingsException refused) {
        return refuse(status, refused);
      }
      halt = jobs.runAll(session.jobRunner(), control);
      if (halt.isPresent()) {
        return halt.get().ending();
      }
      halt = control.check(SafePoint.jobs());
      if (halt.isPresent()) {
        return halt.get().ending();
      }
      if (!iterate) {
        status.state(State.FINISHED, null, null);
        return Ending.FINISHED;
      }
      try {
        state = readState();
      } catch (SettingsException refused) {
        return refuse(status, refused);
      }
      return iterate(session, control, status, state);
    }
  }

  private Ending iterate(RunSession<R, S, I> session, RunControl control, StatusFile status, S start)
      throws Exception {
    Pipeline<K, I> pipeline = new Pipeline<>(kind.steps(), control);
    S state = start;
    int closed = 0;
    while (true) {
      int t = state.next();
      Fixed<R> fixed;
      try {
        if (t > kind.iterations(bind(rows.latest()))) {
          status.state(State.FINISHED, null, null);
          LOG.info("Closed all {} iterations", t - 1);
          return Ending.FINISHED;
        }
        if (options.stopAfter() != null && closed >= options.stopAfter()) {
          String reason = "Closed " + closed + " iterations, the limit of this call";
          status.state(State.STOPPED, null, reason);
          LOG.info("{}", reason);
          return Ending.LIMIT;
        }
        fixed = fix(session, pipeline, state, t);
      } catch (SettingsException refused) {
        return refuse(status, refused);
      }
      status.iteration(t, fixed.row().row());
      I iteration = session.iteration(fixed.file(), fixed.settings(), state);
      Optional<Map<K, StepReport>> reports = pipeline.run(iteration);
      if (reports.isEmpty()) {
        return control.halt().orElseThrow().ending();
      }
      state = close(session, iteration, fixed, state, reports.get());
      closed++;
      status.iteration(state.next(), null);
      Optional<Halt> halt = control.check(SafePoint.iteration());
      if (halt.isPresent()) {
        return halt.get().ending();
      }
      halt = jobs.runAll(session.jobRunner(), control);
      if (halt.isPresent()) {
        return halt.get().ending();
      }
      try {
        state = readState();
      } catch (SettingsException refused) {
        return refuse(status, refused);
      }
      halt = control.check(SafePoint.jobs());
      if (halt.isPresent()) {
        return halt.get().ending();
      }
    }
  }

  /**
   * 反復に設定の行と jar を固定する（settings.lock を取って行う）。初めて固定するときは最新の行を選ぶ。jar が足したキーがあれば、その
   * 既定値を入れた行を 1 つ足してから固定する。固定済みの反復は、同じ行で続ける。固定した jar と違うときは、jar に依る段が済んでいない
   * 場合だけ、この JVM の jar で固定し直す。
   *
   * @throws SettingsException 行がこの jar の設定に結び付かないか、jar に依る段が済んだ反復を違う jar で続けようとした
   */
  private Fixed<R> fix(RunSession<R, S, I> session, Pipeline<K, I> pipeline, S state, int t) throws IOException {
    Path file = run.iterationFile(t);
    try (SettingsRows.Locked locked = rows.lock()) {
      if (Files.exists(file)) {
        IterationFile fixed = readFixed(file, t);
        SettingsRow row = locked.row(fixed.settingsRow());
        R settings = bind(row);
        if (fixed.jar().equals(options.jar())) {
          return new Fixed<>(fixed, row, settings);
        }
        if (pipeline.jarBoundDone(session.iteration(fixed, settings, state)).isPresent()) {
          throw new SettingsException(
              List.of("Iteration " + t + " started with jar " + fixed.jar() + "; restore that jar to finish it"));
        }
        IterationFile refixed = new IterationFile(t, fixed.settingsRow(), options.jar(), fixed.started());
        DurableFiles.write(file, refixed);
        LOG.info(
            "t={} continues with jar {} instead of {}: no step that depends on the jar is done",
            t,
            options.jar(),
            fixed.jar());
        return new Fixed<>(refixed, row, settings);
      }
      SettingsRow row = locked.latest();
      Settings<R> bound = bindSettings(row);
      List<ChangedKey> added = new ArrayList<>();
      addedKeys(bound.tree(), row.settings(), "", bound, added);
      if (!added.isEmpty()) {
        String keys = added.stream().map(ChangedKey::key).collect(Collectors.joining(", "));
        row = locked.append(t, options.jar(), "Jar " + options.jar() + " adds " + keys, added, bound.tree());
        LOG.info("Settings row {} adds the keys of jar {} with their defaults: {}", row.row(), options.jar(), keys);
      }
      IterationFile fixed = new IterationFile(t, row.row(), options.jar(), Instant.now());
      DurableFiles.write(file, fixed);
      return new Fixed<>(fixed, row, bound.value());
    }
  }

  private S close(RunSession<R, S, I> session, I iteration, Fixed<R> fixed, S state, Map<K, StepReport> reports)
      throws Exception {
    int t = iteration.number();
    Map<String, StepReport> steps = new LinkedHashMap<>();
    reports.forEach((step, report) -> steps.put(step.name(), report));
    Closing<S> closing = session.close(iteration, state, steps);
    if (closing.state().next() != t + 1) {
      throw new IllegalStateException(
          "Closing iteration " + t + " must move the state to iteration " + (t + 1) + ", not "
              + closing.state().next());
    }
    double seconds = steps.values().stream().mapToDouble(StepReport::seconds).sum();
    IterationReport report =
        new IterationReport(
            t,
            t + 1,
            fixed.row().row(),
            fixed.file().started(),
            seconds,
            steps,
            closing.carry(),
            Resources.measure(session.vramMiB()),
            addedState);
    Node number = new Node.Numeral(Integer.toString(t), "iteration");
    DurableFiles.appendAfterLast(run.report(RunDirectory.ITERATIONS), "iteration", number, report.tree());
    for (Map.Entry<String, Record> extra : closing.reports().entrySet()) {
      if (extra.getKey().equals(RunDirectory.ITERATIONS) || extra.getKey().equals(RunDirectory.EVALUATIONS)) {
        throw new IllegalStateException("A run kind must not add its own lines to " + extra.getKey());
      }
      DurableFiles.appendAfterLast(
          run.report(extra.getKey()), "iteration", number, numbered(number, extra.getValue()));
    }
    DurableFiles.write(run.state(), closing.state());
    addedState = List.of();
    createJobs(t + 1, fixed.row().row(), fixed.settings());
    DurableFiles.delete(run.work(t));
    session.clean(closing.state());
    String summary = session.summary(report);
    LOG.info(
        "t={} published version {} with settings row {} in {}s{}",
        t,
        t + 1,
        fixed.row().row(),
        String.format("%.1f", seconds),
        summary.isEmpty() ? "" : " " + summary);
    return closing.state();
  }

  /**
   * 閉じる途中で落ちた反復の残りを済ませる。閉じた反復の作業ディレクトリが残っていれば消す。直前の反復のものなら、消す前に公開した
   * バージョンのジョブを作る。古い出力を消し直す。job.json が読めないジョブは、最新の行の卓の定義から作り直す。
   */
  private void repair(RunSession<R, S, I> session, S state) throws Exception {
    int next = state.next();
    for (int leftover : closedWork(next)) {
      Path file = run.iterationFile(leftover);
      if (leftover == next - 1 && Files.exists(file)) {
        SettingsRow row = rows.row(readFixed(file, leftover).settingsRow());
        createJobs(next, row.row(), bind(row));
      }
      DurableFiles.delete(run.work(leftover));
      LOG.info("Removed the work of closed iteration {}", leftover);
    }
    session.clean(state);
    for (String name : jobs.list().unreadable()) {
      Optional<JobFile> fresh = recreated(name);
      if (fresh.isPresent()) {
        jobs.recreate(fresh.get());
        LOG.warn("Recreated job {} from the table of settings row {}: it starts over", name, fresh.get().row());
      } else {
        LOG.warn("Job {} has an unreadable job.json and no table to recreate it from, so it does not run", name);
      }
    }
  }

  private Optional<JobFile> recreated(String name) throws IOException {
    Optional<RunDirectory.JobName> parsed = RunDirectory.parseJobName(name);
    if (parsed.isEmpty()) {
      return Optional.empty();
    }
    int version = parsed.get().version();
    String table = parsed.get().table();
    SettingsRow latest = rows.latest();
    List<JobSpec> specs = kind.jobs(version, bind(latest));
    for (int order = 0; order < specs.size(); order++) {
      if (specs.get(order).table().equals(table)) {
        return Optional.of(JobFile.pending(version, table, order, latest.row(), specs.get(order).definition()));
      }
    }
    return Optional.empty();
  }

  private void createJobs(int version, int row, R settings) throws IOException {
    List<JobSpec> specs = kind.jobs(version, settings);
    int created = 0;
    for (int order = 0; order < specs.size(); order++) {
      JobSpec spec = specs.get(order);
      if (jobs.create(JobFile.pending(version, spec.table(), order, row, spec.definition()))) {
        created++;
      }
    }
    if (created > 0) {
      LOG.info("Created {} jobs for version {}", created, version);
    }
  }

  /** 次の反復より前で、作業ディレクトリが残っている反復の番号（小さい順）。 */
  private List<Integer> closedWork(int next) throws IOException {
    if (!Files.isDirectory(run.work())) {
      return List.of();
    }
    try (Stream<Path> listed = Files.list(run.work())) {
      return listed
          .map(path -> path.getFileName().toString())
          .filter(name -> name.matches("\\d+"))
          .map(Integer::parseInt)
          .filter(number -> number < next)
          .sorted()
          .toList();
    }
  }

  /**
   * state.json を読む。jar が足したフィールドには初期値が入る。そのフィールドは、次に閉じる反復が state.json に書き、その反復の報告にも
   * 並べる（閉じる前に止まれば、次の起動がもう一度見つける）。
   *
   * @return state.json が無ければ null
   * @throws SettingsException フィールドが欠けているか、知らないフィールドがあるか、値が型に合わない
   */
  private S readState() throws IOException {
    Path file = run.state();
    if (!Files.exists(file)) {
      return null;
    }
    RecordFile.Loaded<S> loaded = RecordFile.read(kind.state(), file);
    List<String> fresh = loaded.added().stream().filter(key -> !addedState.contains(key)).toList();
    if (!fresh.isEmpty()) {
      List<String> merged = new ArrayList<>(addedState);
      merged.addAll(fresh);
      addedState = List.copyOf(merged);
      LOG.info("state.json takes the fields {} that jar {} adds, with their first values", fresh, options.jar());
    }
    return loaded.value();
  }

  private static IterationFile readFixed(Path file, int t) throws IOException {
    IterationFile fixed = RecordFile.read(IterationFile.class, file).value();
    if (fixed.iteration() != t) {
      throw new SettingsException(List.of(file + " holds iteration " + fixed.iteration() + ", not " + t));
    }
    return fixed;
  }

  private R bind(SettingsRow row) {
    return bindSettings(row).value();
  }

  /** @throws SettingsException 行がこの jar の設定に結び付かない（メッセージに、行を書いた jar とこの jar を書く） */
  private Settings<R> bindSettings(SettingsRow row) {
    try {
      return row.bind(kind.settings());
    } catch (SettingsException unbound) {
      List<String> problems = new ArrayList<>();
      problems.add(
          "Settings row " + row.row() + " (written by jar " + row.jar() + ") does not bind with jar " + options.jar());
      problems.addAll(unbound.problems());
      throw new SettingsException(problems);
    }
  }

  /** 効いている設定の木にあって、行に書かれていない葉（jar が既定値で足したキー）を集める。 */
  private static void addedKeys(
      Node.Table effective, Node.Table written, String path, Settings<?> bound, List<ChangedKey> added) {
    for (Map.Entry<String, Node> entry : effective.entries().entrySet()) {
      String child = path.isEmpty() ? keyName(entry.getKey()) : path + "." + keyName(entry.getKey());
      Node before = written == null ? null : written.get(entry.getKey());
      if (entry.getValue() instanceof Node.Table table) {
        addedKeys(table, before instanceof Node.Table writtenTable ? writtenTable : null, child, bound, added);
      } else if (before == null) {
        added.add(new ChangedKey(child, null, entry.getValue(), bound.change(child)));
      }
    }
  }

  /** キーをキーのパスに書くときの形。設定のキーのパスと同じく、引用符の要らないキーはそのまま書き、ほかは TOML の文字列にする。 */
  private static String keyName(String key) {
    boolean bare = !key.isEmpty() && key.chars().allMatch(Runner::bareChar);
    return bare ? key : Toml.inline(new Node.Text(key, ""));
  }

  private static boolean bareChar(int c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
  }

  private static Node.Table numbered(Node number, Record line) {
    Map<String, Node> entries = new LinkedHashMap<>();
    entries.put("iteration", number);
    for (Map.Entry<String, Node> entry : RecordFile.tree(line).entries().entrySet()) {
      if (entries.containsKey(entry.getKey())) {
        throw new IllegalStateException(
            "The report line " + line.getClass().getName() + " has a component named " + entry.getKey());
      }
      entries.put(entry.getKey(), entry.getValue());
    }
    return new Node.Table(entries, "report");
  }

  private Set<String> stepNames() {
    return Arrays.stream(kind.steps().getEnumConstants()).map(Enum::name).collect(Collectors.toSet());
  }

  private Ending refuse(StatusFile status, SettingsException refused) throws IOException {
    List<String> problems = refused.problems();
    String reason = problems.getFirst() + (problems.size() > 1 ? " (and " + (problems.size() - 1) + " more)" : "");
    status.state(State.FAILED, null, reason);
    LOG.error("The run cannot continue:\n{}", String.join("\n", problems));
    return Ending.REFUSED;
  }

  private static String oneLine(Throwable failure) {
    String message = failure.getMessage();
    String text = failure.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    int newline = text.indexOf('\n');
    return newline < 0 ? text : text.substring(0, newline);
  }
}
