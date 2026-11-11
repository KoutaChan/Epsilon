package com.epsilon.agent.run;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ジョブのディレクトリ（run の jobs、評価の出力の jobs）。ジョブは jobs/&lt;名前&gt;/ に job.json・games.tsv・looks.tsv・result.json
 * を持つ。result.json があれば済んだジョブ。
 *
 * <p>ジョブはバージョンの古い順（同じバージョンでは卓の順）に、1 つずつ最後まで回す。始めるたびに試行を 1 回数え、計画が無ければ
 * 決めて書く。チャンクを打った後がセーフポイントで、チャンクごとに打ち終えた牌山の数と書いた長さを job.json に書く。落ちた回数が
 * {@value #MAX_FAILURES} になったジョブは FAILED にして、ほかのジョブを続ける。
 *
 * <p>決着したら、result.json を書く、結果を反映する（{@link JobRunner#settle}）、結果の報告の行を足す（ジョブの名前で探して二度
 * 書かない）、job.json を DONE にする、の順に進める。どこで落ちても、次に回すときに残りを同じ順で済ませる。
 */
public final class Jobs {

  /** ジョブを FAILED にする、落ちた回数。 */
  public static final int MAX_FAILURES = 2;

  /** ジョブの記録。 */
  public static final String JOB = "job.json";

  /** ジョブの結果（このファイルがあれば済んだジョブ）。 */
  public static final String RESULT = "result.json";

  private static final Logger LOG = LoggerFactory.getLogger(Jobs.class);

  private static final String ORIGIN = "job";

  /** result.json と報告の行で、結果の record のフィールドより前に置くキー（ジョブの名前・バージョン・卓）。 */
  private static final String NAME_KEY = "job", VERSION_KEY = "version", TABLE_KEY = "table";

  private static final Comparator<JobFile> ORDER =
      Comparator.comparingInt(JobFile::version).thenComparingInt(JobFile::order).thenComparing(JobFile::name);

  private final Path directory;
  private final Path evaluations;

  /**
   * @param directory ジョブを置くディレクトリ
   * @param evaluations 済んだジョブの結果を足す jsonl
   */
  public Jobs(Path directory, Path evaluations) {
    this.directory = directory;
    this.evaluations = evaluations;
  }

  /**
   * ジョブの一覧。
   *
   * @param jobs 読めたジョブ（回す順）
   * @param unreadable job.json が読めないジョブの名前
   */
  public record Listing(List<JobFile> jobs, List<String> unreadable) {

    public Listing {
      jobs = List.copyOf(jobs);
      unreadable = List.copyOf(unreadable);
    }
  }

  /**
   * ジョブを作る。もうあれば何もしない（何度呼んでも結果は同じ）。
   *
   * @return 作ったら true
   */
  public boolean create(JobFile job) throws IOException {
    Path file = directory.resolve(job.name()).resolve(JOB);
    if (Files.exists(file)) {
      return false;
    }
    DurableFiles.write(file, job);
    return true;
  }

  /** ジョブの一覧を読む。job.json の無いディレクトリ（作る途中で落ちたジョブ）は数えない。 */
  public Listing list() throws IOException {
    if (!Files.isDirectory(directory)) {
      return new Listing(List.of(), List.of());
    }
    List<Path> homes;
    try (Stream<Path> listed = Files.list(directory)) {
      homes = listed.filter(Files::isDirectory).sorted().toList();
    }
    List<JobFile> jobs = new ArrayList<>(homes.size());
    List<String> unreadable = new ArrayList<>();
    for (Path home : homes) {
      String name = home.getFileName().toString();
      Path file = home.resolve(JOB);
      if (!Files.exists(file)) {
        continue;
      }
      try {
        JobFile job = RecordFile.read(JobFile.class, file).value();
        if (!job.name().equals(name)) {
          throw new SettingsException(List.of(file + " holds the job " + job.name()));
        }
        jobs.add(job);
      } catch (SettingsException broken) {
        LOG.warn("Cannot read {}: {}", file, broken.getMessage());
        unreadable.add(name);
      }
    }
    jobs.sort(ORDER);
    return new Listing(jobs, unreadable);
  }

  /** 名前を指定してジョブを読む。 */
  public JobFile read(String name) throws IOException {
    return RecordFile.read(JobFile.class, directory.resolve(name).resolve(JOB)).value();
  }

  /**
   * FAILED のジョブの落ちた回数を 0 に戻し、止めた所からの続きを待つ状態にする。
   *
   * @throws IllegalArgumentException ジョブが FAILED でないとき
   */
  public JobFile retry(String name) throws IOException {
    JobFile job = read(name);
    if (job.state() != JobFile.State.FAILED) {
      throw new IllegalArgumentException("Job " + name + " is " + job.state() + ", not FAILED");
    }
    return write(job.retried());
  }

  /** 読めないジョブを作り直す。ジョブのディレクトリを消して新しい記録を置き、初めから回す。 */
  public void recreate(JobFile fresh) throws IOException {
    DurableFiles.delete(directory.resolve(fresh.name()));
    create(fresh);
  }

  /**
   * 回すジョブを、順に最後まで回す。回している間に足されたジョブも回す。
   *
   * @return セーフポイントで止まると決めたときは、その内容。全て回し終えたら空
   */
  public Optional<Halt> runAll(JobRunner runner, RunControl control) throws Exception {
    while (true) {
      Optional<JobFile> next = list().jobs().stream().filter(JobFile::open).findFirst();
      if (next.isEmpty()) {
        return Optional.empty();
      }
      JobFile job = next.get();
      Optional<Halt> halt = control.check(SafePoint.beforeJob(job.name()));
      if (halt.isPresent()) {
        return halt;
      }
      control.job(job.name());
      try {
        halt = run(job, runner, control);
      } finally {
        control.job(null);
      }
      if (halt.isPresent()) {
        return halt;
      }
      halt = control.check(SafePoint.afterJob(job.name()));
      if (halt.isPresent()) {
        return halt;
      }
    }
  }

  /**
   * 1 つのジョブを、済むか、FAILED になるか、セーフポイントで止まるまで回す。落ちた試行の後は、すぐに次の試行を始める。ただし、ジョブを
   * 回す共有のリソースが使えなくなっていれば（{@link JobRunner#requireUsable}）、落ちた試行を記録した後、次の試行を始めずにその例外を
   * 投げる。チャンクの後のセーフポイントの確かめが失敗したら、その試行をきれいに止まった試行として記録し（落ちた回数は増えない）、
   * 次の試行を始めずにその例外を投げる。
   *
   * @return セーフポイントで止まると決めたときは、その内容。済んだか FAILED になったら空
   */
  public Optional<Halt> run(JobFile start, JobRunner runner, RunControl control) throws Exception {
    Path home = directory.resolve(start.name());
    JobFile job = start;
    if (job.state() == JobFile.State.RUNNING && !Files.exists(home.resolve(RESULT))) {
      // 前の JVM が試行の途中で終わった（その JVM は落ちた理由を書けなかった）
      job = write(job.fell("Attempt " + job.attempts() + " ended with its JVM"));
    }
    while (true) {
      if (control.halt().isPresent()) {
        return control.halt();
      }
      if (Files.exists(home.resolve(RESULT))) {
        finish(job, runner);
        return Optional.empty();
      }
      if (job.failures() >= MAX_FAILURES) {
        job = write(job.failed());
        LOG.warn("{} FAILED after {} attempts: {}", job.name(), job.attempts(), job.reason());
        return Optional.empty();
      }
      job = write(job.attempted());
      Attempt attempt;
      try {
        attempt = attempt(job, runner, control);
      } catch (InterruptedException interrupted) {
        throw interrupted;
      } catch (Exception failure) {
        LOG.warn("{} attempt {} failed", job.name(), job.attempts(), failure);
        job = write(read(job.name()).fell(oneLine(failure)));
        runner.requireUsable();
        continue;
      }
      job = attempt.job();
      if (attempt.checkFailure() != null) {
        throw attempt.checkFailure();
      }
      if (attempt.halt() != null) {
        return Optional.of(attempt.halt());
      }
    }
  }

  /**
   * 1 回の試行。
   *
   * @param halt セーフポイントで止まると決めた内容（決着したときと、確かめが失敗したときは null）
   * @param checkFailure セーフポイントの確かめの失敗（確かめられたら null）
   */
  private record Attempt(JobFile job, Halt halt, Exception checkFailure) {}

  private Attempt attempt(JobFile started, JobRunner runner, RunControl control) throws Exception {
    Path home = directory.resolve(started.name());
    JobFile job = started;
    if (job.plan() == null) {
      job = write(job.planned(runner.plan(job)));
    }
    try (JobOutputs outputs = JobOutputs.open(home, job.gamesBytes(), job.looksBytes());
        JobWork work = runner.open(job, home)) {
      while (true) {
        JobWork.Chunk chunk = work.next(job.walls(), outputs);
        if (chunk.walls() < job.walls()) {
          throw new IllegalStateException(
              "Job " + job.name() + " went back from " + job.walls() + " to " + chunk.walls() + " walls");
        }
        outputs.sync();
        job = write(job.progressed(chunk.walls(), outputs.gamesBytes(), outputs.looksBytes()));
        if (chunk.result() != null) {
          writeResult(job, chunk.result());
          return new Attempt(job, null, null);
        }
        Optional<Halt> halt;
        try {
          halt = control.check(SafePoint.chunk(job.name()));
        } catch (IOException | RuntimeException checkFailure) {
          // 確かめの失敗はジョブの失敗ではない。チャンクは書き終えているので、きれいに止まった試行として記録する
          return new Attempt(write(job.stoppedCleanly()), null, checkFailure);
        }
        if (halt.isPresent()) {
          return new Attempt(write(job.stoppedCleanly()), halt.get(), null);
        }
      }
    }
  }

  /**
   * 済んだジョブの結果を読む。ジョブの名前・バージョン・卓のキーを除いた残りを、結果の record に結び付ける。
   *
   * @param home ジョブのディレクトリ
   * @param type 結果の record（{@link JobWork.Chunk#decided} に渡したもの）
   * @throws SettingsException フィールドが record に合わないとき
   */
  public static <R extends Record> R result(Path home, Class<R> type) throws IOException {
    Path file = home.resolve(RESULT);
    return result(Json.read(Files.readString(file, StandardCharsets.UTF_8), file.toString()), type);
  }

  /**
   * 済んだジョブの結果の木（result.json か報告の行）から、ジョブの名前・バージョン・卓のキーを除き、結果の record に結び付ける。
   *
   * @param type 結果の record（{@link JobWork.Chunk#decided} に渡したもの）
   * @throws SettingsException フィールドが record に合わないとき
   */
  public static <R extends Record> R result(Node.Table line, Class<R> type) {
    return RecordFile.read(type, line.without(NAME_KEY).without(VERSION_KEY).without(TABLE_KEY)).value();
  }

  /** result.json を書く。ジョブの名前・バージョン・卓に続けて、結果の record のフィールドを並べる（報告の行と同じ形）。 */
  private void writeResult(JobFile job, Record result) throws IOException {
    Map<String, Node> entries = new LinkedHashMap<>();
    entries.put(NAME_KEY, new Node.Text(job.name(), ORIGIN));
    entries.put(VERSION_KEY, new Node.Numeral(Integer.toString(job.version()), ORIGIN));
    entries.put(TABLE_KEY, new Node.Text(job.table(), ORIGIN));
    for (Map.Entry<String, Node> entry : RecordFile.tree(result).entries().entrySet()) {
      if (entries.containsKey(entry.getKey())) {
        throw new IllegalStateException(
            "The job result " + result.getClass().getName() + " has a component named " + entry.getKey());
      }
      entries.put(entry.getKey(), entry.getValue());
    }
    String line = Json.write(new Node.Table(entries, ORIGIN)) + "\n";
    DurableFiles.write(directory.resolve(job.name()).resolve(RESULT), line);
  }

  /** 結果を反映し、報告の行を足して、ジョブを DONE にする。 */
  private void finish(JobFile job, JobRunner runner) throws Exception {
    Path file = directory.resolve(job.name()).resolve(RESULT);
    Node.Table result = Json.read(Files.readString(file, StandardCharsets.UTF_8), file.toString());
    runner.settle(job, result);
    DurableFiles.appendOnce(evaluations, NAME_KEY, new Node.Text(job.name(), ORIGIN), result);
    JobFile done = write(job.finished());
    LOG.info("{} DONE after {} walls in {} attempts", done.name(), done.walls(), done.attempts());
  }

  private JobFile write(JobFile job) throws IOException {
    DurableFiles.write(directory.resolve(job.name()).resolve(JOB), job);
    return job;
  }

  private static String oneLine(Throwable failure) {
    String message = failure.getMessage();
    String text = failure.getClass().getSimpleName() + (message == null ? "" : ": " + message);
    int newline = text.indexOf('\n');
    return newline < 0 ? text : text.substring(0, newline);
  }
}
