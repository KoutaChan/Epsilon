package com.epsilon.agent.run;

import com.epsilon.agent.run.RunStatus.State;
import com.epsilon.agent.settings.Change;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.OptionalLong;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * status.json のライター。状態・反復・段・ジョブが変わるたびに書き、JVM が動いている間は {@link #HEARTBEAT} ごとにハートビートとして
 * 書き直す。ハートビートのときと状態を書くときには、まだ従っていない要求と、起動し直すと効くキー（起動の後に足した行で変えたもの）を読み直す。
 * 閉じるとハートビートを止める（最後の状態はファイルに残る）。
 */
public final class StatusFile implements AutoCloseable {

  public static final Duration HEARTBEAT = Duration.ofSeconds(60);

  private static final Logger LOG = LoggerFactory.getLogger(StatusFile.class);

  private final RunDirectory run;
  private final ControlRequests requests;
  private final SettingsRows rows;
  private final long pid;
  private final Long pidStart;
  private final String jar;
  private final ScheduledExecutorService heartbeat;

  private State state = State.RUNNING;
  private Instant since = Instant.now();
  private Instant updated = since;
  private Integer iteration;
  private String step;
  private String job;
  private Integer settingsRow;
  private Integer launchRow;
  private List<String> pendingRestart = List.of();
  private List<Integer> pendingRequests = List.of();
  private Integer obeyed;
  private String reason;

  /**
   * 状態を RUNNING として書き、ハートビートを始める。
   *
   * @param jar JVM の jar の ID
   * @param interval ハートビートの間隔
   */
  StatusFile(RunDirectory run, String jar, Duration interval) throws IOException {
    this.run = run;
    this.requests = new ControlRequests(run);
    this.rows = new SettingsRows(run);
    this.pid = ProcessHandle.current().pid();
    OptionalLong start = ProcessStart.of(pid);
    this.pidStart = start.isPresent() ? start.getAsLong() : null;
    this.jar = jar;
    synchronized (this) {
      refresh();
      write();
    }
    heartbeat =
        Executors.newSingleThreadScheduledExecutor(
            task -> {
              Thread thread = new Thread(task, "status-heartbeat");
              thread.setDaemon(true);
              return thread;
            });
    long millis = interval.toMillis();
    heartbeat.scheduleAtFixedRate(this::beat, millis, millis, TimeUnit.MILLISECONDS);
  }

  /** {@link #HEARTBEAT} ごとにハートビートを書くライターを開く。 */
  public static StatusFile open(RunDirectory run, String jar) throws IOException {
    return new StatusFile(run, jar, HEARTBEAT);
  }

  /** 最後に書いた状態。 */
  public synchronized RunStatus current() {
    return snapshot();
  }

  /** 起動のときに読んだ設定の行を書く。 */
  synchronized void launched(int row) throws IOException {
    launchRow = row;
    refresh();
    write();
  }

  /**
   * 今の反復と、その反復に固定した設定の行を書く。
   *
   * @param settingsRow 固定した行。固定する前は null
   */
  synchronized void iteration(int iteration, Integer settingsRow) throws IOException {
    this.iteration = iteration;
    this.settingsRow = settingsRow;
    write();
  }

  /** @param step 今の段。段の外なら null */
  synchronized void step(String step) throws IOException {
    this.step = step;
    write();
  }

  /** @param job 今のジョブ。ジョブを回していなければ null */
  synchronized void job(String job) throws IOException {
    this.job = job;
    write();
  }

  /**
   * 状態を書く。状態が変わったときは、今の状態になった時刻を今の時刻にする。まだ従っていない要求と、起動し直すと効くキーも読み直す。
   *
   * @param obeyed 状態のもとになった要求の番号（無ければ null）
   * @param reason 理由（無ければ null）
   */
  synchronized void state(State state, Integer obeyed, String reason) throws IOException {
    if (state != this.state) {
      since = Instant.now();
    }
    this.state = state;
    this.obeyed = obeyed;
    this.reason = reason;
    refresh();
    write();
  }

  /** ハートビートを止める。書いている途中のハートビートは、書き終えるまで最大 10 秒待つ（最後の状態を書きかけで残さない）。 */
  @Override
  public void close() {
    heartbeat.shutdown();
    try {
      heartbeat.awaitTermination(10, TimeUnit.SECONDS);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private synchronized void beat() {
    try {
      refresh();
      write();
    } catch (IOException | RuntimeException failure) {
      LOG.warn("The heartbeat could not write {}: {}", run.status(), failure.toString());
    }
  }

  /** まだ従っていない要求と、起動し直すと効くキーを読み直す。読めなければ前の値のままにする。 */
  private void refresh() {
    try {
      pendingRequests = requests.pending().stream().map(ControlRequest::number).toList();
    } catch (IOException | RuntimeException unreadable) {
      LOG.warn("Cannot list the control requests of {}: {}", run.root(), unreadable.toString());
    }
    if (launchRow == null) {
      return;
    }
    try {
      TreeSet<String> keys = new TreeSet<>();
      for (SettingsRow row : rows.read()) {
        if (row.row() > launchRow) {
          row.changes().stream()
              .filter(changed -> changed.change() == Change.MACHINE)
              .forEach(changed -> keys.add(changed.key()));
        }
      }
      pendingRestart = List.copyOf(keys);
    } catch (IOException | RuntimeException unreadable) {
      LOG.warn("Cannot read the settings rows of {}: {}", run.root(), unreadable.toString());
    }
  }

  private void write() throws IOException {
    updated = Instant.now();
    DurableFiles.write(run.status(), snapshot());
  }

  private RunStatus snapshot() {
    return new RunStatus(
        state, pid, pidStart, jar, iteration, step, job, since, updated, settingsRow, launchRow, pendingRestart,
        pendingRequests, obeyed, reason);
  }
}
