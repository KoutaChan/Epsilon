package com.epsilon.prism.grp;

import com.epsilon.prism.run.RunFiles;
import com.google.gson.Gson;
import java.io.IOException;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 裏の作業の心拍: 錠を持つあいだ、錠の種類ごとに grp/worker-&lt;score|refit&gt;.heartbeat へ pid・状態・時刻を {@value #PERIOD_SECONDS} 秒
 * ごとに書く（走らせ役の要約の行が読む）。一時ファイルからの rename で書くので読み手は書きかけを見ない。書き込みの失敗は 1 回だけ書き残し、
 * 作業は止めない。
 */
public final class WorkerHeartbeat implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(WorkerHeartbeat.class);

  /** 書く間隔（秒）。 */
  static final int PERIOD_SECONDS = 15;

  private static final Gson GSON = new Gson();

  private final GrpFiles files;
  private final WorkerLock.Kind kind;
  private final Instant started = Instant.now();
  private final ScheduledExecutorService timer;
  private volatile String state;
  private volatile boolean failed;

  private WorkerHeartbeat(GrpFiles files, WorkerLock.Kind kind, String state) {
    this.files = files;
    this.kind = kind;
    this.state = state;
    timer =
        Executors.newSingleThreadScheduledExecutor(
            task -> {
              Thread thread = new Thread(task, "grp-heartbeat");
              thread.setDaemon(true);
              return thread;
            });
  }

  /**
   * 心拍を始める（裏のスレッド）。
   *
   * @param kind 持っている錠の種類
   * @param state 始めの状態（例 scoring c00312、refit g312-a1 epoch 2/3）
   * @throws IOException 書けないとき
   */
  public static WorkerHeartbeat start(GrpFiles files, WorkerLock.Kind kind, String state) throws IOException {
    var heartbeat = new WorkerHeartbeat(files, kind, state);
    heartbeat.write(false);
    heartbeat.timer.scheduleWithFixedDelay(
        heartbeat::beat, PERIOD_SECONDS, PERIOD_SECONDS, TimeUnit.SECONDS);
    return heartbeat;
  }

  /** 状態を変える。 */
  public void state(String state) {
    this.state = state;
    beat();
  }

  /** 心拍を止め、終わった状態を書く。 */
  @Override
  public void close() {
    timer.shutdownNow();
    try {
      timer.awaitTermination(PERIOD_SECONDS, TimeUnit.SECONDS);
      write(true);
    } catch (IOException failure) {
      LOG.warn("the worker heartbeat was not closed: {}", failure.toString());
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private void beat() {
    try {
      write(false);
    } catch (IOException | RuntimeException failure) {
      if (!failed) LOG.warn("the worker heartbeat was not written: {}", failure.toString());
      failed = true;
    }
  }

  private synchronized void write(boolean done) throws IOException {
    Map<String, Object> beat = new LinkedHashMap<>();
    beat.put("pid", ProcessHandle.current().pid());
    beat.put("state", done ? "done" : state);
    beat.put("started", started.toString());
    beat.put("time", Instant.now().toString());
    RunFiles.writeAtomically(files.heartbeat(kind), GSON.toJson(beat) + "\n");
  }
}
