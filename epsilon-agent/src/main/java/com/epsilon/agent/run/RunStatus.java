package com.epsilon.agent.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * 段とジョブを回す JVM の状態（status.json）。JVM は一時ファイルに書いてから rename で置き換える。動いている間は、ハートビートで最後に
 * 書いた時刻を進める。
 *
 * @param state 状態
 * @param pid JVM のプロセス ID
 * @param pidStart そのプロセスの起動時刻（{@link ProcessStart}）。分からなければ無い
 * @param jar JVM の jar の ID
 * @param iteration 今の反復。state を読む前は無い
 * @param step 今の段。段を回していなければ無い
 * @param job 今のジョブ。ジョブを回していなければ無い
 * @param since 今の状態になった時刻
 * @param updated 最後に書いた時刻
 * @param settingsRow 今の反復に固定した設定の行。固定する前は無い
 * @param launchRow 起動のときに読んだ設定の行（次の起動から効くキーは、この行の値を使う）。読む前は無い
 * @param pendingRestart 起動の後に足した行で変わったキーのうち、起動し直さないと効かないもの
 * @param requests まだ従っていない要求の番号
 * @param obeyed 今の状態のもとになった要求の番号
 * @param reason 止まった理由か落ちた理由（英語の 1 行）
 */
public record RunStatus(
    State state,
    long pid,
    @Key(doc = "プロセスの起動時刻", presence = Presence.ABSENT) Long pidStart,
    String jar,
    @Key(doc = "今の反復", presence = Presence.ABSENT) Integer iteration,
    @Key(doc = "今の段", presence = Presence.ABSENT) String step,
    @Key(doc = "今のジョブ", presence = Presence.ABSENT) String job,
    Instant since,
    Instant updated,
    @Key(doc = "今の反復に固定した設定の行", presence = Presence.ABSENT) Integer settingsRow,
    @Key(doc = "起動のときに読んだ設定の行", presence = Presence.ABSENT) Integer launchRow,
    List<String> pendingRestart,
    List<Integer> requests,
    @Key(doc = "今の状態のもとになった要求の番号", presence = Presence.ABSENT) Integer obeyed,
    @Key(doc = "止まった理由か落ちた理由", presence = Presence.ABSENT) String reason) {

  /** 最後のハートビートがこれより古ければ、動いているはずの JVM はきれいに止まっていない。ハートビート 3 回ぶん。 */
  public static final Duration STALE = StatusFile.HEARTBEAT.multipliedBy(3);

  /** JVM の状態。 */
  public enum State {
    RUNNING,
    PAUSED,
    STOPPED,
    YIELDED,
    RESTARTING,
    FINISHED,
    FAILED;

    /** JVM が動いている間だけの状態か。ほかの状態は、JVM が終わる前に書く。 */
    public boolean live() {
      return this == RUNNING || this == PAUSED;
    }
  }

  public RunStatus {
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(jar, "jar");
    Objects.requireNonNull(since, "since");
    Objects.requireNonNull(updated, "updated");
    pendingRestart = List.copyOf(pendingRestart);
    requests = List.copyOf(requests);
  }

  public static RunStatus read(Path file) throws IOException {
    return RecordFile.read(RunStatus.class, file).value();
  }

  /**
   * 状態は動いているままなのに、最後のハートビートが {@link #STALE} より古いか、JVM のプロセスがもういない（pid が使い回されていれば、
   * いないとみなす）。自動では何もしない（status のコマンドが「きれいに止まっていない」と表示する）。
   *
   * @param now 今の時刻
   * @return きれいに止まっていなければ true
   */
  public boolean unclean(Instant now) {
    if (!state.live()) {
      return false;
    }
    boolean alive = pidStart == null ? ProcessHandle.of(pid).isPresent() : ProcessStart.alive(pid, pidStart);
    return !alive || updated.plus(STALE).isBefore(now);
  }
}
