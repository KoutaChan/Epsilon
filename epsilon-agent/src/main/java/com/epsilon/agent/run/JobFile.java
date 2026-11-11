package com.epsilon.agent.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import java.time.Instant;
import java.util.Objects;

/**
 * ジョブの記録（jobs/&lt;名前&gt;/job.json）。定義と計画は木のまま持ち、run の種類が自分の record に結び付ける。
 *
 * <p>落ちた回数は、始めた回数から、セーフポイントできれいに止まった回数を引き、DONE か RUNNING なら最後の試行の 1 回も引いたもの。
 * DONE の最後の試行は結果を書いた試行で、RUNNING の最後の試行はまだ回っている（その JVM が落ちていれば、次に回すときに落ちた試行と
 * して記録する）。GPU を譲る・止める・起動し直すときは、きれいに止まった回数に入るので、落ちた回数は増えない。
 *
 * @param name ジョブの名前（{@link RunDirectory#jobName}）
 * @param version 測るバージョン
 * @param table 卓の名前
 * @param order 同じバージョンの中での順番（卓を宣言した順）
 * @param row ジョブを作った設定の行
 * @param definition ジョブの定義
 * @param plan 始めたときに決めた、変わらない計画。始める前は null
 * @param state 状態
 * @param attempts 始めた回数
 * @param cleanStops セーフポイントできれいに止まった回数
 * @param walls 打ち終えた牌山の数
 * @param gamesBytes games.tsv の、打ち終えた牌山までのバイト数
 * @param looksBytes looks.tsv の、済んだ判定の時点までのバイト数
 * @param created ジョブを作った時刻
 * @param reason 最後に落ちた理由（英語の 1 行）
 */
public record JobFile(
    String name,
    int version,
    String table,
    int order,
    int row,
    Node.Table definition,
    @Key(doc = "始めたときに決めた計画", presence = Presence.ABSENT) Node.Table plan,
    State state,
    int attempts,
    int cleanStops,
    long walls,
    long gamesBytes,
    long looksBytes,
    Instant created,
    @Key(doc = "最後に落ちた理由", presence = Presence.ABSENT) String reason) {

  /** ジョブの状態。 */
  public enum State {
    /** まだ始めていないか、止めた所からの続きを待っている。 */
    PENDING,
    /** 始めた（JVM が落ちた後もこのまま残り、次の起動で続ける）。 */
    RUNNING,
    /** 結果を書いた。 */
    DONE,
    /** 落ちた回数が上限に達した（{@code jobs retry} で戻すまで回さない）。 */
    FAILED
  }

  public JobFile {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(table, "table");
    Objects.requireNonNull(definition, "definition");
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(created, "created");
    if (version < 1 || order < 0 || row < 1) {
      throw new IllegalArgumentException(
          "Invalid job " + name + ": version " + version + ", order " + order + ", row " + row);
    }
    if (attempts < 0 || cleanStops < 0 || cleanStops > attempts || walls < 0 || gamesBytes < 0 || looksBytes < 0) {
      throw new IllegalArgumentException(
          "Invalid progress of job " + name + ": attempts " + attempts + ", clean stops " + cleanStops + ", walls "
              + walls);
    }
  }

  /** まだ始めていないジョブを作る。 */
  public static JobFile pending(int version, String table, int order, int row, Node.Table definition) {
    return new JobFile(
        RunDirectory.jobName(version, table), version, table, order, row, definition, null, State.PENDING, 0, 0, 0, 0,
        0, Instant.now(), null);
  }

  /** 落ちた回数。 */
  public int failures() {
    int last = state == State.DONE || state == State.RUNNING ? 1 : 0;
    return attempts - cleanStops - last;
  }

  /** まだ済んでおらず、これから回すジョブなら true（PENDING か RUNNING）。 */
  public boolean open() {
    return state == State.PENDING || state == State.RUNNING;
  }

  JobFile attempted() {
    return with(plan, State.RUNNING, attempts + 1, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  JobFile planned(Node.Table plan) {
    Objects.requireNonNull(plan, "plan");
    return with(plan, state, attempts, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  JobFile progressed(long walls, long gamesBytes, long looksBytes) {
    return with(plan, state, attempts, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  JobFile stoppedCleanly() {
    return with(plan, State.PENDING, attempts, cleanStops + 1, walls, gamesBytes, looksBytes, reason);
  }

  JobFile fell(String reason) {
    return with(plan, State.PENDING, attempts, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  JobFile failed() {
    return with(plan, State.FAILED, attempts, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  JobFile finished() {
    return with(plan, State.DONE, attempts, cleanStops, walls, gamesBytes, looksBytes, reason);
  }

  /**
   * 始めた回数ときれいに止まった回数を 0 に戻し（落ちた回数も 0 になる）、落ちた理由を消して、止めた所からの続きを待つ状態に
   * する。
   */
  JobFile retried() {
    return with(plan, State.PENDING, 0, 0, walls, gamesBytes, looksBytes, null);
  }

  private JobFile with(
      Node.Table plan,
      State state,
      int attempts,
      int cleanStops,
      long walls,
      long gamesBytes,
      long looksBytes,
      String reason) {
    return new JobFile(
        name, version, table, order, row, definition, plan, state, attempts, cleanStops, walls, gamesBytes, looksBytes,
        created, reason);
  }
}
