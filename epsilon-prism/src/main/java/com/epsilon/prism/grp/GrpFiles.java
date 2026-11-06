package com.epsilon.prism.grp;

import com.epsilon.prism.rows.CohortWriter;
import java.nio.file.Path;

/**
 * run の GRP の自己対局のファイルの置き場（run/grp と run/reports/phimon.jsonl）。名前はここだけで決める。
 *
 * @param run run のディレクトリ
 */
public record GrpFiles(Path run) {

  /** run/grp。 */
  public Path directory() {
    return run.resolve("grp");
  }

  /** コホート t の主経路の対局の記録（{@link GameLog}）。 */
  public Path games(int t) {
    return directory().resolve("games").resolve(String.format("c%05d.bin.gz", t));
  }

  /** コホート t の行のディレクトリ（封印済みかは {@link CohortWriter#MANIFEST}）。 */
  public Path cohort(int t) {
    return CohortWriter.directory(run, t);
  }

  /** run の進み（state.json。次に集めるコホート）。 */
  public Path state() {
    return run.resolve("state.json");
  }

  /** Φ の切り替えの記録（追記だけ。{@link PhiHistory}）。 */
  public Path phiHistory() {
    return directory().resolve("phi.jsonl");
  }

  /** 学習の試行の記録（追記だけ。{@link RefitRecord}）。 */
  public Path refits() {
    return directory().resolve("refits.jsonl");
  }

  /** ずれの警報の記録（追記だけ。{@link DriftAlarm}）。 */
  public Path drift() {
    return directory().resolve("drift.jsonl");
  }

  /** 裏の採点の記録（追記だけ。{@link PhiMonitorLine}）。 */
  public Path monitor() {
    return run.resolve("reports").resolve("phimon.jsonl");
  }

  /** 公開した候補の GRP の置き場。 */
  public Path candidates() {
    return directory().resolve("candidates");
  }

  /** 試行 id（g&lt;k&gt;-a&lt;n&gt;）の候補の GRP のディレクトリ（書き終えてから rename で現れる）。 */
  public Path candidate(String id) {
    return candidates().resolve(id);
  }

  /** 書きかけの候補の置き場（candidates と同じファイルシステム。rename の元）。 */
  public Path staging() {
    return directory().resolve("staging");
  }

  /** 試行 id の候補の境目の点検の見本（{@link BoundaryCheck}）。 */
  public Path boundary(String id) {
    return directory().resolve("boundary").resolve(id + ".bin");
  }

  /** 手動の学習の要求（grp-refit が書き、学習の JVM の {@link GrpBackground} が読んで消す）。 */
  public Path refitRequest() {
    return directory().resolve("refit-request.json");
  }

  /** 記録のファイルの錠（{@link PhiHistory}。phi.jsonl・refits.jsonl の書き手と学習の JVM の印）。 */
  public Path lock(String name) {
    return directory().resolve(name + ".lock");
  }
}
