package com.epsilon.prism.grp;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.prism.rows.CohortWriter;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GRP の自己対局で run が使うファイルの場所（run/grp と run/reports/phimon.jsonl）。ファイル名はここだけで決める。
 *
 * @param run run のディレクトリ
 */
public record GrpFiles(Path run) {

  private static final Pattern RECORD = Pattern.compile("c(\\d{5})\\.bin");

  /** run/grp。 */
  public Path directory() {
    return run.resolve("grp");
  }

  /** コホート t の主経路の半荘の記録（GameRecordCodec）。 */
  public Path records(int t) {
    return directory().resolve("records").resolve(String.format("c%05d.bin", t));
  }

  /** コホート t の数のファイル（CountFile）。主経路の全ての局について、局の始まりの卓と結果を数えたもの。 */
  public Path counts(int t) {
    return directory().resolve("counts").resolve(String.format("c%05d.bin", t));
  }

  /** コホート t の行のディレクトリ。{@link CohortWriter#MANIFEST} があれば確定済み。 */
  public Path cohort(int t) {
    return CohortWriter.directory(run, t);
  }

  /** コホートの中に置く半荘の記録のファイル名。コホートが確定した後に {@link #records} へ移す。 */
  public static final String UNMOVED_RECORDS = "games.grp";

  /** コホートの中に置く数のファイルのファイル名。コホートが確定した後に {@link #counts} へ移す。 */
  public static final String UNMOVED_COUNTS = "counts.grp";

  /** 収集の段がコホート t の中に書く半荘の記録。コホートが確定した後に {@link #records} へ移す。 */
  public Path unmovedRecords(int t) {
    return cohort(t).resolve(UNMOVED_RECORDS);
  }

  /** 収集の段がコホート t の中に書く数のファイル。コホートが確定した後に {@link #counts} へ移す。 */
  public Path unmovedCounts(int t) {
    return cohort(t).resolve(UNMOVED_COUNTS);
  }

  /** コホート t が確定済みで、その中に半荘の記録も数のファイルも残っていなければ true。移し終えたときと、記録しなかったときに当たる。 */
  public boolean sealed(int t) {
    return Files.exists(cohort(t).resolve(CohortWriter.MANIFEST)) && moved(t);
  }

  /**
   * コホート t の半荘の記録があり、コホートの中に移していないファイルが残っていなければ true。記録は確定したコホートから移すので、
   * 行のコホートを消した後も true のまま。
   */
  public boolean recorded(int t) {
    return Files.exists(records(t)) && moved(t);
  }

  /**
   * コホート t の数のファイルがあり、コホートの中に移していないファイルが残っていなければ true。数のファイルは確定したコホートから
   * 移すので、行のコホートを消した後も true のまま。
   */
  public boolean counted(int t) {
    return Files.exists(counts(t)) && moved(t);
  }

  /** コホート t の中に、半荘の記録も数のファイルも残っていなければ true。 */
  private boolean moved(int t) {
    return !Files.exists(unmovedRecords(t)) && !Files.exists(unmovedCounts(t));
  }

  /** 半荘の記録（書き終えた grp/records/c&lt;t&gt;.bin）があるコホートを昇順で返す。 */
  public TreeSet<Integer> recordedCohorts() throws IOException {
    TreeSet<Integer> recorded = new TreeSet<>();
    Path directory = records(0).getParent();
    if (!Files.isDirectory(directory)) return recorded;
    try (DirectoryStream<Path> listed = Files.newDirectoryStream(directory, "c*.bin")) {
      for (Path file : listed) {
        Matcher matcher = RECORD.matcher(file.getFileName().toString());
        if (matcher.matches()) recorded.add(Integer.parseInt(matcher.group(1)));
      }
    }
    return recorded;
  }

  /** run の進み具合を書いた state.json。次に集めるコホートを持つ。 */
  public Path state() {
    return run.resolve("state.json");
  }

  /** Φ の段の完了マーカーのファイル名。反復の作業ディレクトリに置く。 */
  public static final String PHI_STEP = "PHI.json";

  /** 反復 t の Φ の段の完了マーカー。あれば、反復 t の Φ はもう決まっている。反復を閉じると、作業ディレクトリごと消える。 */
  public Path phiStep(int t) {
    return new RunDirectory(run).work(t).resolve(PHI_STEP);
  }

  /** Φ の切り替えの記録。追記だけする（{@link PhiHistory}）。 */
  public Path phiHistory() {
    return directory().resolve("phi.jsonl");
  }

  /** Φ のモードの切り替えの記録。追記だけする（{@link PhiHistory}）。 */
  public Path modeHistory() {
    return directory().resolve("mode.jsonl");
  }

  /** 学び直しの試行の記録。追記だけする（{@link RefitRecord}）。 */
  public Path refits() {
    return directory().resolve("refits.jsonl");
  }

  /** GRP を切り替えた後の監視の記録。追記だけする（{@link SwitchWatchRecord}）。 */
  public Path switchWatch() {
    return directory().resolve("switch-watch.jsonl");
  }

  /** 裏の採点の記録。追記だけする（{@link PhiMonitorLine}）。 */
  public Path monitor() {
    return run.resolve("reports").resolve("phimon.jsonl");
  }

  /** 試行 id（g&lt;k&gt;-a&lt;n&gt;）の候補の GRP のディレクトリ。書き終えてから rename で置くので、書きかけは見えない。 */
  public Path candidate(String id) {
    return directory().resolve("candidates").resolve(id);
  }

  /** 試行 id の候補の q（局の結果のネットワーク）のディレクトリ（&lt;id&gt;.q）。書き方は {@link #candidate} と同じ。 */
  public Path candidateOutcomes(String id) {
    return candidate(id + ".q");
  }

  /**
   * 記録のファイルのロック（{@link PhiHistory}）。phi.jsonl・mode.jsonl・refits.jsonl に書き込む側が取るほか、学習の JVM が動いている印にも
   * なる。
   */
  public Path lock(String name) {
    return directory().resolve(name + ".lock");
  }
}
