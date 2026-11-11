package com.epsilon.prism.grp;

import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.train.RecencyWindow;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GRP を使う run を手で操作する。自分の GRP の run では、学び直しを要求し、Φ の項目を切り替えたり前の項目に戻したりする。Φ のモードは、
 * 固定の GRP の run でも切り替えられる。
 */
public final class PhiControl {

  private static final Logger LOG = LoggerFactory.getLogger(PhiControl.class);

  private PhiControl() {}

  /**
   * 設定の Φ が開けるかを確かめる。固定の GRP と、元の GRP から始める g000 は、チェックポイントの sha・表・ルールを照合しながら開いて
   * みる。q を持てば LATE のモードでも開き、q の sha・ルール・形と、葉が GrpNetwork であることを照合する。点数だけの Φ も開いてみる。
   * 点数のなめらかな関数は、σ の係数を持たないルール（三人打ち）では開けないので、問題として返る。
   *
   * @return 見つけた問題（1 件ごとに英語で 1 行）
   */
  public static List<String> problems(PhiSetup setup) {
    try {
      switch (setup.phi()) {
        case PhiSettings.FixedGrp fixed -> open(PhiScorer.original(setup), fixed.late(), setup);
        case PhiSettings.SelfGrp self -> open(PhiHistory.initialEntry(setup), self.late(), setup);
        case PhiSettings.ScoreRank rank ->
            HandEndPhi.open(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, setup.variant()), setup.utility());
        case PhiSettings.ScoreSmooth smooth ->
            HandEndPhi.open(PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, setup.variant()), setup.utility());
      }
      return List.of();
    } catch (IOException | RuntimeException failure) {
      return List.of("The Φ of the settings cannot be opened: " + failure);
    }
  }

  /** 項目を GRP のモードで開いてみる。q を持てば、設定の打ち切りの値で LATE のモードでも開いてみる（終盤のメモは 1 スロット）。 */
  private static void open(PhiEntry entry, LateSettings late, PhiSetup setup) throws IOException {
    HandEndPhi.open(entry, setup.utility());
    if (entry.hasOutcomes()) {
      PhiValue value = PhiValue.of(entry, ModeLine.late(late.cutoff(), 1, "check"));
      HandEndPhi.openSingleSlot(value, setup.utility());
    }
  }

  /**
   * 手動の学び直しの頼みを run の control に置く。学び直すのは、学習の JVM で GRP の裏の作業（{@link GrpBackground}）が次に動いたとき。
   * 関門はコホート n ＋ 1 で判定する。頼みは置いた数だけ番号の順に従い、学び終える前なら取り消しの頼みで消せる。
   *
   * @param run run のディレクトリ
   * @param operations 自分の GRP の運用の設定
   * @param slot 学び直す時点 n（null なら、半荘の記録がある最新の確定済みのコホート）
   * @return 置いた頼み
   * @throws IllegalArgumentException その時点のコホートが確定済みでないか半荘の記録が無いとき、または学び直しに使えるコホートが無いとき
   */
  public static ControlRequest requestRefit(Path run, GrpOperations operations, Integer slot) throws IOException {
    var files = new GrpFiles(run);
    int chosen = slot == null ? latestRecorded(files) : slot;
    if (!files.recorded(chosen))
      throw new IllegalArgumentException("Cohort " + chosen + " of " + run + " is not sealed with game records");
    var window = new RecencyWindow(chosen, operations.halfLife(), operations.window());
    if (window.cohorts() == 0) throw new IllegalArgumentException("No usable cohort up to " + chosen);
    var directory = new RunDirectory(run);
    ControlRequest written = new ControlRequests(directory).submit(new Request.GrpRefit(chosen));
    LOG.info(
        "grp-refit request {} for slot {} window {}..{} waits in {}",
        written.number(),
        chosen,
        window.first(),
        chosen,
        directory.control());
    return written;
  }

  /**
   * Φ を切り替えるか、前の項目に戻す。指定した反復から効く項目を phi.jsonl に 1 行足す。足す前に、候補なら refits.jsonl の最後の行が
   * READY であること、項目が sha を持つチェックポイントと q（run の外から来たもの）なら、sha が今も変わらないことを確かめる。項目の
   * チェックポイントと q は、読めることも確かめる。その反復で既に同じ Φ が効いていれば何も足さない。
   *
   * @param run run のディレクトリ
   * @param to 切り替える先。READY の候補の試行 id か、既にある項目の id（戻すとき）
   * @param from 効き始める反復（null なら、最後に始まった反復の次）
   * @param reason 書く理由
   * @return 足した項目。既に同じ Φ が効いていれば空
   * @throws IllegalStateException g000 がまだ無いとき、sha が変わったとき、チェックポイントか q が読めないとき
   * @throws IllegalArgumentException 候補でも項目でもないか、READY でないとき
   */
  public static Optional<PhiEntry> switchTo(Path run, String to, Integer from, String reason) throws IOException {
    var files = new GrpFiles(run);
    List<PhiEntry> entries = entries(files);
    int iteration = from == null ? PhiHistory.firstOpenIteration(files) : from;
    PhiEntry entry = target(files, entries, to, iteration, reason);
    Optional<PhiEntry> effective = PhiHistory.effectiveAt(entries, iteration);
    if (effective.isPresent() && effective.get().sameValue(entry)) {
      LOG.warn("grp-switch: {} is already the Φ at iteration {} ({}); nothing appended", to, iteration, effective.get().id());
      return Optional.empty();
    }
    PhiHistory.append(files, entry);
    LOG.info(
        "grp-switch: Φ {} from iteration {} (was {}): {}",
        entry.id(),
        iteration,
        effective.map(PhiEntry::id).orElse("-"),
        entry);
    return Optional.of(entry);
  }

  /**
   * Φ のモードを切り替える。指定した反復から効くモードの行を mode.jsonl に 1 行足し、GRP の項目は変えない。理由の先頭には
   * {@value ModeLine#SWITCH} を付ける。その反復で既に同じモードが効いていれば（LATE なら打ち切りの値も同じなら）、何も足さない。固定の GRP
   * の run では、設定で決めた最初の LATE の行を先に書く（{@link PhiHistory#initializeModes}）。
   *
   * <p>LATE の行は、その反復に効く GRP の項目の q で数え上げる。項目が q を持てば、足す前に LATE のモードで開けることを確かめる。
   * 確かめるのは、既に効いたことのある項目が LATE のモードで開けないと、系譜がその反復で止まってしまうから。q を持たなければ、警告を
   * 出して行を書く。q を持つ項目が効くまでは、局末を GRP のモードで求める。
   *
   * @param run run のディレクトリ
   * @param setup run の最新の設定（Φ の出どころは自分の GRP か固定の GRP）
   * @param mode 切り替える先のモード
   * @param cutoff LATE の打ち切りの値（null なら設定の値。GRP のモードでは null）
   * @param from 効き始める反復（null なら、最後に始まった反復の次）
   * @param reason 書く理由
   * @return 足した行。既に同じモードが効いていれば空
   * @throws IllegalStateException 自分の GRP の run に g000 がまだ無いとき、LATE のモードで開けないとき
   * @throws IllegalArgumentException Φ の出どころがモードを持たないとき、効き始めが始まった反復以前のとき
   */
  public static Optional<ModeLine> switchMode(
      Path run, PhiSetup setup, PhiMode mode, Double cutoff, Integer from, String reason) throws IOException {
    LateSettings late =
        setup.phi()
            .lateSettings()
            .orElseThrow(() -> new IllegalArgumentException("The Φ source " + setup.phi().source() + " has no Φ modes"));
    var files = new GrpFiles(run);
    List<PhiEntry> entries = setup.selfPlay() ? entries(files) : List.of();
    if (!setup.selfPlay()) PhiHistory.initializeModes(files, late);
    int iteration = from == null ? PhiHistory.firstOpenIteration(files) : from;
    Double lineCutoff = mode == PhiMode.LATE ? Objects.requireNonNullElse(cutoff, late.cutoff()) : cutoff;
    var line = new ModeLine(mode, lineCutoff, iteration, ModeLine.SWITCH + " " + reason);
    Optional<ModeLine> effective = ModeLine.effectiveAt(PhiHistory.readModes(files), iteration);
    PhiMode was = effective.map(ModeLine::mode).orElse(PhiMode.GRP);
    if (was == mode && Objects.equals(effective.map(ModeLine::cutoff).orElse(null), line.cutoff())) {
      LOG.warn("grp-switch: the {} mode is already in effect at iteration {}; nothing appended", mode, iteration);
      return Optional.empty();
    }
    if (mode == PhiMode.LATE) {
      PhiEntry entry =
          setup.selfPlay() ? PhiHistory.effectiveAt(entries, iteration).orElseThrow() : PhiScorer.original(setup);
      checkLate(entry, line, setup);
    }
    PhiHistory.appendMode(files, line);
    LOG.info("grp-switch: {} mode from iteration {} (was {}): {}", mode, iteration, was, line);
    return Optional.of(line);
  }

  /**
   * LATE の行が効き始める反復で効く GRP の項目を確かめる。項目が q を持てば、終盤のメモを 1 スロットにして LATE のモードで開いてみる。
   * q を持たなければ警告を出す。
   *
   * @throws IllegalStateException 項目が LATE のモードで開けないとき
   */
  private static void checkLate(PhiEntry entry, ModeLine line, PhiSetup setup) {
    if (!entry.hasOutcomes()) {
      LOG.warn(
          "grp-switch: Φ entry {} in effect at iteration {} has no outcome network, so the hand ends take the GRP mode"
              + " until an entry with one is in effect",
          entry.id(),
          line.effectiveFrom());
      return;
    }
    try {
      HandEndPhi.openSingleSlot(PhiValue.of(entry, line), setup.utility());
    } catch (IOException | RuntimeException failure) {
      throw new IllegalStateException(
          "Φ entry " + entry.id() + " in effect at iteration " + line.effectiveFrom()
              + " cannot be opened in the LATE mode",
          failure);
    }
  }

  /**
   * phi.jsonl の全ての行（書いた順）。
   *
   * @throws IllegalStateException g000 がまだ無いとき（g000 は学習の JVM が書く）
   */
  private static List<PhiEntry> entries(GrpFiles files) throws IOException {
    List<PhiEntry> entries = PhiHistory.read(files);
    if (entries.isEmpty())
      throw new IllegalStateException("No g000 in " + files.phiHistory() + ": start the SELF_GRP lineage first");
    return entries;
  }

  /** 切り替える先の項目を反復 {@code from} から効かせる行を作る。既にある項目なら戻しの行、無ければ READY の候補の行にする。 */
  private static PhiEntry target(GrpFiles files, List<PhiEntry> entries, String to, int from, String reason)
      throws IOException {
    Optional<RefitRecord> record = RefitRecord.latest(RefitRecord.readAll(files), to);
    if (record.isPresent() && record.get().status() != RefitStatus.READY)
      throw new IllegalArgumentException("Candidate " + to + " is " + record.get().status() + ": " + record.get().detail());
    PhiEntry existing = null;
    for (PhiEntry entry : entries) if (entry.id().equals(to)) existing = entry;
    PhiEntry entry;
    if (existing != null) {
      entry = existing.from(from, PhiEntry.ROLLBACK + " " + reason);
    } else {
      if (record.isEmpty()) throw new IllegalArgumentException("No Φ entry or refit attempt named " + to);
      entry = record.get().entry(entries.getFirst().variant(), from, reason);
    }
    if (!entry.sha256().isEmpty()) {
      String sha = GrpCheckpoint.sha256(Path.of(entry.path()));
      if (!sha.equals(entry.sha256()))
        throw new IllegalStateException("GRP checkpoint " + entry.path() + " has sha " + sha + ", not " + entry.sha256());
    }
    if (!entry.outcomesSha256().isEmpty()) {
      String sha = GrpCheckpoint.sha256(Path.of(entry.outcomes()));
      if (!sha.equals(entry.outcomesSha256()))
        throw new IllegalStateException(
            "Outcome network " + entry.outcomes() + " has sha " + sha + ", not " + entry.outcomesSha256());
    }
    checkReadable(entry);
    return entry;
  }

  /**
   * 項目のチェックポイントと q を読んでみる。run が書いた候補は sha を持たないので、ファイルが消えたり壊れたりしていても、読まなければ
   * 分からない。確かめるのは、既に効いたことのある項目が開けないと、系譜がその反復で止まってしまうから。
   *
   * @throws IllegalStateException チェックポイントか q が読めないとき
   */
  private static void checkReadable(PhiEntry entry) {
    try {
      if (!entry.kind().scoreBased()) GrpCheckpoint.read(Path.of(entry.path()));
      if (entry.hasOutcomes()) GrpCheckpoint.readOutcomes(Path.of(entry.outcomes()));
    } catch (IOException failure) {
      throw new IllegalStateException("Φ entry " + entry.id() + " cannot be opened", failure);
    }
  }

  /** 半荘の記録がある、最新の確定済みのコホート。 */
  private static int latestRecorded(GrpFiles files) throws IOException {
    TreeSet<Integer> recorded = files.recordedCohorts();
    for (int cohort : recorded.descendingSet()) if (files.recorded(cohort)) return cohort;
    throw new IllegalArgumentException("No sealed cohort with game records in " + files.run());
  }
}
