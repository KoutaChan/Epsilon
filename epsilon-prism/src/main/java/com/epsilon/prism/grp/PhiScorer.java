package com.epsilon.prism.grp;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpMetrics;
import com.epsilon.grp.train.GrpReport;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.grp.train.LateReport;
import com.epsilon.grp.train.NextHands;
import com.epsilon.grp.train.OutcomeCalibration;
import com.epsilon.grp.train.PhiStamp;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 裏で行う採点。半荘の記録があり、phimon.jsonl にまだ行が無いコホートを採点して、1 行書く（{@link PhiMonitorLine}）。値は
 * {@link HandEndPhi#openSingleSlot} で開いた Phi から出すので、ラベルに入った値とビット単位で同じになる。
 *
 * <p>採点する値は、記録の先頭にある GRP の項目に、数のファイルのヘッダーにあるモードと打ち切りの値を当てて作り直す。数のファイルが
 * 無いか読めなければ、mode.jsonl のうちコホートの反復に効く行からモードと打ち切りの値を取る。
 *
 * <p>GRP の項目は、記録の先頭の id で引き直す。{@value PhiEntry#FIXED_ID} なら設定の元の GRP、点数だけの Φ なら固定の項目、それ以外は
 * phi.jsonl の同じ id の行を使う。phi.jsonl の同じ id の行は同じ Φ を持つので、記録したときの項目と同じになる。
 *
 * <p>項目が q を持てば、同じ項目のもう一方のモードの値でも採点し、LATE が数え上げる局末で 2 つのモードを比べる（{@link LateReport}）。
 * さらに、数のファイルの局のうち LATE が数え上げる局で、q の較正を測る。GRP のモードで集めたコホートについて LATE のモードの値を作るときは、その反復までに
 * 効いた最後の LATE の行の打ち切りの値を使い、そういう行が無ければ設定の値を使う。
 *
 * <p>影の採点は、自分の GRP の run のときだけ行う。refits.jsonl の最後の状態が READY で、今使っていない候補のうち新しいほうから
 * {@value #SHADOW_CANDIDATES} つを、使ったモードの値で採点する。自分の GRP で WITH_ORIGINAL なら、元の GRP も影に加える。使ったモードに
 * ならない影（LATE のモードで q を持たない影）と、開けない影は飛ばす。
 *
 * <p>数え上げと q の較正は、採点専用の ForkJoinPool の中で並べて行い、共通のプールは使わない。自分の GRP ではプールのスレッド数を
 * grp.threads にする。
 */
public final class PhiScorer {
  private static final Logger LOG = LoggerFactory.getLogger(PhiScorer.class);

  /** 採点するコホートの遅れの上限。最新の記録からこの数を超えて古い未採点のコホートは採点しない。 */
  public static final int MAX_LAG = 8;

  /** 影として採点する READY の候補の数の上限。新しいものから取る。 */
  static final int SHADOW_CANDIDATES = 2;

  private PhiScorer() {}

  /**
   * 採点するコホートの記録と、採点する値。
   *
   * @param active 記録の先頭の GRP の項目
   * @param ends 記録の半荘の局末
   * @param value 採点する値。項目に、使ったモードと打ち切りの値を当てたもの
   * @param counted 数のファイルの中身。無いか読めなければ null
   */
  private record Recorded(int cohort, PhiEntry active, HandEnds ends, PhiValue value, CountedHands counted) {}

  /**
   * 使ったモードの値と、同じ項目のもう一方のモードの値で採点した結果。
   *
   * @param other もう一方のモードの値で採点した結果。測らなかったら null
   * @param late LATE が数え上げる局末で、2 つのモードを比べた報告。測らなかったら null
   */
  private record Modes(GrpReport active, GrpReport other, LateReport late) {}

  /** 採点するコホートを昇順に返す。記録があってまだ採点しておらず、最新の記録からの遅れが {@value #MAX_LAG} 以内のもの。 */
  public static List<Integer> pending(GrpFiles files) throws IOException {
    TreeSet<Integer> recorded = files.recordedCohorts();
    List<Integer> pending = new ArrayList<>();
    if (recorded.isEmpty()) return pending;
    Set<Integer> scored = new HashSet<>();
    for (PhiMonitorLine line : PhiMonitorLine.readAll(files)) scored.add(line.cohort());
    int newest = recorded.last();
    for (int cohort : recorded) if (newest - cohort <= MAX_LAG && !scored.contains(cohort)) pending.add(cohort);
    return pending;
  }

  /**
   * コホート t を採点して phimon.jsonl に書く。
   *
   * @return 書いた行。記録が読めないか、使ったモードの値を開けなければ空で、そのときはログに 1 回だけ書く
   * @throws IOException phimon.jsonl が書けないとき
   */
  public static Optional<PhiMonitorLine> score(GrpFiles files, PhiSetup setup, int cohort) throws IOException {
    long started = System.nanoTime();
    Optional<PhiMonitorLine> line;
    try (ForkJoinPool pool = new ForkJoinPool(threads(setup))) {
      line = pool.submit(() -> measure(files, setup, cohort, started)).get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedIOException("Scoring c" + cohort + " was interrupted");
    } catch (ExecutionException failed) {
      if (failed.getCause() instanceof Error error) throw error;
      throw new IOException("Scoring c" + cohort + " failed", failed.getCause());
    }
    if (line.isPresent()) line.get().append(files);
    return line;
  }

  /** コホート t の行を作る。採点のプールの中で呼ぶ。記録が読めないか、使ったモードの値を開けなければ空。 */
  private static Optional<PhiMonitorLine> measure(GrpFiles files, PhiSetup setup, int cohort, long started) {
    Recorded recorded;
    Modes modes;
    try {
      recorded = recorded(files, setup, cohort);
      modes = modes(files, setup, recorded);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} was not scored: {}", cohort, failure.toString());
      return Optional.empty();
    }
    OutcomeCalibration.Check calibration = calibration(recorded);
    PhiEntry active = recorded.active();
    PhiValue value = recorded.value();
    Map<String, GrpReport> shadows = new LinkedHashMap<>();
    List<PhiEntry> shadowEntries;
    try {
      shadowEntries = shadows(files, setup, active);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} has no shadow Φ: {}", cohort, failure.toString());
      shadowEntries = List.of();
    }
    for (PhiEntry shadow : shadowEntries) {
      PhiValue shadowValue = PhiValue.of(shadow, value.mode(), value.cutoff());
      if (shadowValue.mode() != value.mode()) {
        LOG.info(
            "c{} skipped the shadow Φ {}: it has no outcome network for the {} mode", cohort, shadow.id(), value.mode());
        continue;
      }
      try {
        shadows.put(shadow.id(), GrpMetrics.measure(recorded.ends(), values(shadowValue, recorded.ends(), setup)));
      } catch (IOException | RuntimeException failure) {
        LOG.warn("c{} skipped the shadow Φ {}: {}", cohort, shadow.id(), failure.toString());
      }
    }
    return Optional.of(
        new PhiMonitorLine(
            cohort,
            active.id(),
            value.mode(),
            value.cutoff(),
            modes.active(),
            modes.other(),
            modes.late(),
            calibration,
            shadows,
            (System.nanoTime() - started) / 1e9));
  }

  /**
   * コホートの記録を読み、採点する値を作り直す。数のファイルが読めなければログに書き、モードと打ち切りの値を mode.jsonl から取る。
   *
   * @throws IOException 記録が読めないか別のコホートのものであるとき、記録の先頭の項目が引けないとき
   */
  private static Recorded recorded(GrpFiles files, PhiSetup setup, int cohort) throws IOException {
    GameCohort games = GameRecordCodec.read(files.records(cohort));
    if (games.cohort() != cohort)
      throw new IOException("The game records of cohort " + cohort + " say " + games.cohort());
    PhiEntry active = active(files, setup, games);
    HandEnds ends = HandEnds.of(setup.variant(), setup.utility(), games.games());
    if (files.counted(cohort)) {
      try {
        CountedHands counted = CountFile.read(files.counts(cohort));
        if (counted.cohort() != cohort)
          throw new IOException("The count file of cohort " + cohort + " says " + counted.cohort());
        PhiStamp stamp = counted.phi();
        PhiValue value =
            PhiValue.of(active, PhiMode.valueOf(stamp.mode()), Double.isNaN(stamp.cutoff()) ? null : stamp.cutoff());
        return new Recorded(cohort, active, ends, value, counted);
      } catch (IOException | RuntimeException unreadable) {
        LOG.warn("c{} takes its Φ mode from {}: {}", cohort, files.modeHistory(), unreadable.toString());
      }
    }
    PhiValue value =
        ModeLine.effectiveAt(PhiHistory.readModes(files), cohort)
            .map(line -> PhiValue.of(active, line))
            .orElseGet(() -> PhiValue.grp(active));
    return new Recorded(cohort, active, ends, value, null);
  }

  /** 記録の先頭の id で項目を引き直す。 */
  private static PhiEntry active(GrpFiles files, PhiSetup setup, GameCohort recorded) throws IOException {
    return switch (recorded.phiId()) {
      case PhiEntry.FIXED_ID -> original(setup);
      case PhiEntry.SCORE_RANK_ID -> PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, setup.variant());
      case PhiEntry.SCORE_SMOOTH_ID -> PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, setup.variant());
      default -> {
        PhiEntry found = null;
        for (PhiEntry candidate : PhiHistory.read(files))
          if (candidate.id().equals(recorded.phiId())) found = candidate;
        if (found == null)
          throw new IOException("Φ entry " + recorded.phiId() + " is not in " + files.phiHistory());
        yield found;
      }
    };
  }

  /**
   * 使ったモードの値で採点する。項目が q を持てば、同じ項目のもう一方のモードの値でも採点し、LATE が数え上げる局末で 2 つのモードを
   * 比べる。もう一方のモードで失敗したときはログに書き、使ったモードの採点だけを返す。
   *
   * @throws IOException 使ったモードの値を開けないとき
   */
  private static Modes modes(GrpFiles files, PhiSetup setup, Recorded recorded) throws IOException {
    HandEnds ends = recorded.ends();
    PhiValue value = recorded.value();
    PhiEntry entry = recorded.active();
    HandEndValues used = values(value, ends, setup);
    GrpReport active = GrpMetrics.measure(ends, used);
    if (!entry.hasOutcomes()) return new Modes(active, null, null);
    boolean late = value.mode() == PhiMode.LATE;
    HandEndValues other;
    GrpReport otherReport;
    try {
      PhiValue otherValue =
          late ? PhiValue.grp(entry) : PhiValue.of(entry, PhiMode.LATE, lateCutoff(files, setup, recorded.cohort()));
      other = values(otherValue, ends, setup);
      otherReport = GrpMetrics.measure(ends, other);
    } catch (IOException | RuntimeException failure) {
      LOG.warn(
          "c{} has no {} mode report: {}", recorded.cohort(), late ? PhiMode.GRP : PhiMode.LATE, failure.toString());
      return new Modes(active, null, null);
    }
    try {
      return new Modes(
          active, otherReport, late ? LateReport.measure(ends, used, other) : LateReport.measure(ends, other, used));
    } catch (RuntimeException failure) {
      LOG.warn("c{} has no late report: {}", recorded.cohort(), failure.toString());
      return new Modes(active, otherReport, null);
    }
  }

  /**
   * 数のファイルの局のうち LATE が数え上げる局で、項目の q の較正を測る。項目が q を持たないか数のファイルが無ければ null。失敗したら
   * ログに書いて null。
   */
  private static OutcomeCalibration.Check calibration(Recorded recorded) {
    PhiEntry entry = recorded.active();
    if (!entry.hasOutcomes() || recorded.counted() == null) return null;
    HandEnds ends = recorded.ends();
    try {
      NextHands next = NextHands.of(ends, recorded.counted());
      return OutcomeCalibration.check(next, h -> FinalRound.covers(ends.state(h)), HandEndPhi.outcomes(entry));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} has no q calibration: {}", recorded.cohort(), failure.toString());
      return null;
    }
  }

  /** 局末ごとの Φ を値から求める。収集と同じ値の Phi を開く。局末ごとに 1 回ずつ求めるだけなので、終盤のメモは 1 スロットにする。 */
  private static HandEndValues values(PhiValue value, HandEnds ends, PhiSetup setup) throws IOException {
    return HandEndValues.of(ends, HandEndPhi.openSingleSlot(value, setup.utility()));
  }

  /**
   * GRP のモードで集めた反復 t について、LATE のモードの値に使う打ち切りの値を返す。t までに効いた最後の LATE の行の値を使い、そういう
   * 行が無ければ設定の値を使う。
   */
  private static double lateCutoff(GrpFiles files, PhiSetup setup, int t) throws IOException {
    List<ModeLine> late = new ArrayList<>();
    for (ModeLine line : PhiHistory.readModes(files)) if (line.mode() == PhiMode.LATE) late.add(line);
    return ModeLine.effectiveAt(late, t)
        .map(ModeLine::cutoff)
        .orElseGet(() -> setup.phi().lateSettings().orElseThrow().cutoff());
  }

  /** 影として採点する項目を返す。今使っている項目は除く。 */
  private static List<PhiEntry> shadows(GrpFiles files, PhiSetup setup, PhiEntry active) throws IOException {
    List<PhiEntry> shadows = new ArrayList<>();
    if (setup.selfPlay()) {
      // 同じ id では後の行が前の行に勝つ。id ごとに最後の行を、その行を書いた順に並べ、READY のものを新しいほうから取る
      Map<String, RefitRecord> latest = new LinkedHashMap<>();
      for (RefitRecord record : RefitRecord.readAll(files)) {
        latest.remove(record.id());
        latest.put(record.id(), record);
      }
      List<RefitRecord> ready = new ArrayList<>();
      for (RefitRecord record : latest.values())
        if (record.status() == RefitStatus.READY && !record.id().equals(active.id())) ready.add(record);
      for (RefitRecord record : ready.subList(Math.max(0, ready.size() - SHADOW_CANDIDATES), ready.size()))
        shadows.add(record.entry(setup.variant(), active.effectiveFrom(), "shadow"));
    }
    if (setup.phi() instanceof PhiSettings.SelfGrp self
        && self.grp().shadow() == PhiShadowReference.WITH_ORIGINAL
        && !active.id().equals(PhiEntry.FIXED_ID)) {
      PhiEntry original = original(setup);
      if (!original.sameValue(active)) shadows.add(original);
    }
    return shadows;
  }

  /** 採点の数え上げと q の較正を並べるスレッドの数。自分の GRP では grp.threads、ほかの出どころでは 1。 */
  private static int threads(PhiSetup setup) {
    return setup.phi() instanceof PhiSettings.SelfGrp self ? self.grp().threads() : 1;
  }

  /**
   * 元の GRP の項目を作る。固定の GRP ならその設定の GRP・PhiTable・q のディレクトリを、自分の GRP なら g000 の元の GRP の
   * ディレクトリ・PhiTable・q のディレクトリを使う。
   *
   * @throws IOException 元の GRP がないか読めないとき
   */
  public static PhiEntry original(PhiSetup setup) throws IOException {
    Path grp;
    Path table;
    Path outcomes;
    switch (setup.phi()) {
      case PhiSettings.FixedGrp fixed -> {
        grp = fixed.grp();
        table = fixed.table();
        outcomes = fixed.outcomes();
      }
      case PhiSettings.SelfGrp self when self.start() instanceof PhiStartSettings.GrpCheckpoint start -> {
        grp = start.grp();
        table = start.table();
        outcomes = start.outcomes();
      }
      default -> throw new IOException("No original GRP for the Φ source " + setup.phi().source());
    }
    return HandEndPhi.checkpointEntry(
        PhiEntry.FIXED_ID, setup.variant(), grp, table == null ? "" : table.toString(), outcomes, 1, "",
        "phi.source = FIXED_GRP");
  }
}
