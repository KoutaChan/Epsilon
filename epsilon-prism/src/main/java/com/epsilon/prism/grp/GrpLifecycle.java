package com.epsilon.prism.grp;

import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GateReport;
import com.epsilon.grp.train.GrpGate;
import com.epsilon.grp.train.GrpMetrics;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.grp.train.MonotoneProbe;
import com.epsilon.grp.train.NextHands;
import com.epsilon.grp.train.OutcomeGate;
import com.epsilon.grp.train.SwitchWatchLine;
import com.epsilon.grp.train.SwitchWatchVerdict;
import com.epsilon.mahjong.rules.RankUtility;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * SELF_GRP の反復の区切りで行う処理。区切り h は、コホート h が確定した後、反復 h ＋ 1 の収集の前に来る。学び直す時点は番号だけで
 * 決まる（RefitSlot）。M(n) は n までのコホートで学び（{@link GrpRefitter}）、区切り n ＋ 1 で関門に通し、AUTO なら反復 n ＋ 2 の収集から
 * 使う。
 *
 * <p>区切り h では、まず監視の行を書く。行に書くのは、h を集めた項目と、その前に使った別の項目とを h で比べた、対応のある MSE の差と
 * その揺れ。重ならない 8 行がそろえば判定する。ROLLBACK なら、直近に使った 8 個の項目のうち h で最も良いものへ反復 h ＋ 1 から戻し、
 * τ を 2 倍にする。τ が 2 倍の間にまたアラートが出れば HALT にして、学び直しを止める。
 *
 * <p>次に、M(h − 1) を h の全ての半荘で関門に通す。使用中の項目が自己対局の候補でなければ、構造を入れ替えるときの関門（−2 SE より
 * 良いこと）を使う。それ以外は毎回の関門（+2 SE 以内）を使い、監視で戻した後の {@value GrpRefitter#FAST_ITERATIONS} 反復の間は、
 * 戻した項目とも比べる。どちらの関門も、単調性の違反率がしきい値以下であることを求める。READY にするかはこの関門だけで決め、
 * q（局の結果のネットワーク）の出来には依らない。
 *
 * <p>関門を通った候補の q は、h の数のファイルを使って q の関門（{@link OutcomeGate}）に通す。q が落ちても候補は REJECTED にせず、
 * 使用中の項目の q を引き継いで READY にする。候補ごと落とさないのは、落とすと q を使わない GRP のモードの間もネットワークの更新が
 * 止まってしまうから。
 *
 * <p>反復 h ＋ 1 の Φ がもう決まっているときは、監視の行と関門の結果は書くが、phi.jsonl には戻しの行も切り替えの行も足さない（決まった
 * Φ は書き換えない）。決まっているのは、閉じた反復、Φ の段の完了マーカーがある反復、呼び出し元が Φ の段を待たせていない反復。再起動の
 * 後に古い区切りをまとめて処理するとき、失敗した区切りを処理し直すとき、手動の学び直しが反復の間に終わらなかったときに起こる。
 */
final class GrpLifecycle {
  private static final Logger LOG = LoggerFactory.getLogger(GrpLifecycle.class);

  private final GrpFiles files;
  private final PhiSetup setup;
  private final RankUtility utility;

  GrpLifecycle(GrpFiles files, PhiSetup setup) {
    this.files = files;
    this.setup = setup;
    utility = setup.utility();
  }

  /** 最後に処理した区切りのコホート（無ければ 0）。 */
  int lastBoundary() throws IOException {
    return SwitchWatchRecord.state(SwitchWatchRecord.readAll(files)).lastCohort();
  }

  /** 監視が HALT にして、学び直しを止めているなら true。 */
  boolean halted() throws IOException {
    return SwitchWatchRecord.state(SwitchWatchRecord.readAll(files)).halted();
  }

  /** 確定したコホート h の区切りの処理を行う。呼び出し元が、反復 h ＋ 1 の Φ の段を待たせていること。 */
  void boundary(int h) throws IOException {
    boundary(h, true);
  }

  /**
   * 封印したコホート h の境目の手続き。
   *
   * @param held 呼び出し元が、この手続きの間、反復 h ＋ 1 の Φ の段を待たせているなら true。false なら、その反復の Φ は決まったものと
   *     して、phi.jsonl に足さない
   */
  void boundary(int h, boolean held) throws IOException {
    boolean late = !held || decided(h + 1);
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry used =
        PhiHistory.effectiveAt(history, h).orElseThrow(() -> new IOException("No Φ entry for cohort " + h));
    HandEnds ends = ends(h, used);
    Map<String, HandEndValues> values = new HashMap<>();
    SwitchWatchRecord watch = watch(h, history, used, ends, values, late);
    watch.append(files);
    if (watch.verdict() != null)
      LOG.info("c{} switch watch verdict {} (rollback to {}, widened {}, halted {})", h, watch.verdict(),
          watch.rollbackTo().isEmpty() ? "-" : watch.rollbackTo(), watch.widened(), watch.halted());
    if (watch.verdict() == SwitchWatchVerdict.HALT)
      LOG.error("c{} the GRP switch watch halted the refits: the procedure keeps losing to earlier models", h);
    gate(h, PhiHistory.read(files), used, ends, values, watch, late);
  }

  /**
   * 監視の行を書いた区切り h で、M(h − 1) の関門か AUTO の切り替えが済んでいなければ true。関門の途中で処理が止まった区切りが当たる。
   * 残りは {@link #lateGate(int, boolean)} が済ませる。
   */
  boolean gateLeft(int h) throws IOException {
    Optional<RefitRecord> found = RefitRecord.latestAt(RefitRecord.readAll(files), h - 1);
    if (found.isEmpty()) return false;
    RefitRecord record = found.get();
    if (record.status() == RefitStatus.TRAINED) return true;
    return record.status() == RefitStatus.READY
        && setup.selfGrp().grp().switching() == GrpSwitch.AUTO
        && !decided(h + 1)
        && PhiHistory.read(files).stream().noneMatch(entry -> entry.id().equals(record.id()));
  }

  /** 再学習の時点 slot の候補の関門。呼び出し元が、反復 slot ＋ 2 の Φ の段を待たせていること。 */
  void lateGate(int slot) throws IOException {
    lateGate(slot, true);
  }

  /**
   * 時点 {@code slot} で学び直した候補を、確定したコホート slot ＋ 1 で関門に通す。手動の学び直しで、区切りの処理の後に候補ができたときに
   * 使う。そのコホートが確定していないか、そのコホートで使う項目か監視の行が無ければ何もしない。
   *
   * @param held 呼び出し元が、この手続きの間、反復 slot ＋ 2 の Φ の段を待たせているなら true。false なら、その反復の Φ は決まった
   *     ものとして、phi.jsonl に足さない
   */
  void lateGate(int slot, boolean held) throws IOException {
    int h = slot + 1;
    if (!files.sealed(h) && !files.recorded(h)) return;
    boolean late = !held || decided(h + 1);
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry used = PhiHistory.effectiveAt(history, h).orElse(null);
    if (used == null) return;
    List<SwitchWatchRecord> records = SwitchWatchRecord.readAll(files);
    SwitchWatchRecord watch = null;
    for (SwitchWatchRecord record : records) if (record.cohort() == h) watch = record;
    if (watch == null) return;
    gate(h, history, used, ends(h, used), new HashMap<>(), watch, late);
  }

  /** 反復 t の Φ がもう決まっていれば true（閉じた反復か、Φ の段の完了マーカーがある反復）。 */
  private boolean decided(int t) throws IOException {
    return t < PhiHistory.nextIteration(files) || Files.exists(files.phiStep(t));
  }

  /** コホート h の局末を返す。半荘の記録が無いか、記録したときの項目が使用中の項目と違えば null。 */
  private HandEnds ends(int h, PhiEntry used) throws IOException {
    if (!files.recorded(h)) return null;
    GameCohort cohort = GameRecordCodec.read(files.records(h));
    if (!cohort.phiId().equals(used.id())) {
      LOG.warn("c{} was recorded with {}, not {}", h, cohort.phiId(), used.id());
      return null;
    }
    return HandEnds.of(setup.variant(), utility, cohort.games());
  }

  private SwitchWatchRecord watch(
      int h, List<PhiEntry> history, PhiEntry used, HandEnds ends, Map<String, HandEndValues> values, boolean late)
      throws IOException {
    SwitchWatchRecord.State state = SwitchWatchRecord.state(SwitchWatchRecord.readAll(files));
    PhiEntry previous = PhiHistory.previousAt(history, h).orElse(null);
    SwitchWatchLine line = null;
    if (ends != null && previous != null)
      line =
          SwitchWatchLine.of(
              h, used.id(), previous.id(), ends, values(used, ends, values), values(previous, ends, values));
    SwitchWatchVerdict verdict = null;
    boolean widened = state.widened();
    boolean halted = state.halted();
    String rollbackTo = "";
    if (line != null && !halted) {
      List<SwitchWatchLine> group = new ArrayList<>(state.pending());
      group.add(line);
      if (group.size() == SwitchWatchVerdict.LINES) {
        verdict = SwitchWatchVerdict.judge(group, widened);
        switch (verdict) {
          case QUIET -> widened = false;
          case ROLLBACK -> {
            widened = true;
            PhiEntry best = best(h, history, ends, values);
            rollbackTo = best.id();
            if (!best.sameValue(used) && !late)
              PhiHistory.appendUnstarted(
                  files,
                  best.from(
                      h + 1,
                      PhiEntry.ROLLBACK + " switch watch at c" + h + " chose " + best.id() + " over " + used.id()));
          }
          case HALT -> halted = true;
        }
      }
    }
    return new SwitchWatchRecord(
        h, used.id(), previous == null ? "" : previous.id(), line, verdict, widened, halted, rollbackTo);
  }

  /** 直近に使った {@value SwitchWatchVerdict#LINES} 個の別々の項目のうち、コホート h で MSE が最も小さいもの。測れない項目は飛ばす。 */
  private PhiEntry best(int h, List<PhiEntry> history, HandEnds ends, Map<String, HandEndValues> values)
      throws IOException {
    List<PhiEntry> recent = new ArrayList<>();
    for (int t = h; t >= 1 && recent.size() < SwitchWatchVerdict.LINES; t--) {
      PhiEntry entry = PhiHistory.effectiveAt(history, t).orElse(null);
      if (entry != null && recent.stream().noneMatch(entry::sameValue)) recent.add(entry);
    }
    PhiEntry best = recent.getFirst();
    double bestMse = Double.POSITIVE_INFINITY;
    for (PhiEntry entry : recent) {
      double mse;
      try {
        mse = GrpMetrics.mse(ends, values(entry, ends, values));
      } catch (IOException | RuntimeException unreadable) {
        LOG.warn("c{} could not measure {} for the rollback: {}", h, entry.id(), unreadable.toString());
        continue;
      }
      if (mse < bestMse) {
        bestMse = mse;
        best = entry;
      }
    }
    return best;
  }

  /** M(h − 1) を関門に通し、AUTO なら反復 h ＋ 1 からその候補に切り替える。 */
  private void gate(
      int h,
      List<PhiEntry> history,
      PhiEntry used,
      HandEnds ends,
      Map<String, HandEndValues> values,
      SwitchWatchRecord watch,
      boolean late)
      throws IOException {
    Optional<RefitRecord> found = RefitRecord.latestAt(RefitRecord.readAll(files), h - 1);
    if (found.isEmpty()) return;
    RefitRecord record = found.get();
    if (record.status() == RefitStatus.TRAINED) {
      long started = System.nanoTime();
      String refused =
          watch.halted() ? "the switch watch halted the refits"
              : !watch.rollbackTo().isEmpty() ? "the switch watch rolled back at c" + h
              : ends == null ? "cohort " + h + " has no usable game records"
              : null;
      if (refused != null) {
        record = record.with(RefitStatus.REJECTED, null, refused, 0);
      } else {
        GateReport report = judge(h, history, used, record, ends, values);
        String detail = report.kind() + " gate at c" + h + (report.passed() ? "" : ": " + report.failures());
        if (report.passed()) {
          OutcomeChoice choice = gateOutcomes(h, used, record, ends);
          record = record.withOutcomes(choice.directory(), choice.sha256(), choice.report());
          detail += "; " + choice.note();
        }
        record =
            record.with(
                report.passed() ? RefitStatus.READY : RefitStatus.REJECTED,
                report,
                detail,
                (System.nanoTime() - started) / 1e9);
      }
      record.append(files);
      LOG.info("GRP candidate {} {} at c{}: {}", record.id(), record.status(), h, record.detail());
    }
    String id = record.id();
    if (record.status() == RefitStatus.READY
        && setup.selfGrp().grp().switching() == GrpSwitch.AUTO
        && !late
        && watch.rollbackTo().isEmpty()
        && history.stream().noneMatch(entry -> entry.id().equals(id))) {
      PhiEntry entry = record.entry(setup.variant(), h + 1, "AUTO: " + id + " passed at c" + h);
      PhiHistory.appendUnstarted(files, entry);
      LOG.info("t={} AUTO switch to {}", h + 1, entry);
    }
  }

  /** 候補を関門で判定する。 */
  private GateReport judge(
      int h, List<PhiEntry> history, PhiEntry used, RefitRecord record, HandEnds ends, Map<String, HandEndValues> values)
      throws IOException {
    PhiEntry candidate = record.entry(setup.variant(), 1, "gate");
    Phi candidatePhi = HandEndPhi.open(candidate, utility);
    Phi usedPhi = HandEndPhi.open(used, utility);
    HandEndValues candidateValues = HandEndValues.of(ends, candidatePhi);
    HandEndValues usedValues = values(used, ends, values);
    MonotoneProbe probe = MonotoneProbe.of(ends);
    double usedMonotone = probe.violationRate(usedPhi);
    double candidateMonotone = probe.violationRate(candidatePhi);
    if (!used.candidate())
      return GrpGate.replacement(ends, usedValues, candidateValues, usedMonotone, candidateMonotone);
    PhiEntry anchor = anchor(history, h);
    if (anchor == null) return GrpGate.refit(ends, usedValues, candidateValues, usedMonotone, candidateMonotone);
    return GrpGate.refit(
        ends, usedValues, candidateValues, values(anchor, ends, values), usedMonotone, candidateMonotone);
  }

  /**
   * 関門を通った候補に残す q を決める。候補の q は、コホート h の数のファイルが覆う局で q の関門（{@link OutcomeGate}）に通す。使用中の
   * 項目が q を持てばその q と比べ、持たなければ一様な q と比べる。通れば候補の q を残し、通らなければ使用中の項目の q を引き継ぐ。
   * 候補が q を持たないとき、数のファイルが無いとき、q か数のファイルが読めないときも引き継ぐ。使用中の項目が q を持たなければ、
   * 引き継いだ候補も q を持たない。
   */
  private OutcomeChoice gateOutcomes(int h, PhiEntry used, RefitRecord record, HandEnds ends) {
    if (!record.hasOutcomes()) return OutcomeChoice.kept(used, null, "the refit learned no outcome network");
    if (!files.counted(h)) return OutcomeChoice.kept(used, null, "cohort " + h + " has no count file");
    OutcomeGate.Report report;
    try {
      NextHands next = NextHands.of(ends, CountFile.read(files.counts(h)));
      OutcomeNetwork candidate = HandEndPhi.outcomes(record.entry(setup.variant(), 1, "gate"));
      report =
          used.hasOutcomes()
              ? OutcomeGate.refit(next, HandEndPhi.outcomes(used), candidate)
              : OutcomeGate.first(next, candidate);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("c{} could not gate the outcome network of {}: {}", h, record.id(), failure.toString(), failure);
      return OutcomeChoice.kept(used, null, failure.toString());
    }
    String measured =
        String.format(
            "q %s gate at c%d: mean %+.6f SE %.6f over %d hand ends",
            report.kind(), h, report.versus().mean(), report.versus().standardError(), report.handEnds());
    return report.passed()
        ? new OutcomeChoice(record.outcomes(), record.outcomesSha256(), report, measured)
        : OutcomeChoice.kept(used, report, "failed " + measured);
  }

  /**
   * 関門を通った候補に残す q。
   *
   * @param directory q のチェックポイントのディレクトリの絶対パス（持たなければ空）
   * @param sha256 その q の sha（run の外から引き継いだ q のときだけ持ち、ほかは空）
   * @param report q の関門の判定（測らなかったときは null）
   * @param note refits.jsonl の理由に足す文
   */
  private record OutcomeChoice(String directory, String sha256, OutcomeGate.Report report, String note) {

    /** 使用中の項目の q を引き継ぐ。使用中の項目が q を持たなければ、q を持たない。 */
    static OutcomeChoice kept(PhiEntry used, OutcomeGate.Report report, String why) {
      return new OutcomeChoice(
          used.outcomes(),
          used.outcomesSha256(),
          report,
          "q kept from " + used.id() + (used.hasOutcomes() ? "" : " (none)") + ": " + why);
    }
  }

  /** 監視で戻した行が効き始めてから {@value GrpRefitter#FAST_ITERATIONS} 反復の間に h があれば、戻した先の項目を返す。それ以外は null。 */
  private static PhiEntry anchor(List<PhiEntry> history, int h) {
    PhiEntry anchor = null;
    for (PhiEntry entry : history)
      if (entry.rollback() && entry.effectiveFrom() <= h && h < entry.effectiveFrom() + GrpRefitter.FAST_ITERATIONS)
        anchor = entry;
    return anchor;
  }

  /** 項目で求めた局末の値。同じ区切りの中では、項目ごとに 1 度だけ求める。 */
  private HandEndValues values(PhiEntry entry, HandEnds ends, Map<String, HandEndValues> values) throws IOException {
    HandEndValues held = values.get(entry.id());
    if (held == null) {
      held = HandEndValues.of(ends, HandEndPhi.open(entry, utility));
      values.put(entry.id(), held);
    }
    return held;
  }
}
