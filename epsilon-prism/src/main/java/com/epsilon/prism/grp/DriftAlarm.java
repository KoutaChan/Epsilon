package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.IOException;
import java.util.List;
import java.util.Optional;

/**
 * ずれの警報（phiDrift が OFF でないとき、採点の行ごとに進める）。使用中の項目の基準（{@link Baseline}）からの増分で決める: ON の線 = F_E +
 * max(phiDriftK·W, phiDriftMargin·F_E)、OFF の線 = F_E + (ON の線 − F_E)/2、W = √(SD_EWMA² + SD(F_E)²)。F_E は項目の D_SPG の床、SD_EWMA は
 * 1 コホートの SD(D_SPG) を EWMA（λ = {@link PhiMonitorLine#EWMA_WEIGHT}）で縮めた分。W は 1 コホート 6.5 万半荘では床の数 % しかなく、打ち方の
 * 小さな変化でも越えるので、割合の下限で意味のある悪化だけを拾う。D_SPG の EWMA が ON の線以上か、区分の傾きの基準からの |z| が
 * phiDriftSlopeZ を越えた状態が phiDriftConsecutive 回連続し、前の切り替えが grpSwitchSettle まで入っていれば ON（{@link
 * RefitScheduler#settled}）。D_SPG の EWMA が OFF の線以下で傾きも線の内に戻るまで ON。使用中の項目が変わったら EWMA と連続の数を初めから
 * やり直す。使用中が点数だけの Φ なら出さない。状態は grp/drift.jsonl に書く。
 *
 * <p>基準は、本番の Φ（元の GRP ＋ 較正の表）なら床 {@value #FLOOR}・傾き 1、自己対局で学んだ項目なら公開のときに取り置きのコホートで測った
 * その候補の D_SPG・区分の傾きと SE（学んだ GRP の床は本番より大きいことがあり、大域の床で比べると切り替えた瞬間から ON になるため）。基準の
 * 見つからない候補は警報を出さない。EWMA の分散は 1 コホートの SD² を重みの二乗で足すので、切り替えの直後ほど線が広い。
 */
public final class DriftAlarm {

  /** 床 F（本番の Φ の D_SPG を {@link CellStats#measure} で測った値）。 */
  static final double FLOOR = 4.535e-5;

  /** {@value #REFERENCE_GAMES} 半荘の 1 コホートの SD(D_SPG)（半荘のブートストラップ。半荘の数の平方根に反比例させる）。 */
  static final double COHORT_SD = 7.97e-6;

  /** COHORT_SD を測った半荘の数。 */
  static final double REFERENCE_GAMES = 65_536;

  private static final Gson GSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private DriftAlarm() {}

  /**
   * 警報の線（設定の phiDriftK・phiDriftMargin・phiDriftSlopeZ・phiDriftConsecutive）。
   *
   * @param k ON の線の統計の幅 W の倍数
   * @param margin ON の線の床からの割合の下限
   * @param slopeZ 越えたとみなす区分の傾きの基準からの |z|
   * @param consecutive ON にする連続の数
   */
  record Lines(double k, double margin, double slopeZ, int consecutive) {
    static Lines of(PrismConfig config) {
      return new Lines(
          config.phiDriftK(), config.phiDriftMargin(), config.phiDriftSlopeZ(), config.phiDriftConsecutive());
    }
  }

  /**
   * 項目の基準（警報はここからの増分で決める）。
   *
   * @param floor D_SPG の床 F_E
   * @param floorSd その測りの SD（本番の床は 0）
   * @param slope 区分ごとの傾きの基準（null なら β = 1・SE 0）
   * @param slopeSe その SE
   */
  record Baseline(double floor, double floorSd, double[] slope, double[] slopeSe) {

    /** 本番の Φ（元の GRP ＋ 較正の表）: 床 {@link #FLOOR}、傾き 1。 */
    static final Baseline PRODUCTION = new Baseline(FLOOR, 0, null, null);

    /** 自己対局で学んだ候補: 公開のときに取り置きのコホートで測った候補の較正。 */
    static Baseline of(CellReport candidate) {
      return new Baseline(candidate.dSpg(), cohortSd(candidate.games()), candidate.slope(), candidate.slopeSe());
    }

    /** 区分 group の傾きの基準からの z（測れなければ NaN）。 */
    double slopeZ(CellReport report, int group) {
      if (slope == null) return report.slopeZ()[group];
      return (report.slope()[group] - slope[group]) / Math.hypot(report.slopeSe()[group], slopeSe[group]);
    }
  }

  /**
   * 警報の状態（drift.jsonl の 1 行）。
   *
   * @param cohort コホート t
   * @param phiId 使用中の項目の id
   * @param on ON なら true
   * @param dSpgEwma D_SPG の EWMA
   * @param ewmaSd EWMA の SD
   * @param floor 項目の床 F_E
   * @param onLine ON の線
   * @param offLine OFF の線
   * @param maxSlopeZ 区分の傾きの基準からの |z| の最大
   * @param streak 線を越えた連続の数
   */
  public record State(
      int cohort,
      String phiId,
      boolean on,
      double dSpgEwma,
      double ewmaSd,
      double floor,
      double onLine,
      double offLine,
      double maxSlopeZ,
      int streak) {}

  /**
   * 採点の行 1 つで警報を進めて drift.jsonl に書く。既に進めたコホート以前の行は書かずに最後の状態を返す。
   *
   * @return 新しい状態
   * @throws IOException 読めないか書けないとき
   */
  public static State update(GrpFiles files, PrismConfig config, PhiMonitorLine line)
      throws IOException {
    Optional<State> previous = last(files);
    if (previous.isPresent() && line.cohort() <= previous.get().cohort()) return previous.get();
    Baseline baseline = baseline(config, files, line).orElse(null);
    boolean settled = baseline == null || RefitScheduler.settled(files, config, line.cohort());
    State next = next(previous.orElse(null), line, Lines.of(config), baseline, settled);
    PhiHistory.write(files.drift(), GSON.toJson(next));
    return next;
  }

  /** drift.jsonl の最後の状態が ON なら true（なければ false）。 */
  public static boolean isOn(GrpFiles files) throws IOException {
    return last(files).map(State::on).orElse(false);
  }

  /**
   * 警報を 1 コホート進める（ファイルに依らない本体）。
   *
   * @param previous 前の状態（なければ null）
   * @param lines 警報の線
   * @param baseline 使用中の項目の基準（null なら警報を出さない: 点数だけの Φ か、基準の見つからない候補）
   * @param settled 前の切り替えが grpSwitchSettle まで入っていれば true
   */
  static State next(State previous, PhiMonitorLine line, Lines lines, Baseline baseline, boolean settled) {
    double maxZ = maxSlopeZ(line.active(), baseline == null ? Baseline.PRODUCTION : baseline);
    if (baseline == null)
      return new State(
          line.cohort(), line.phiId(), false, Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN, maxZ, 0);
    double sd = cohortSd(line.active().games());
    double x = line.active().dSpg();
    boolean restart =
        previous == null
            || !previous.phiId().equals(line.phiId())
            || Double.isNaN(previous.ewmaSd());
    double lambda = PhiMonitorLine.EWMA_WEIGHT;
    double ewma;
    double variance;
    if (restart) {
      ewma = x;
      variance = sd * sd;
    } else {
      ewma = (1 - lambda) * previous.dSpgEwma() + lambda * x;
      variance =
          (1 - lambda) * (1 - lambda) * previous.ewmaSd() * previous.ewmaSd() + lambda * lambda * sd * sd;
    }
    double ewmaSd = Math.sqrt(variance);
    double width = Math.hypot(ewmaSd, baseline.floorSd());
    double excess = Math.max(lines.k() * width, lines.margin() * baseline.floor());
    double onLine = baseline.floor() + excess;
    double offLine = baseline.floor() + excess / 2;
    boolean slopeOut = maxZ > lines.slopeZ();
    boolean exceeded = ewma >= onLine || slopeOut;
    int streak = exceeded ? (restart ? 1 : previous.streak() + 1) : 0;
    boolean on =
        !restart && previous.on() ? ewma > offLine || slopeOut : streak >= lines.consecutive() && settled;
    return new State(
        line.cohort(), line.phiId(), on, ewma, ewmaSd, baseline.floor(), onLine, offLine, maxZ, streak);
  }

  /** 半荘の数 games の 1 コホートの SD(D_SPG)。 */
  static double cohortSd(long games) {
    if (games < 1) throw new IllegalArgumentException("No games in the scored cohort");
    return COHORT_SD * Math.sqrt(REFERENCE_GAMES / games);
  }

  /** 区分の傾きの基準からの |z| の最大（有限の区分がなければ 0）。 */
  private static double maxSlopeZ(CellReport report, Baseline baseline) {
    double max = 0;
    for (int group = 0; group < report.slopeZ().length; group++) {
      double z = baseline.slopeZ(report, group);
      if (Double.isFinite(z)) max = Math.max(max, Math.abs(z));
    }
    return max;
  }

  /**
   * 行のコホートを集めた項目の基準。FIXED_GRP と g000 の GRP は本番の床、SELF_GRP の自己対局の項目は refits.jsonl の同じ id の関門の
   * candidate（巻き戻しと境目の点検の代わりの行も id は同じ）。点数だけの Φ と、関門の測りのない候補は空。
   */
  private static Optional<Baseline> baseline(PrismConfig config, GrpFiles files, PhiMonitorLine line)
      throws IOException {
    if (config.phiSource() != PhiSource.SELF_GRP)
      return config.scoreOnlyPhi() ? Optional.empty() : Optional.of(Baseline.PRODUCTION);
    PhiEntryKind kind = config.phiStart().kind();
    for (PhiEntry entry : PhiHistory.read(files))
      if (entry.id().equals(line.phiId())) {
        kind = entry.kind();
        break;
      }
    if (kind.scoreBased()) return Optional.empty();
    if (kind != PhiEntryKind.SELF_CHECKPOINT) return Optional.of(Baseline.PRODUCTION);
    return RefitRecord.latest(RefitRecord.readAll(files), line.phiId())
        .map(RefitRecord::gate)
        .map(GateReport::candidate)
        .map(Baseline::of);
  }

  private static Optional<State> last(GrpFiles files) throws IOException {
    List<String> lines = PhiHistory.lines(files.drift());
    for (int i = lines.size() - 1; i >= 0; i--) {
      try {
        State state = GSON.fromJson(lines.get(i), State.class);
        if (state != null) return Optional.of(state);
      } catch (JsonParseException broken) {
        // 壊れた行は読み飛ばす
      }
    }
    return Optional.empty();
  }
}
