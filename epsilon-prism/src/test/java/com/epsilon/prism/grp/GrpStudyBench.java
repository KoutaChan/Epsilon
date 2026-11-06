package com.epsilon.prism.grp;

import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.prism.collect.PhiCalibration;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GRP の Φ のオフラインの調べ（CPU だけ）: audit-grp の CSV（{@code -Dprism.grp.study.csv=<audit.csv.gz>}）と較正の表
 * （{@code -Dprism.grp.study.calibration=<phi-calib.json>}）、元の GRP（{@code -Dprism.grp.study.grp=<GRP の正確なディレクトリ>}）で、
 * 手順 0 の組み直し、腕 A〜E の較正の測り、床と警報の線、学習の速さを測り、{@code -Dprism.grp.study.out=<dir>} に書く。名前が *Test でないので、
 * 既定の試験には入らず {@code -Dtest} で名指ししたときだけ走る。
 *
 * <p>腕: A 元の GRP、A+表 本番の表（全データで当てた表なので標本内）、B A ＋ 交差で当てた表、C 元の GRP から温めた自己対局の GRP（最大 6 epoch）、
 * D 白紙の自己対局の GRP（8k・16k・32k 半荘）、E 点数の順位、F 点数のなめらかな関数（{@link ScoreSmoothPhi}。σ の係数は同じ監査で
 * 当てた 1 個の定数なので標本内）。{@code -Dprism.grp.study.scoreOnly=true} なら学習する腕（C・D）を飛ばし、A・B・E・F だけ測る。分け方は半荘 ID の偶奇（grp_calib_util.py と同じ）。学習は片方の折りの最後の
 * {@value #VALIDATION} 半荘を検証（早めの打ち切り）に取り、残りで学ぶ。交差の腕（B・C・D32）は偶の折りで学んだものを奇の折りに、奇で学んだものを
 * 偶に当てて全ての半荘で測る。D8・D16 は偶の折りで学び奇の折りだけで測る。{@code -Dprism.grp.study.games=N} で先頭 N 半荘に絞る（煙試験）。
 * 結果の行は {@code PRISM-GRP-STUDY} で始まり、数値は全て study.json にも書く。
 */
public class GrpStudyBench {
  private static final Logger log = LoggerFactory.getLogger(GrpStudyBench.class);
  private static final String CSV_PROPERTY = "prism.grp.study.csv";
  private static final String CALIBRATION_PROPERTY = "prism.grp.study.calibration";
  private static final String GRP_PROPERTY = "prism.grp.study.grp";
  private static final String OUT_PROPERTY = "prism.grp.study.out";
  private static final String GAMES_PROPERTY = "prism.grp.study.games";
  private static final String SCORE_ONLY_PROPERTY = "prism.grp.study.scoreOnly";
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;
  private static final GrpGameTrainer.EpochListener NONE = GrpGameTrainer.EpochListener.NONE;

  /** 学習の折りの最後の検証の半荘の数。 */
  private static final int VALIDATION = 2048;

  /** 手順 0 の線（CSV は小数 6 桁）。 */
  private static final double STEP0_LINE = 1e-5;

  /** 交差の表の保留の二乗誤差の減り（監査の値）と許す幅。 */
  private static final double TABLE_GAIN = 0.000366, TABLE_GAIN_TOLERANCE = 0.00001;

  /** 監査（grp_calib_util.py の base、終わった局の段ごとの今 4 位の席の残差）。S4+ は S4 −0.0068（15,743）と W+ +0.0630（2,325）の重み付き。 */
  private static final double[] AUDIT_B4 = {
    0.0069, 0.0214, 0.0270, 0.0257, 0.0277, 0.0203, -0.0281, (15743 * -0.0068 + 2325 * 0.0630) / (15743 + 2325)
  };

  /** 警報の EWMA の重み、線の倍数、見る反復の数、模擬の回数。 */
  private static final double EWMA = 1.0 / 8, ON_K = 4, OFF_K = 2;

  private static final int ITERATIONS = 700, SIMULATIONS = 20_000, RESAMPLES = 200;

  /** grp_calib_util.py の表の形（段は終わった局の局番号、9 段目が西入）。 */
  private static final double TABLE_LO = -1, TABLE_HI = 2.0 / 3.0, TABLE_PRIOR = 50;

  private static final int TABLE_BINS = 24, TABLE_STAGES = 9;

  private final JsonObject results = new JsonObject();
  private Path out;

  @Test
  public void study() throws Exception {
    Path csv = required(CSV_PROPERTY);
    Path calibrationFile = required(CALIBRATION_PROPERTY);
    Path grp = required(GRP_PROPERTY);
    out = required(OUT_PROPERTY);
    Files.createDirectories(out);
    long started = System.nanoTime();
    AuditCsvGames audit = AuditCsvGames.read(csv);
    int limit = Integer.parseInt(System.getProperty(GAMES_PROPERTY, "0"));
    List<RecordedGame> games =
        limit > 0 ? audit.games().subList(0, Math.min(limit, audit.games().size())) : audit.games();
    List<RecordedGame> even = games.stream().filter(game -> game.gameId() % 2 == 0).toList();
    List<RecordedGame> odd = games.stream().filter(game -> game.gameId() % 2 != 0).toList();
    int handEnds = handEnds(games);
    line("data", "games %d hand ends %d (even %d odd %d) torch threads %s",
        games.size(), handEnds, even.size(), odd.size(), System.getProperty("ai.djl.pytorch.num_threads"));

    PhiValues a;
    try (CheckpointPhi original = CheckpointPhi.open(grp, new double[0], null, PROFILE)) {
      long t = System.nanoTime();
      a = original.gameValues(games);
      line("speed", "inference %.0f hand ends/s (maxBatch %d)", handEnds / seconds(t), original.maxBatch());
      step0(audit, a, handEnds);
    }
    PhiCalibration production = PhiCalibration.load(calibrationFile);
    PhiValues aTable = tabled(games, a, production);
    PhiValues b = merge(games, tabled(even, subset(games, a, true), fitTable(odd, subset(games, a, false), "odd")),
        tabled(odd, subset(games, a, false), fitTable(even, subset(games, a, true), "even")));
    PhiValues e = ScorePhi.offline(PhiEntryKind.SCORE_RANK, PROFILE).gameValues(games);

    CellReport reportA = report("A", games, a);
    report("A+table", games, aTable);
    CellReport reportB = report("B", games, b);
    report("E", games, e);
    scoreSmooth(games, a, b);
    CellStats.Paired gain = CellStats.paired(games, b, a, PROFILE);
    boolean reproduces = Math.abs(gain.mseDifference() - TABLE_GAIN) <= TABLE_GAIN_TOLERANCE;
    line("check", "B held-out squared-error reduction %+.6f ± %.6f (audit %.6f ± %.6f) -> %s",
        gain.mseDifference(), gain.standardError(), TABLE_GAIN, TABLE_GAIN_TOLERANCE, reproduces ? "OK" : "FAIL");
    results.addProperty("tableGain", gain.mseDifference());
    results.addProperty("tableGainSe", gain.standardError());
    tools(games, reportA, reportB, b);
    if (Boolean.getBoolean(SCORE_ONLY_PROPERTY)) {
      write(started);
      return;
    }

    // C: 元の GRP から温める（最大 6 epoch、学習曲線は偶の折りで学んだものを奇の折りで測る）
    List<RecordedGame> oddHoldout = odd;
    JsonArray curve = new JsonArray();
    GrpGameTrainer.EpochListener curveListener = (epoch, trainLoss, validationLoss, model) -> {
      PhiValues values = values(model, oddHoldout);
      CellReport r = CellStats.measure(oddHoldout, values, PROFILE);
      JsonObject point = new JsonObject();
      point.addProperty("epoch", epoch);
      point.addProperty("trainNll", trainLoss);
      point.addProperty("validationNll", validationLoss);
      point.addProperty("holdoutMse", r.mse());
      point.addProperty("holdoutDSpg", r.dSpg());
      point.addProperty("holdoutRankNll", r.rankNll());
      CellStats.Paired s3 = CellStats.residual(oddHoldout, values, PROFILE, GrpStudyBench::nearFourthS3);
      point.addProperty("holdoutS3NearFourth", s3.mseDifference());
      curve.add(point);
      line("curve", "C-even epoch %d train NLL %.5f validation NLL %.5f odd MSE %.6f D_SPG %.2e rank NLL %.5f"
              + " S3 near 4th %+.4f",
          epoch, trainLoss, validationLoss, r.mse(), r.dSpg(), r.rankNll(), s3.mseDifference());
    };
    Arm cEven = train("C-even", GrpInit.WARM, grp, even, Integer.MAX_VALUE, 2e-4, 6, 101, curveListener);
    results.add("curveC", curve);
    Arm cOdd = train("C-odd", GrpInit.WARM, grp, odd, Integer.MAX_VALUE, 2e-4, 6, 102, NONE);
    Pair c = pair("C", games, cEven, cOdd);

    // D: 白紙
    Arm dEven = train("D32-even", GrpInit.SCRATCH, null, even, Integer.MAX_VALUE, 1e-3, 40, 201, NONE);
    Arm dOdd = train("D32-odd", GrpInit.SCRATCH, null, odd, Integer.MAX_VALUE, 1e-3, 40, 202, NONE);
    Pair d32 = pair("D32", games, dEven, dOdd);
    JsonObject curveD = new JsonObject();
    PhiValues bOdd = subset(games, b, false);
    CellReport oddB = report("B@odd", odd, bOdd);
    curveD.addProperty("B", oddB.mse());
    for (int size : new int[] {8192, 16384}) {
      if (size >= even.size() - VALIDATION) continue;
      Arm arm = train("D" + size / 1024 + "-even", GrpInit.SCRATCH, null, even, size, 1e-3, 40, 200 + size, NONE);
      PhiValues values = phi(arm, odd);
      CellReport r = report("D" + size / 1024 + "@odd", odd, values);
      CellStats.Paired vsB = CellStats.paired(odd, bOdd, values, PROFILE);
      curveD.addProperty("D" + size / 1024, r.mse());
      line("curve", "D%dk trained on %d games: odd MSE %.6f, minus B %+.6f ± %.6f", size / 1024, arm.games(),
          r.mse(), vsB.mseDifference(), vsB.standardError());
    }
    CellReport d32Odd = report("D32-even@odd", odd, subset(games, d32.cross, false));
    curveD.addProperty("D32", d32Odd.mse());
    results.add("curveD", curveD);

    verdict(games, a, b, c, reportA, grp, odd, cEven);
    CellStats.Paired dMinusB = CellStats.paired(games, b, d32.cross, PROFILE);
    line("record", "D(32k) − B %+.6f ± %.6f (promising if ≤ 0.0002)",
        dMinusB.mseDifference(), dMinusB.standardError());
    results.addProperty("d32MinusB", dMinusB.mseDifference());
    results.addProperty("d32MinusBSe", dMinusB.standardError());
    for (Pair pair : new Pair[] {new Pair("B", b), c, d32}) {
      CellStats.Absolute absolute =
          CellStats.absolute(games, pair.cross, PROFILE, GateEvaluator.CELL_LIMIT, GateEvaluator.CELL_Z);
      line("record", "%s absolute calibration RMS %.5f floor %.5f flagged %d",
          pair.name, absolute.rms(), absolute.floor(), absolute.flagged());
      results.addProperty("absoluteRms" + pair.name, absolute.rms());
      results.addProperty("absoluteFloor" + pair.name, absolute.floor());
      results.addProperty("absoluteFlagged" + pair.name, absolute.flagged());
    }
    determinism(grp, even);
    write(started);
  }

  private void write(long started) throws IOException {
    line("done", "%.0f s", seconds(started));
    results.addProperty("seconds", seconds(started));
    Files.writeString(
        out.resolve("study.json"),
        new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create().toJson(results),
        StandardCharsets.UTF_8);
  }

  /** 腕 F（点数のなめらかな関数）: 較正の測り、A・B との対の差、S3 で 4 位が 3 位に近い升目、絶対の較正。 */
  private void scoreSmooth(List<RecordedGame> games, PhiValues a, PhiValues b) throws IOException {
    long t = System.nanoTime();
    PhiValues f = ScorePhi.offline(PhiEntryKind.SCORE_SMOOTH, PROFILE).gameValues(games);
    line("speed", "score-smooth %.0f hand ends/s", handEnds(games) / seconds(t));
    report("F", games, f);
    CellStats.Paired vsB = CellStats.paired(games, b, f, PROFILE);
    CellStats.Paired vsA = CellStats.paired(games, a, f, PROFILE);
    CellStats.Paired s3 = CellStats.residual(games, f, PROFILE, GrpStudyBench::nearFourthS3);
    CellStats.Absolute absolute = CellStats.absolute(games, f, PROFILE, GateEvaluator.CELL_LIMIT, GateEvaluator.CELL_Z);
    line("record", "F − B MSE %+.6f ± %.6f, F − A %+.6f ± %.6f, S3 4th within 4000 of 3rd %+.4f ± %.4f,"
            + " absolute calibration RMS %.5f floor %.5f flagged %d",
        vsB.mseDifference(), vsB.standardError(), vsA.mseDifference(), vsA.standardError(), s3.mseDifference(),
        s3.standardError(), absolute.rms(), absolute.floor(), absolute.flagged());
    results.addProperty("fMinusB", vsB.mseDifference());
    results.addProperty("fMinusBSe", vsB.standardError());
    results.addProperty("fMinusA", vsA.mseDifference());
    results.addProperty("fMinusASe", vsA.standardError());
    results.addProperty("fS3NearFourth", s3.mseDifference());
    results.addProperty("absoluteRmsF", absolute.rms());
    results.addProperty("absoluteFloorF", absolute.floor());
    results.addProperty("absoluteFlaggedF", absolute.flagged());
  }

  /** 手順 0: 元の GRP を maxBatch の行の固定のバッチで計算した順位分布が CSV の p0..p15 と 1e-5 以内。外れたらここで止める。 */
  private void step0(AuditCsvGames audit, PhiValues a, int handEnds) {
    double worstP = 0, worstPhi = 0;
    for (int h = 0; h < handEnds; h++) {
      for (int k = 0; k < 16; k++)
        worstP = Math.max(worstP, Math.abs(a.rankProbabilities()[h][k] - audit.marginals()[h][k]));
      for (int seat = 0; seat < 4; seat++)
        worstPhi = Math.max(worstPhi, Math.abs(a.utilities()[h][seat] - audit.phi()[h][seat]));
    }
    line("step0", "max |p − CSV p| %.2e, max |Φ − CSV Φ| %.2e (line %.0e) -> %s", worstP, worstPhi, STEP0_LINE,
        worstP <= STEP0_LINE ? "OK" : "STOP");
    results.addProperty("step0MaxP", worstP);
    results.addProperty("step0MaxPhi", worstPhi);
    Assert.assertTrue(worstP <= STEP0_LINE, "step 0: the original GRP does not reproduce the audit CSV");
  }

  /** 道具の確認 O1〜O4（床・警報の線・誤警報の率）。 */
  private void tools(List<RecordedGame> games, CellReport reportA, CellReport reportB, PhiValues b) {
    boolean o1 = reportA.dSb() >= 3.0e-4 && reportA.dSb() <= 4.4e-4;
    boolean o2 = reportB.dSb() <= 0.5e-4;
    line("check", "O1 D_SB(A) %.3e in [3.0e-4, 4.4e-4] -> %s", reportA.dSb(), o1 ? "OK" : "FAIL");
    line("check", "O2 D_SB(B) %.3e ≤ 0.5e-4 -> %s", reportB.dSb(), o2 ? "OK" : "FAIL");
    boolean o3 = true;
    StringBuilder b4 = new StringBuilder();
    for (HandGroup group : HandGroup.values()) {
      double mine = reportA.b4()[group.ordinal()], audit = AUDIT_B4[group.ordinal()];
      o3 &= Math.abs(mine - audit) <= 0.003;
      b4.append(String.format(Locale.ROOT, " %s %+.4f(%+.4f)", group, mine, audit));
    }
    line("check", "O3 B4(A) vs audit:%s -> %s", b4, o3 ? "OK" : "FAIL");
    double floor = reportB.dSpg();
    double sd = CellStats.bootstrapDSpgSd(games, b, PROFILE, RESAMPLES, 17);
    double sdEwma = sd * Math.sqrt(EWMA / (2 - EWMA));
    double on = floor + ON_K * sdEwma, off = floor + OFF_K * sdEwma;
    Alarm stationary = alarms(floor, sd, on, off, false, 1);
    Alarm consecutive = alarms(floor, sd, on, off, false, 2);
    Alarm restart = alarms(floor, sd, on, off, true, 2);
    line("check", "O4 floor F = D_SPG(B) %.3e, cohort SD(D_SPG) %.3e (%d game resamples of %d games),"
            + " SD_EWMA %.3e, ON %.3e, OFF %.3e",
        floor, sd, RESAMPLES, games.size(), sdEwma, on, off);
    line("check", "O4 false alarms over %d iterations (stationary EWMA): P(any) %.4f mean %.3f (one crossing),"
            + " P(any) %.4f mean %.3f (two in a row); after a restart from one cohort P(any in first 16) %.4f",
        ITERATIONS, stationary.any, stationary.mean, consecutive.any, consecutive.mean, restart.any);
    results.addProperty("o1", o1);
    results.addProperty("o2", o2);
    results.addProperty("o3", o3);
    results.addProperty("floorF", floor);
    results.addProperty("cohortSdDSpg", sd);
    results.addProperty("sdEwma", sdEwma);
    results.addProperty("onLine", on);
    results.addProperty("offLine", off);
    results.addProperty("falseAlarmAnyOne", stationary.any);
    results.addProperty("falseAlarmMeanOne", stationary.mean);
    results.addProperty("falseAlarmAnyTwo", consecutive.any);
    results.addProperty("falseAlarmMeanTwo", consecutive.mean);
    results.addProperty("falseAlarmRestart16", restart.any);
  }

  /** 誤警報の模擬の結果: 1 回以上の確率と回数の平均。 */
  private record Alarm(double any, double mean) {}

  /**
   * 1 コホートの D_SPG が N(F, SD) の独立な値のとき、EWMA が ON の線に run 回続けて届くと警報、OFF の線まで下がると解く。
   *
   * @param restart 切り替えの直後（EWMA を最初のコホートの値から始め、最初の 16 反復だけ見る）なら true、定常（EWMA の定常分布から始め
   *     {@value #ITERATIONS} 反復）なら false
   */
  private static Alarm alarms(double floor, double sd, double on, double off, boolean restart, int run) {
    SplittableRandom random = new SplittableRandom(23);
    double sdEwma = sd * Math.sqrt(EWMA / (2 - EWMA));
    int steps = restart ? 16 : ITERATIONS;
    long any = 0, total = 0;
    for (int s = 0; s < SIMULATIONS; s++) {
      double ewma = restart ? floor + sd * gaussian(random) : floor + sdEwma * gaussian(random);
      boolean alarmed = false;
      int streak = 0, count = 0;
      for (int t = 0; t < steps; t++) {
        if (t > 0) ewma = (1 - EWMA) * ewma + EWMA * (floor + sd * gaussian(random));
        if (alarmed) {
          if (ewma <= off) alarmed = false;
          continue;
        }
        streak = ewma >= on ? streak + 1 : 0;
        if (streak >= run) {
          alarmed = true;
          streak = 0;
          count++;
        }
      }
      if (count > 0) any++;
      total += count;
    }
    return new Alarm(any / (double) SIMULATIONS, total / (double) SIMULATIONS);
  }

  /** 自己対局の GRP へ進むかの判定（GO / NO-GO / HOLD）と、偶奇の ΔΦ の RMS・差の変化の床。 */
  private void verdict(
      List<RecordedGame> games, PhiValues a, PhiValues b, Pair c, CellReport reportA, Path grp,
      List<RecordedGame> odd, Arm cEven) throws IOException {
    CellReport reportC = CellStats.measure(games, c.cross, PROFILE);
    CellStats.Paired cMinusB = CellStats.paired(games, b, c.cross, PROFILE);
    CellStats.Paired s3 = CellStats.residual(games, c.cross, PROFILE, GrpStudyBench::nearFourthS3);
    CellStats.Paired s3A = CellStats.residual(games, a, PROFILE, GrpStudyBench::nearFourthS3);
    CellStats.Paired s3B = CellStats.residual(games, b, PROFILE, GrpStudyBench::nearFourthS3);
    CellStats.CellTable cellsA = CellStats.spgTable(games, a, PROFILE);
    CellStats.CellTable cellsC = CellStats.spgTable(games, c.cross, PROFILE);
    int worse = 0;
    for (int cell = 0; cell < CellStats.SPG_CELLS; cell++)
      if (Math.abs(cellsC.means()[cell]) - Math.abs(cellsA.means()[cell]) > 2 * cellsC.errors()[cell]) {
        worse++;
        line("verdict", "worse cell %s rank %d gap bin %d (%.0f seat-hands): A %+.4f C %+.4f ± %.4f",
            HandGroup.values()[cell / 16], cell / 4 % 4 + 1, cell % 4, cellsC.counts()[cell], cellsA.means()[cell],
            cellsC.means()[cell], cellsC.errors()[cell]);
      }
    GateEvaluator.Probe probe = GateEvaluator.Probe.sample(odd, GateEvaluator.PROBE_SAMPLES, 31);
    double monotonicA, monotonicC;
    try (CheckpointPhi original = CheckpointPhi.open(grp, new double[0], null, PROFILE);
        CheckpointPhi candidate = CheckpointPhi.open(cEven.trained.directory(), new double[0], null, PROFILE)) {
      monotonicA = probe.monotonicRate(probe.respond(original));
      monotonicC = probe.monotonicRate(probe.respond(candidate));
    }
    boolean go1 = cMinusB.mseDifference() <= 0;
    boolean go2 = Math.abs(s3.mseDifference()) <= 0.016;
    boolean go3 = worse == 0;
    boolean go4 = reportC.rankNll() < reportA.rankNll();
    boolean go5 = monotonicC <= monotonicA;
    boolean noGo = cMinusB.mseDifference() > 2 * cMinusB.standardError();
    String verdict = go1 && go2 && go3 && go4 && go5 ? "GO" : noGo ? "NO-GO" : "HOLD";
    line("verdict", "MSE(C) − MSE(B) %+.6f ± %.6f (≤ 0: %s)", cMinusB.mseDifference(), cMinusB.standardError(), go1);
    line("verdict", "C S3 4th within 4000 of 3rd: r %+.4f ± %.4f (|r| ≤ 0.016: %s); A %+.4f, B %+.4f",
        s3.mseDifference(), s3.standardError(), go2, s3A.mseDifference(), s3B.mseDifference());
    line("verdict", "cells where C is worse than A by 2 SE: %d (none: %s)", worse, go3);
    line("verdict", "rank NLL C %.5f vs A %.5f (C < A: %s)", reportC.rankNll(), reportA.rankNll(), go4);
    line("verdict", "monotonic violations C %.5f vs A %.5f on %d"
            + " odd hand ends (C ≤ A: %s)",
        monotonicC, monotonicA, probe.samples(), go5);
    line("verdict", "stage 3: %s (NO-GO if MSE(C) > MSE(B) + 2 SE: %s)", verdict, noGo);
    results.addProperty("verdict", verdict);
    results.addProperty("cMinusB", cMinusB.mseDifference());
    results.addProperty("cMinusBSe", cMinusB.standardError());
    results.addProperty("cS3NearFourth", s3.mseDifference());
    results.addProperty("cWorseCells", worse);
    results.addProperty("rankNllA", reportA.rankNll());
    results.addProperty("rankNllC", reportC.rankNll());
    results.addProperty("monotonicA", monotonicA);
    results.addProperty("monotonicC", monotonicC);
  }

  /** 南 3 局が終わった局末で今 4 位、3 位との点差が 4000 未満の席（grp_calib_util.py の near）。 */
  private static boolean nearFourthS3(HandGroup group, int rank, int gap) {
    return group == HandGroup.S3 && rank == 3 && gap < 4000;
  }

  /** 同じ形の腕を半荘の偶奇の 2 つの折りで学んだ組: 交差の値（偶の半荘は奇の折りの網、奇は偶の網）と、偶奇の ΔΦ の RMS・差の変化の床。 */
  private record Pair(String name, PhiValues cross) {}

  private Pair pair(String name, List<RecordedGame> games, Arm evenArm, Arm oddArm) throws IOException {
    PhiValues fromEven, fromOdd;
    double[] diff;
    GateEvaluator.Probe probe = GateEvaluator.Probe.sample(games, GateEvaluator.PROBE_SAMPLES, 37);
    try (CheckpointPhi evenPhi = CheckpointPhi.open(evenArm.trained.directory(), new double[0], null, PROFILE);
        CheckpointPhi oddPhi = CheckpointPhi.open(oddArm.trained.directory(), new double[0], null, PROFILE)) {
      fromEven = evenPhi.gameValues(games);
      fromOdd = oddPhi.gameValues(games);
      diff = probe.diffShifts(probe.respond(evenPhi), probe.respond(oddPhi));
    }
    double rms = rms(fromEven, fromOdd);
    PhiValues cross = merge(games, subset(games, fromOdd, true), subset(games, fromEven, false));
    report(name, games, cross);
    line("record", "%s odd/even ΔΦ RMS %.5f; difference-change floor δ=1000 %.5f δ=4000 %.5f δ=12000 %.5f",
        name, rms, diff[0], diff[1], diff[2]);
    results.addProperty("twinRms" + name, rms);
    JsonArray floors = new JsonArray();
    for (double value : diff) floors.add(value);
    results.add("diffShiftFloor" + name, floors);
    return new Pair(name, cross);
  }

  /** 席局あたりの ΔΦ の RMS。 */
  private static double rms(PhiValues a, PhiValues b) {
    double squares = 0;
    long n = 0;
    for (int h = 0; h < a.size(); h++)
      for (int seat = 0; seat < 4; seat++) {
        double d = a.utilities()[h][seat] - b.utilities()[h][seat];
        squares += d * d;
        n++;
      }
    return Math.sqrt(squares / n);
  }

  /** 学んだ腕。 */
  private record Arm(String name, GrpGameTrainer.Trained trained, int games, double sequencesPerSecond) {}

  private Arm train(
      String name,
      GrpInit init,
      Path start,
      List<RecordedGame> fold,
      int limit,
      double learningRate,
      int epochs,
      long seed,
      GrpGameTrainer.EpochListener listener)
      throws IOException {
    List<RecordedGame> validation = fold.subList(fold.size() - VALIDATION, fold.size());
    List<RecordedGame> training = fold.subList(0, Math.min(limit, fold.size() - VALIDATION));
    Path directory = out.resolve("grp-" + name);
    deleteTree(directory);
    GrpTrainingPlan plan = new GrpTrainingPlan(init, 128, learningRate, epochs, seed, Duration.ofHours(3));
    long t = System.nanoTime();
    GrpGameTrainer.Trained trained =
        GrpGameTrainer.train(start, training, validation, plan, () -> false, directory, listener);
    double perGame = handEnds(training) / (double) training.size();
    Arm arm = new Arm(name, trained, training.size(), trained.gamesPerSecond() * perGame);
    line("train", "%s %s lr %.0e on %d games: %d epochs %d steps, train NLL %.5f validation NLL"
            + " %.5f, %.0f games/s %.0f sequences/s, %.0f s, sha %s",
        name, init, learningRate, training.size(), trained.epochs(), trained.steps(), trained.trainLoss(),
        trained.validationLoss(), trained.gamesPerSecond(), arm.sequencesPerSecond, seconds(t),
        trained.sha256().substring(0, 16));
    JsonObject json = new JsonObject();
    json.addProperty("init", init.name());
    json.addProperty("games", training.size());
    json.addProperty("epochs", trained.epochs());
    json.addProperty("steps", trained.steps());
    json.addProperty("trainNll", trained.trainLoss());
    json.addProperty("validationNll", trained.validationLoss());
    json.addProperty("gamesPerSecond", trained.gamesPerSecond());
    json.addProperty("sequencesPerSecond", arm.sequencesPerSecond);
    json.addProperty("seconds", seconds(t));
    json.addProperty("sha256", trained.sha256());
    results.add("train-" + name, json);
    Assert.assertEquals(trained.ending(), GrpGameTrainer.Ending.COMPLETED, name);
    return arm;
  }

  /** 同じ種・同じ入力で 2 回学んで params の sha が一致するか（温め 1 epoch、4,096 半荘）。 */
  private void determinism(Path grp, List<RecordedGame> even) throws IOException {
    List<RecordedGame> slice = even.subList(0, Math.min(4096, even.size()));
    String[] shas = new String[2];
    for (int run = 0; run < 2; run++) {
      Path directory = out.resolve("grp-repeat-" + run);
      deleteTree(directory);
      GrpTrainingPlan plan = new GrpTrainingPlan(GrpInit.WARM, 128, 2e-4, 1, 7, Duration.ofHours(1));
      shas[run] = GrpGameTrainer.train(grp, slice, List.of(), plan, () -> false, directory).sha256();
    }
    line("record", "two trainings with the same seed: sha %s / %s -> %s",
        shas[0].substring(0, 16), shas[1].substring(0, 16), shas[0].equals(shas[1]) ? "same" : "different");
    results.addProperty("sameSha", shas[0].equals(shas[1]));
  }

  private PhiValues phi(Arm arm, List<RecordedGame> games) throws IOException {
    try (CheckpointPhi phi = CheckpointPhi.open(arm.trained.directory(), new double[0], null, PROFILE)) {
      return phi.gameValues(games);
    }
  }

  /** 学習中の重みの Φ（学習曲線）。 */
  private static PhiValues values(Model model, List<RecordedGame> games) {
    List<float[]> prefixes = new ArrayList<>();
    for (RecordedGame game : games) for (int k = 0; k < game.continuingHandEnds(); k++) prefixes.add(game.prefix(k));
    float[][] marginals =
        GrpSequenceForward.marginals(model, EpsilonSettings.defaults().bind(GrpInferenceSettings.class).maxBatch(), prefixes);
    double[][] utilities = new double[marginals.length][4], ranks = new double[marginals.length][16];
    for (int h = 0; h < marginals.length; h++)
      for (int seat = 0; seat < 4; seat++) {
        for (int rank = 0; rank < 4; rank++) {
          ranks[h][4 * seat + rank] = marginals[h][4 * seat + rank];
          utilities[h][seat] += marginals[h][4 * seat + rank] * PROFILE.utilityForRankDouble(rank);
        }
      }
    return new PhiValues(utilities, ranks);
  }

  private CellReport report(String name, List<RecordedGame> games, PhiValues phi) {
    CellReport r = CellStats.measure(games, phi, PROFILE);
    StringBuilder groups = new StringBuilder();
    for (HandGroup group : HandGroup.values())
      groups.append(String.format(Locale.ROOT, " %s B4 %+.4f β %.3f z %+.1f;", group, r.b4()[group.ordinal()],
          r.slope()[group.ordinal()], r.slopeZ()[group.ordinal()]));
    line("arm", "%s games %d seat-hands %d MSE %.6f D_SPG %.3e D_SB %.3e rank NLL %.5f Brier %.5f 4th ECE %.4f |%s",
        name, r.games(), r.seatHands(), r.mse(), r.dSpg(), r.dSb(), r.rankNll(), r.brier(), r.fourthEce(), groups);
    JsonObject json = new JsonObject();
    json.addProperty("games", r.games());
    json.addProperty("seatHands", r.seatHands());
    json.addProperty("mse", r.mse());
    json.addProperty("dSpg", r.dSpg());
    json.addProperty("dSb", r.dSb());
    json.addProperty("rankNll", r.rankNll());
    json.addProperty("brier", r.brier());
    json.addProperty("fourthEce", r.fourthEce());
    JsonArray b4 = new JsonArray(), slope = new JsonArray(), z = new JsonArray();
    for (int g = 0; g < r.b4().length; g++) {
      b4.add(r.b4()[g]);
      slope.add(r.slope()[g]);
      z.add(r.slopeZ()[g]);
    }
    json.add("b4", b4);
    json.add("slope", slope);
    json.add("slopeZ", z);
    results.add("arm-" + name, json);
    return r;
  }

  /** grp_calib_util.py と同じ表（段 × Φ の 24 区分、区分の残差の和 /（数 + 50））を当てて JSON に書き、読む。 */
  private PhiCalibration fitTable(List<RecordedGame> games, PhiValues a, String fold) throws IOException {
    double[][] sums = new double[TABLE_STAGES][TABLE_BINS], counts = new double[TABLE_STAGES][TABLE_BINS];
    int h = 0;
    for (RecordedGame game : games) {
      double[] u = game.finalUtilities(PROFILE);
      for (int k = 0; k < game.continuingHandEnds(); k++, h++) {
        int stage = Math.min(k == 0 ? 0 : game.starts().get(k - 1).kyokuIndex(), TABLE_STAGES - 1);
        for (int seat = 0; seat < 4; seat++) {
          double value = a.utilities()[h][seat];
          int bin = Math.max(0, Math.min(TABLE_BINS - 1, (int) ((value - TABLE_LO) / (TABLE_HI - TABLE_LO) * TABLE_BINS)));
          sums[stage][bin] += u[seat] - value;
          counts[stage][bin]++;
        }
      }
    }
    JsonObject json = new JsonObject();
    JsonArray stages = new JsonArray();
    for (String stage : new String[] {"E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4", "W+"}) stages.add(stage);
    json.add("stages", stages);
    json.addProperty("lo", TABLE_LO);
    json.addProperty("hi", TABLE_HI);
    json.addProperty("bins", TABLE_BINS);
    JsonObject table = new JsonObject();
    for (int stage = 0; stage < TABLE_STAGES; stage++) {
      JsonArray cells = new JsonArray();
      for (int bin = 0; bin < TABLE_BINS; bin++) cells.add(sums[stage][bin] / (counts[stage][bin] + TABLE_PRIOR));
      table.add(Integer.toString(stage), cells);
    }
    json.add("table", table);
    Path file = out.resolve("calib-" + fold + ".json");
    Files.writeString(file, json.toString(), StandardCharsets.UTF_8);
    return PhiCalibration.load(file);
  }

  /** Φ の値に較正の表を掛けたもの（順位分布はそのまま）。 */
  private static PhiValues tabled(List<RecordedGame> games, PhiValues phi, PhiCalibration calibration) {
    double[][] utilities = new double[phi.size()][];
    int h = 0;
    for (RecordedGame game : games)
      for (int k = 0; k < game.continuingHandEnds(); k++, h++)
        utilities[h] = calibration.apply(phi.utilities()[h], game.prefix(k));
    return new PhiValues(utilities, phi.rankProbabilities());
  }

  /** 全ての半荘の値から、偶（evenFold）か奇の折りの半荘の行だけ（順は保つ）。 */
  private static PhiValues subset(List<RecordedGame> games, PhiValues values, boolean evenFold) {
    List<double[]> utilities = new ArrayList<>(), ranks = new ArrayList<>();
    int h = 0;
    for (RecordedGame game : games)
      for (int k = 0; k < game.continuingHandEnds(); k++, h++)
        if ((game.gameId() % 2 == 0) == evenFold) {
          utilities.add(values.utilities()[h]);
          ranks.add(values.rankProbabilities()[h]);
        }
    return new PhiValues(utilities.toArray(double[][]::new), ranks.toArray(double[][]::new));
  }

  /** 偶の折りの値と奇の折りの値を全ての半荘の順に並べ直す。 */
  private static PhiValues merge(List<RecordedGame> games, PhiValues evenValues, PhiValues oddValues) {
    int size = evenValues.size() + oddValues.size();
    double[][] utilities = new double[size][], ranks = new double[size][];
    int h = 0, e = 0, o = 0;
    for (RecordedGame game : games) {
      boolean isEven = game.gameId() % 2 == 0;
      for (int k = 0; k < game.continuingHandEnds(); k++, h++) {
        PhiValues from = isEven ? evenValues : oddValues;
        int row = isEven ? e++ : o++;
        utilities[h] = from.utilities()[row];
        ranks[h] = from.rankProbabilities()[row];
      }
    }
    return new PhiValues(utilities, ranks);
  }

  private static int handEnds(List<RecordedGame> games) {
    int count = 0;
    for (RecordedGame game : games) count += game.continuingHandEnds();
    return count;
  }

  private static double seconds(long started) {
    return (System.nanoTime() - started) / 1e9;
  }

  private static double gaussian(SplittableRandom random) {
    double u1 = 1 - random.nextDouble(), u2 = random.nextDouble();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  private static void line(String kind, String format, Object... arguments) {
    log.info("PRISM-GRP-STUDY {} {}", kind, String.format(Locale.ROOT, format, arguments));
  }

  private static void deleteTree(Path root) throws IOException {
    if (!Files.exists(root)) return;
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }

  private static Path required(String property) {
    String given = System.getProperty(property);
    if (given == null || given.isBlank()) throw new SkipException("set -D" + property + "=<path> to run the GRP study");
    return Path.of(given);
  }
}
