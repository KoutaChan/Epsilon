package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * 候補の GRP の関門。線はコードの定数（設定の鍵にしない）。REFIT の列は良くなった・順位の当てはまり・変化の幅・単調・傾き・形、
 * FIRST_REPLACEMENT の列は絶対の較正（局 × Φ 区間の残差の重み付き RMS が雑音の床 + 余裕以下で、大きい升目がない）・安定（偶奇の 2 候補の
 * ΔΦ の RMS）・単調・形・窓の半荘の数。
 *
 * <p>点を動かす調べ（{@link Probe}）: 取り置きから決まった種で選んだ局末で、次局の点数を席 j から i へ δ ∈ {1000, 4000, 12000} 点動かした
 * 公開系列（最後の局境界だけを替える）を評価する。逆に動くのは Φ_i が下がるか Φ_j が上がること（{@value #MONOTONIC_TOLERANCE} を超えて）。
 * 差の変化は δ と局の区分ごとの (ΔΦ_i 候補 − ΔΦ_i 使用中) の RMS の最大。傾きは δ = 4000 の ΔΦ_i の平均の比（候補/使用中）。
 */
public final class GateEvaluator {
  /** 良くなった: 対の MSE の差がこの SE の倍数より下。 */
  static final double BETTER_SE = 2;

  /** 順位の当てはまり: NLL の悪化の上限（対の差の SE の倍数）と 4 位の ECE の悪化の上限。 */
  static final double NLL_SE = 1, FOURTH_ECE_WORSENING = 0.005;

  /** 変化の幅: 席局あたりの平均 |ΔΦ|、局ごとの平均 |ΔΦ|、局ごとの 99% 点、差の変化の上限。 */
  static final double MEAN_SHIFT = 0.03, GROUP_MEAN_SHIFT = 0.05, GROUP_P99_SHIFT = 0.20, DIFF_SHIFT = 0.02;

  /** 単調: 逆に動く率の上限 max(倍率 × 使用中, 床)。点数だけの Φ からは床だけ。 */
  static final double MONOTONIC_FACTOR = 1.5, MONOTONIC_FLOOR = 0.005;

  /** 傾き: 4000 点の傾きの比の範囲。 */
  static final double SLOPE_LOW = 0.67, SLOPE_HIGH = 1.5;

  /** 絶対の較正: 重み付き RMS の雑音の床からの余裕、大きいとみなす升目の |r| と z。 */
  static final double CALIBRATION_MARGIN = 0.005, CELL_LIMIT = 0.03, CELL_Z = 3;

  /** 安定: 偶奇の 2 候補の ΔΦ の RMS の上限。 */
  static final double TWIN_RMS = 0.015;

  /** 安定の数値の名前（{@link TrustCriteria#merge} が 2 回目の関門から読む）。 */
  static final String TWIN_RMS_MEASURE = "twinRms";

  /** 逆に動いたとみなす Φ の変化の大きさ（float の丸めより大きく）。 */
  static final double MONOTONIC_TOLERANCE = 1e-6;

  /** 点を動かす調べの局末の数と種。 */
  static final int PROBE_SAMPLES = 4096;

  private static final long PROBE_SEED = 0x6A7E5EEDL;

  private GateEvaluator() {}

  /**
   * 関門を通す。取り置きに半荘が続く局末がなければ NOT_ENOUGH_DATA だけを返す（測りは空、active・candidate は null）。
   *
   * @throws IOException Φ が評価できないとき
   * @throws IllegalArgumentException FIRST_REPLACEMENT で twin が null のとき
   */
  public static GateReport evaluate(GateInput input, EpsilonUtilityProfile profile)
      throws IOException {
    if (input.mode() == GateMode.FIRST_REPLACEMENT && input.twin() == null)
      throw new IllegalArgumentException("FIRST_REPLACEMENT needs the twin candidate");
    List<RecordedGame> holdout = input.holdout().stream().filter(game -> game.continuingHandEnds() > 0).toList();
    if (holdout.isEmpty())
      return new GateReport(
          input.mode(), GateResult.NOT_ENOUGH_DATA, List.of(GateResult.NOT_ENOUGH_DATA), Map.of(), null, null);
    PhiValues active = input.active().gameValues(holdout);
    CellReport activeReport = CellStats.measure(holdout, active, profile);
    Map<String, Double> measures = new LinkedHashMap<>();
    measures.put("windowGames", (double) input.windowGames());
    PhiValues candidate = finite(input.candidate(), holdout);
    if (!input.candidateValid() || candidate == null)
      return new GateReport(
          input.mode(), GateResult.INVALID_OUTPUT, List.of(GateResult.INVALID_OUTPUT), measures, activeReport, null);
    CellReport candidateReport = CellStats.measure(holdout, candidate, profile);
    Probe probe = Probe.sample(holdout, PROBE_SAMPLES, PROBE_SEED);
    double[][] candidateProbe = probe.respond(input.candidate());
    if (!finite(candidateProbe))
      return new GateReport(
          input.mode(), GateResult.INVALID_OUTPUT, List.of(GateResult.INVALID_OUTPUT), measures, activeReport,
          candidateReport);
    EnumSet<GateResult> failures = EnumSet.noneOf(GateResult.class);
    double monotonic = probe.monotonicRate(candidateProbe);
    measures.put("monotonicCandidate", monotonic);
    switch (input.mode()) {
      case REFIT -> {
        CellStats.Paired mse = CellStats.paired(holdout, active, candidate, profile);
        CellStats.Paired nll = CellStats.pairedNll(holdout, active, candidate, profile);
        double eceWorsening = candidateReport.fourthEce() - activeReport.fourthEce();
        Shift shift = Shift.of(holdout, active, candidate);
        double[][] activeProbe = probe.respond(input.active());
        double diffShift = probe.diffShift(candidateProbe, activeProbe);
        double activeMonotonic = probe.monotonicRate(activeProbe);
        double monotonicLine = Math.max(MONOTONIC_FACTOR * activeMonotonic, MONOTONIC_FLOOR);
        double slopeRatio =
            probe.meanGain(candidateProbe, Probe.SLOPE_POINTS) / probe.meanGain(activeProbe, Probe.SLOPE_POINTS);
        measures.put("mseDifference", mse.mseDifference());
        measures.put("mseDifferenceSe", mse.standardError());
        measures.put("nllDifference", nll.mseDifference());
        measures.put("nllDifferenceSe", nll.standardError());
        measures.put("fourthEceWorsening", eceWorsening);
        measures.put("meanShift", shift.mean);
        measures.put("maxGroupMeanShift", shift.maxGroupMean);
        measures.put("maxGroupP99Shift", shift.maxGroupP99);
        measures.put("diffShift", diffShift);
        measures.put("monotonicActive", activeMonotonic);
        measures.put("monotonicLine", monotonicLine);
        measures.put("slopeRatio", slopeRatio);
        if (!(mse.mseDifference() < -BETTER_SE * mse.standardError())) failures.add(GateResult.NOT_BETTER);
        if (!(nll.mseDifference() <= NLL_SE * nll.standardError()) || !(eceWorsening <= FOURTH_ECE_WORSENING))
          failures.add(GateResult.RANK_FIT_WORSE);
        if (!(shift.mean <= MEAN_SHIFT)
            || !(shift.maxGroupMean <= GROUP_MEAN_SHIFT)
            || !(shift.maxGroupP99 <= GROUP_P99_SHIFT)
            || !(diffShift <= DIFF_SHIFT)) failures.add(GateResult.SHIFT_TOO_LARGE);
        if (!(monotonic <= monotonicLine)) failures.add(GateResult.MONOTONIC_VIOLATIONS);
        if (!(slopeRatio >= SLOPE_LOW && slopeRatio <= SLOPE_HIGH)) failures.add(GateResult.SLOPE_OUT_OF_RANGE);
      }
      case FIRST_REPLACEMENT -> {
        CellStats.Absolute calibration = CellStats.absolute(holdout, candidate, profile, CELL_LIMIT, CELL_Z);
        PhiValues twin = finite(input.twin(), holdout);
        double twinRms = twin == null ? Double.NaN : rms(candidate, twin);
        measures.put("calibrationRms", calibration.rms());
        measures.put("calibrationFloor", calibration.floor());
        measures.put("calibrationFlagged", (double) calibration.flagged());
        measures.put(TWIN_RMS_MEASURE, twinRms);
        measures.put("monotonicLine", MONOTONIC_FLOOR);
        if (twin == null) failures.add(GateResult.INVALID_OUTPUT);
        if (!(calibration.rms() <= calibration.floor() + CALIBRATION_MARGIN) || calibration.flagged() > 0)
          failures.add(GateResult.RANK_FIT_WORSE);
        if (!(twinRms <= TWIN_RMS)) failures.add(GateResult.UNSTABLE);
        if (!(monotonic <= MONOTONIC_FLOOR)) failures.add(GateResult.MONOTONIC_VIOLATIONS);
        if (input.windowGames() < input.trustGames()) failures.add(GateResult.NOT_ENOUGH_DATA);
      }
    }
    List<GateResult> failed = List.copyOf(failures);
    return new GateReport(
        input.mode(),
        failed.isEmpty() ? GateResult.PASS : failed.getFirst(),
        failed,
        measures,
        activeReport,
        candidateReport);
  }

  /** 全て有限なら Φ の値、評価が壊れた出力で落ちるか有限でない値があれば null。 */
  private static PhiValues finite(HandEndPhi phi, List<RecordedGame> games) throws IOException {
    PhiValues values;
    try {
      values = phi.gameValues(games);
    } catch (IllegalStateException invalid) {
      return null;
    }
    return finite(values.utilities()) && finite(values.rankProbabilities()) ? values : null;
  }

  private static boolean finite(double[][] rows) {
    for (double[] row : rows) for (double value : row) if (!Double.isFinite(value)) return false;
    return true;
  }

  /** 席局あたりの ΔΦ の RMS。 */
  static double rms(PhiValues a, PhiValues b) {
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

  /** 使用中から候補への |ΔΦ| の席局あたりの平均と、局の区分ごとの平均・99% 点の最大。 */
  record Shift(double mean, double maxGroupMean, double maxGroupP99) {
    static Shift of(List<RecordedGame> games, PhiValues active, PhiValues candidate) {
      int groups = HandGroup.values().length;
      int[] sizes = new int[groups];
      int h = 0;
      for (RecordedGame game : games)
        for (int k = 0; k < game.continuingHandEnds(); k++, h++) sizes[CellStats.endedGroup(game, k).ordinal()] += 4;
      double[][] shifts = new double[groups][];
      for (int g = 0; g < groups; g++) shifts[g] = new double[sizes[g]];
      int[] filled = new int[groups];
      double total = 0;
      h = 0;
      for (RecordedGame game : games)
        for (int k = 0; k < game.continuingHandEnds(); k++, h++) {
          int group = CellStats.endedGroup(game, k).ordinal();
          for (int seat = 0; seat < 4; seat++) {
            double shift = Math.abs(candidate.utilities()[h][seat] - active.utilities()[h][seat]);
            shifts[group][filled[group]++] = shift;
            total += shift;
          }
        }
      double maxMean = 0, maxP99 = 0;
      for (double[] values : shifts) {
        if (values.length == 0) continue;
        double sum = 0;
        for (double value : values) sum += value;
        Arrays.sort(values);
        maxMean = Math.max(maxMean, sum / values.length);
        maxP99 = Math.max(maxP99, values[Math.min(values.length - 1, (int) Math.ceil(0.99 * values.length) - 1)]);
      }
      return new Shift(total / (4.0 * h), maxMean, maxP99);
    }
  }

  /**
   * 点を動かす調べ。局末 s の入力は [s · (1 + M)]、その m 番目の動かし方は [s · (1 + M) + 1 + m]（M = 点 3 通り × 席の順序対 12 通り）。
   */
  static final class Probe {
    /** 動かす点。 */
    static final int[] POINTS = {1000, 4000, 12000};

    /** 傾きを測る点。 */
    static final int SLOPE_POINTS = 4000;

    private static final int PAIRS = 12, MOVES = POINTS.length * PAIRS, STRIDE = 1 + MOVES;
    private final List<HandEndInput> inputs;
    private final byte[] groups;

    private Probe(List<HandEndInput> inputs, byte[] groups) {
      this.inputs = inputs;
      this.groups = groups;
    }

    /** 選んだ局末の数。 */
    int samples() {
      return groups.length;
    }

    /**
     * 半荘が続く局末から、種 seed で重複なく最大 samples 個を選ぶ（局末の順に並べる）。
     *
     * @param games 取り置きの半荘
     */
    static Probe sample(List<RecordedGame> games, int samples, long seed) {
      List<long[]> all = new ArrayList<>();
      for (int g = 0; g < games.size(); g++)
        for (int k = 0; k < games.get(g).continuingHandEnds(); k++) all.add(new long[] {g, k});
      int n = Math.min(samples, all.size());
      SplittableRandom random = new SplittableRandom(seed);
      for (int i = 0; i < n; i++) Collections.swap(all, i, i + random.nextInt(all.size() - i));
      List<long[]> chosen = new ArrayList<>(all.subList(0, n));
      chosen.sort((a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
      List<HandEndInput> inputs = new ArrayList<>(n * STRIDE);
      byte[] groups = new byte[n];
      for (int s = 0; s < n; s++) {
        RecordedGame game = games.get((int) chosen.get(s)[0]);
        int k = (int) chosen.get(s)[1];
        HandEndInput base = game.input(k);
        groups[s] = (byte) CellStats.endedGroup(game, k).ordinal();
        inputs.add(base);
        HandStart next = base.next();
        for (int points : POINTS)
          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
              if (i == j) continue;
              int[] scores = next.scores().clone();
              scores[i] += points;
              scores[j] -= points;
              HandStart moved = new HandStart(next.kyokuIndex(), next.honba(), next.kyotaku(), scores);
              float[] prefix = base.prefix().clone();
              float[] feature = moved.feature();
              System.arraycopy(feature, 0, prefix, prefix.length - feature.length, feature.length);
              inputs.add(new HandEndInput(prefix, moved));
            }
      }
      return new Probe(inputs, groups);
    }

    /** 全ての入力の 4 席の Φ。 */
    double[][] respond(HandEndPhi phi) throws IOException {
      return phi.values(inputs).utilities();
    }

    /** 逆に動く率（動かし方ごとに受け手と出し手の 2 つを数える）。 */
    double monotonicRate(double[][] values) {
      long violations = 0, checks = 0;
      for (int s = 0; s < samples(); s++) {
        double[] base = values[s * STRIDE];
        int m = 0;
        for (int p = 0; p < POINTS.length; p++)
          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
              if (i == j) continue;
              double[] moved = values[s * STRIDE + 1 + m++];
              if (moved[i] - base[i] < -MONOTONIC_TOLERANCE) violations++;
              if (moved[j] - base[j] > MONOTONIC_TOLERANCE) violations++;
              checks += 2;
            }
      }
      return checks == 0 ? 0 : violations / (double) checks;
    }

    /** δ = points の受け手の ΔΦ_i の平均。 */
    double meanGain(double[][] values, int points) {
      int p = Arrays.binarySearch(POINTS, points);
      if (p < 0) throw new IllegalArgumentException("Points " + points + " are not probed");
      double sum = 0;
      long n = 0;
      for (int s = 0; s < samples(); s++) {
        double[] base = values[s * STRIDE];
        int m = p * PAIRS;
        for (int i = 0; i < 4; i++)
          for (int j = 0; j < 4; j++) {
            if (i == j) continue;
            sum += values[s * STRIDE + 1 + m++][i] - base[i];
            n++;
          }
      }
      return sum / n;
    }

    /** δ と局の区分ごとの (ΔΦ_i a − ΔΦ_i b) の RMS の最大。 */
    double diffShift(double[][] a, double[][] b) {
      double worst = 0;
      for (double value : diffShifts(a, b)) if (value > worst) worst = value;
      return worst;
    }

    /** δ ごと（{@link #POINTS} の順）の、局の区分ごとの (ΔΦ_i a − ΔΦ_i b) の RMS の最大。 */
    double[] diffShifts(double[][] a, double[][] b) {
      int groupCount = HandGroup.values().length;
      double[] out = new double[POINTS.length];
      for (int p = 0; p < POINTS.length; p++) {
        double[] squares = new double[groupCount];
        long[] counts = new long[groupCount];
        for (int s = 0; s < samples(); s++) {
          int m = p * PAIRS;
          for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
              if (i == j) continue;
              int row = s * STRIDE + 1 + m++;
              double d = (a[row][i] - a[s * STRIDE][i]) - (b[row][i] - b[s * STRIDE][i]);
              squares[groups[s]] += d * d;
              counts[groups[s]]++;
            }
        }
        for (int g = 0; g < groupCount; g++)
          if (counts[g] > 0) out[p] = Math.max(out[p], Math.sqrt(squares[g] / counts[g]));
      }
      return out;
    }
  }
}
