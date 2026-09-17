package com.epsilon.nano.ai.decision.arena;

import com.epsilon.config.settings.DecisionExplorationSchedule;
import com.epsilon.config.settings.DecisionFullSupportSettings;
import com.epsilon.config.settings.DecisionFullSupportSettings.AdaptiveExplorationSettings;
import com.epsilon.core.Action;
import com.epsilon.core.DecisionLearningRole;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 探索を加える前の合法手の確率分布から不確実性を測り、学習の反復開始時に固定した累積分布関数（CDF）で探索倍率を決める。
 *
 * <p>収集中のヒストグラムは次の反復から利用する。現在の反復で選ぶ行動は、GPU 処理や収集スレッドの完了順に影響されない。
 */
public final class DecisionAdaptiveExploration {

  public static final int HISTOGRAM_BINS = 2048;
  public static final int MINIMUM_CALIBRATION_SAMPLES = 1024;
  private static final double PROBABILITY_EPSILON = 1.0e-30;

  private final AdaptiveExplorationSettings settings;
  private final double unweightedConfiguredNodeMassMean;
  private Calibration calibration = Calibration.empty();
  private MacroSession activeSession;

  DecisionAdaptiveExploration(AdaptiveExplorationSettings settings) {
    this.settings = Objects.requireNonNull(settings, "settings");
    unweightedConfiguredNodeMassMean = 0.0;
  }

  public DecisionAdaptiveExploration(DecisionFullSupportSettings fullSupport) {
    DecisionFullSupportSettings actual = Objects.requireNonNull(fullSupport, "fullSupport");
    settings = actual.adaptiveExploration();
    unweightedConfiguredNodeMassMean =
        (actual.terminalGateExplorationMass()
                + actual.callGateExplorationMass()
                + actual.meldTypeExplorationMass()
                + actual.meldCandidateExplorationMass()
                + actual.kanGateExplorationMass()
                + actual.kanTypeExplorationMass()
                + actual.kanCandidateExplorationMass()
                + actual.discardIdentityExplorationMass()
                + actual.riichiGateExplorationMass())
            / 9.0;
  }

  /** 次の学習の反復が参照するCDFをチェックポイント境界で固定する。 */
  public synchronized State exportState() {
    requireNoActiveMacro();
    return new State(calibration.sourceSnapshotId, calibration.cumulative, calibration.hash);
  }

  /** チェックポイントまたはロールバックで、収集開始前のCDFへ戻す。 */
  public synchronized void restoreState(State state) {
    requireNoActiveMacro();
    Objects.requireNonNull(state, "state");
    long[] totals = new long[state.cumulative.length];
    if (state.sourceSnapshotId >= 0L) {
      for (int key = 0; key < totals.length; key++) {
        totals[key] = state.cumulative[key][HISTOGRAM_BINS - 1];
      }
    }
    // State が変更されない配列を保持し、Calibration は参照するだけなので、再度複製する必要はない。
    calibration = new Calibration(state.sourceSnapshotId, state.cumulative, totals, state.hash);
  }

  private void requireNoActiveMacro() {
    if (activeSession != null) {
      throw new IllegalStateException("adaptive exploration macro is active");
    }
  }

  /** CDFの保存形式。配列の複製は外部との入出力境界だけに限定する。 */
  public record State(long sourceSnapshotId, long[][] cumulative, String hash) {
    public State {
      if (sourceSnapshotId < -1L || cumulative.length != HistogramKey.values().length) {
        throw new IllegalArgumentException("invalid adaptive exploration calibration identity");
      }
      int expectedBins = sourceSnapshotId < 0L ? 0 : HISTOGRAM_BINS;
      for (long[] histogram : cumulative) {
        if (histogram.length != expectedBins) {
          throw new IllegalArgumentException("invalid adaptive exploration histogram size");
        }
        long previous = 0L;
        for (long value : histogram) {
          if (value < previous) {
            throw new IllegalArgumentException("adaptive exploration CDF must be nondecreasing");
          }
          previous = value;
        }
      }
      if (sourceSnapshotId >= 0L) {
        Objects.requireNonNull(hash, "hash");
      }
      cumulative = copyHistograms(cumulative);
    }

    @Override
    public long[][] cumulative() {
      return copyHistograms(cumulative);
    }

    private static long[][] copyHistograms(long[][] source) {
      long[][] result = new long[source.length][];
      for (int key = 0; key < source.length; key++) {
        result[key] = source[key].clone();
      }
      return result;
    }
  }

  /** 前回の学習の反復の変更不可 CDFだけを参照する収集セッションを開始する。 */
  public synchronized MacroSession beginMacro(long actorSnapshotId) {
    if (actorSnapshotId <= 0L) {
      throw new IllegalArgumentException("actorSnapshotId must be positive");
    }
    if (activeSession != null) {
      throw new IllegalStateException("adaptive exploration macro is already active");
    }
    activeSession = new MacroSession(actorSnapshotId, settings, calibration);
    return activeSession;
  }

  /** 今回の学習の反復の集計をレポートへ固定し、そのヒストグラムを次の学習の反復のCDFへ切り替える。 */
  public synchronized Report completeMacro(MacroSession session) {
    if (session == null || session != activeSession) {
      throw new IllegalArgumentException("adaptive exploration session is not active");
    }
    MacroAggregate aggregate = session.finish();
    Calibration next = Calibration.from(session.actorSnapshotId(), aggregate.histogram());
    Report report =
        aggregate.toReport(
            settings,
            calibration.isEmpty() ? null : calibration.sourceSnapshotId(),
            calibration.isEmpty() ? null : calibration.hash(),
            calibration.samples(),
            session.actorSnapshotId(),
            next.hash(),
            next.samples(),
            unweightedConfiguredNodeMassMean);
    calibration = next;
    activeSession = null;
    return report;
  }

  /** 収集失敗時は途中ヒストグラムを次の学習の反復へ持ち越さない。 */
  public synchronized void abortMacro(MacroSession session) {
    if (session == activeSession) {
      session.abort();
      activeSession = null;
    }
  }

  enum Kind {
    FORCED,
    AGARI_OR_ABORT,
    CALL,
    KAN,
    RIICHI,
    DISCARD
  }

  enum LookupSource {
    KIND,
    TURN,
    RESPONSE,
    GLOBAL,
    NONE
  }

  /** 一つの合法手分布から得る、探索前の不確実性。 */
  record Measurement(float normalizedEntropy, float topTwoAmbiguity, float uncertainty) {}

  /** CDF参照結果。未較正時は百分位がNaN、入力元がNONE。 */
  record Lookup(float percentile, LookupSource source) {
    private static Lookup none() {
      return new Lookup(Float.NaN, LookupSource.NONE);
    }

    boolean available() {
      return source != LookupSource.NONE;
    }
  }

  /** 選択時とエンジン結果確定時を結ぶ、行動履歴に永続化しない軽量な観測値。 */
  record Observation(
      Kind kind, Measurement measurement, Lookup lookup, float multiplier, int decile) {
    private static final int DECILE_CODES_PER_KIND = 11;

    private static Observation forced() {
      return new Observation(Kind.FORCED, null, Lookup.none(), 1.0f, -1);
    }

    int roleCode() {
      return 1 + kind.ordinal() * DECILE_CODES_PER_KIND + decile + 1;
    }

    private static Kind roleKind(int roleCode) {
      requireRoleCode(roleCode);
      return Kind.values()[(roleCode - 1) / DECILE_CODES_PER_KIND];
    }

    private static int roleDecile(int roleCode) {
      requireRoleCode(roleCode);
      return (roleCode - 1) % DECILE_CODES_PER_KIND - 1;
    }

    private static void requireRoleCode(int roleCode) {
      if (roleCode <= 0 || roleCode > Kind.values().length * DECILE_CODES_PER_KIND) {
        throw new IllegalArgumentException("invalid adaptive exploration role code: " + roleCode);
      }
    }
  }

  /** 一回の学習の反復中に全方策モデルプレイヤーが共有する収集セッション。CDF自体は変更不可。 */
  public static final class MacroSession {
    private final long actorSnapshotId;
    private final AdaptiveExplorationSettings settings;
    private final Calibration calibration;
    private final ConcurrentLinkedQueue<LocalAccumulator> accumulators =
        new ConcurrentLinkedQueue<>();
    private final ThreadLocal<LocalAccumulator> localAccumulator =
        ThreadLocal.withInitial(
            () -> {
              LocalAccumulator accumulator = new LocalAccumulator();
              accumulators.add(accumulator);
              return accumulator;
            });
    private final AtomicBoolean closed = new AtomicBoolean();

    private MacroSession(
        long actorSnapshotId, AdaptiveExplorationSettings settings, Calibration calibration) {
      this.actorSnapshotId = actorSnapshotId;
      this.settings = settings;
      this.calibration = calibration;
    }

    long actorSnapshotId() {
      return actorSnapshotId;
    }

    /** 合法手の確率分布を観測し、今回だけに使う探索倍率を返す。 */
    Observation decide(List<Action> legalActions, float[] rolloutPolicy) {
      return decide(legalActions, rolloutPolicy, 0.5f);
    }

    /**
     * CDF 区間内の同順位をプレイヤーごとの乱数で分散して、離散スコアでも百分位を一様に保つ。
     *
     * @param tieFraction 行動の無作為抽出とは独立な[0, 1)の決定論的乱数
     */
    Observation decide(List<Action> legalActions, float[] rolloutPolicy, double tieFraction) {
      requireOpen();
      if (!Double.isFinite(tieFraction) || tieFraction < 0.0 || tieFraction >= 1.0) {
        throw new IllegalArgumentException("tieFraction must be finite and in [0, 1)");
      }
      Kind kind = classify(legalActions);
      LocalAccumulator accumulator = localAccumulator.get();
      if (kind == Kind.FORCED) {
        return Observation.forced();
      }
      Measurement measurement = measure(rolloutPolicy, settings);
      accumulator.histogram.observe(kind, isResponse(legalActions), measurement.uncertainty());
      Lookup lookup =
          calibration.lookup(
              kind, isResponse(legalActions), measurement.uncertainty(), tieFraction);
      float multiplier =
          settings.schedule() == DecisionExplorationSchedule.PERCENTILE_SCALE && lookup.available()
              ? multiplier(lookup.percentile(), settings)
              : 1.0f;
      int decile = lookup.available() ? Math.min(9, (int) (lookup.percentile() * 10.0f)) : -1;
      return new Observation(kind, measurement, lookup, multiplier, decile);
    }

    /** 確率の下限を適用した後の探索分布と、実際に選択した候補の位置を記録する。 */
    void recordSelection(
        Observation observation, float[] rolloutPolicy, float[] behaviorPolicy, int selectedSlot) {
      requireOpen();
      Objects.requireNonNull(observation, "observation");
      localAccumulator.get().selection(observation, rolloutPolicy, behaviorPolicy, selectedSlot);
    }

    /** 選択した応答がCAUSALかPREEMPTEDかを確定後に記録する。 */
    void recordLearningRole(Observation observation, DecisionLearningRole role) {
      requireOpen();
      Objects.requireNonNull(observation, "observation");
      recordLearningRole(observation.roleCode(), role);
    }

    /** PendingDecisionが保持する有効要素のみのコードから種別/十分位別学習役割を記録する。 */
    void recordLearningRole(int roleCode, DecisionLearningRole role) {
      requireOpen();
      Objects.requireNonNull(role, "role");
      localAccumulator
          .get()
          .learningRole(Observation.roleKind(roleCode), Observation.roleDecile(roleCode), role);
    }

    private MacroAggregate finish() {
      if (!closed.compareAndSet(false, true)) {
        throw new IllegalStateException("adaptive exploration macro is already closed");
      }
      MacroAggregate aggregate = new MacroAggregate();
      for (LocalAccumulator accumulator : accumulators) {
        aggregate.merge(accumulator);
      }
      return aggregate;
    }

    private void abort() {
      closed.set(true);
    }

    private void requireOpen() {
      if (closed.get()) {
        throw new IllegalStateException("adaptive exploration macro is closed");
      }
    }
  }

  /** JSONへそのまま保存する学習の反復集計。 */
  public record Report(
      String schedule,
      int histogramBins,
      int minimumCalibrationSamples,
      Long calibrationSourceSnapshotId,
      String calibrationSourceHash,
      long calibrationSourceSamples,
      long collectedSnapshotId,
      String collectedHistogramHash,
      long collectedHistogramSamples,
      double unweightedConfiguredNodeMassMean,
      double meanScaledUnweightedConfiguredNodeMass,
      Summary overall,
      List<KindReport> byKind,
      List<DecileReport> byKindAndDecile) {

    public String summary() {
      return "schedule="
          + schedule
          + ",sourceSnapshot="
          + calibrationSourceSnapshotId
          + ",calibrated="
          + overall.calibratedSelections()
          + "/"
          + overall.nonForcedSelections()
          + ",meanScale="
          + String.format(java.util.Locale.ROOT, "%.4f", overall.meanMultiplier())
          + ",meanTv="
          + String.format(java.util.Locale.ROOT, "%.6f", overall.meanTotalVariation())
          + ",meanNodeMass="
          + String.format(java.util.Locale.ROOT, "%.6f", meanScaledUnweightedConfiguredNodeMass);
    }
  }

  record KindReport(Kind kind, Summary statistics) {}

  record DecileReport(Kind kind, int decile, Summary statistics) {}

  record Summary(
      long selections,
      long nonForcedSelections,
      long calibratedSelections,
      long kindCalibrations,
      long turnCalibrations,
      long responseCalibrations,
      long globalCalibrations,
      long uncalibratedSelections,
      long nonTopOneSelections,
      long forcedRoles,
      long causalRoles,
      long preemptedRoles,
      double meanNormalizedEntropy,
      double meanTopTwoAmbiguity,
      double meanUncertainty,
      double meanPercentile,
      double meanMultiplier,
      double meanTotalVariation,
      double meanBehaviorToRolloutKl,
      double meanSelectedRolloutProbability,
      double meanSelectedBehaviorProbability,
      double meanSelectedRolloutToBehaviorRatio,
      double meanSelectedRolloutRank,
      double meanTopOneMassChange) {}

  static Measurement measure(float[] policy) {
    Objects.requireNonNull(policy, "policy");
    if (policy.length < 2) {
      throw new IllegalArgumentException("uncertainty requires at least two legal actions");
    }
    double sum = 0.0;
    for (float probability : policy) {
      if (!Float.isFinite(probability) || probability < 0.0f) {
        throw new IllegalArgumentException("policy contains an invalid probability");
      }
      sum += probability;
    }
    if (!(sum > 0.0) || !Double.isFinite(sum)) {
      throw new IllegalArgumentException("policy probability sum must be finite and positive");
    }
    double entropy = 0.0;
    double first = -1.0;
    double second = -1.0;
    for (float raw : policy) {
      double probability = raw / sum;
      if (probability > 0.0) {
        entropy -= probability * Math.log(probability);
      }
      if (probability > first) {
        second = first;
        first = probability;
      } else if (probability > second) {
        second = probability;
      }
    }
    float normalizedEntropy = clamp01((float) (entropy / Math.log(policy.length)));
    float topTwoAmbiguity = clamp01((float) (2.0 * second / (first + second + 1.0e-8)));
    float uncertainty = clamp01(0.5f * normalizedEntropy + 0.5f * topTwoAmbiguity);
    return new Measurement(normalizedEntropy, topTwoAmbiguity, uncertainty);
  }

  static Measurement measure(float[] policy, float entropyWeight) {
    Measurement base = measure(policy);
    float uncertainty =
        clamp01(
            entropyWeight * base.normalizedEntropy()
                + (1.0f - entropyWeight) * base.topTwoAmbiguity());
    return new Measurement(base.normalizedEntropy(), base.topTwoAmbiguity(), uncertainty);
  }

  private static Measurement measure(float[] policy, AdaptiveExplorationSettings settings) {
    return measure(policy, settings.entropyWeight());
  }

  static float multiplier(float percentile, AdaptiveExplorationSettings settings) {
    if (!Float.isFinite(percentile) || percentile < 0.0f || percentile > 1.0f) {
      throw new IllegalArgumentException("percentile must be finite and in [0, 1]");
    }
    double curved = Math.pow(percentile, settings.percentileExponent());
    return (float)
        (settings.minimumScale() + (settings.maximumScale() - settings.minimumScale()) * curved);
  }

  static Kind classify(List<Action> legalActions) {
    Objects.requireNonNull(legalActions, "legalActions");
    if (legalActions.isEmpty()) {
      throw new IllegalArgumentException("legalActions must not be empty");
    }
    if (legalActions.size() == 1) {
      return Kind.FORCED;
    }
    boolean call = false;
    boolean kan = false;
    boolean riichi = false;
    for (Action action : legalActions) {
      switch (action.type()) {
        case RON_AGARI, TSUMO_AGARI, KYUSHU_KYUHAI -> {
          return Kind.AGARI_OR_ABORT;
        }
        case CHI, PON, DAIMINKAN -> call = true;
        case ANKAN, KAKAN -> kan = true;
        case RIICHI_DAHAI -> riichi = true;
        case DAHAI, PASS -> {
          // 優先度の低い基本行動。
        }
      }
    }
    if (call) {
      return Kind.CALL;
    }
    if (kan) {
      return Kind.KAN;
    }
    if (riichi) {
      return Kind.RIICHI;
    }
    return Kind.DISCARD;
  }

  private static boolean isResponse(List<Action> legalActions) {
    boolean response = legalActions.getFirst().type().isResponse();
    for (int slot = 1; slot < legalActions.size(); slot++) {
      if (legalActions.get(slot).type().isResponse() != response) {
        throw new IllegalArgumentException("turn and response actions cannot share a policy");
      }
    }
    return response;
  }

  private static float clamp01(float value) {
    return Math.max(0.0f, Math.min(1.0f, value));
  }

  private enum HistogramKey {
    AGARI_OR_ABORT,
    CALL,
    KAN,
    RIICHI,
    DISCARD,
    TURN,
    RESPONSE,
    GLOBAL
  }

  private static final class Histogram {
    private final long[][] counts = new long[HistogramKey.values().length][HISTOGRAM_BINS];

    private void observe(Kind kind, boolean response, float uncertainty) {
      increment(key(kind), uncertainty);
      increment(response ? HistogramKey.RESPONSE : HistogramKey.TURN, uncertainty);
      increment(HistogramKey.GLOBAL, uncertainty);
    }

    private void increment(HistogramKey key, float uncertainty) {
      counts[key.ordinal()][bin(uncertainty)]++;
    }

    private void merge(Histogram other) {
      for (int key = 0; key < counts.length; key++) {
        for (int bin = 0; bin < HISTOGRAM_BINS; bin++) {
          counts[key][bin] = Math.addExact(counts[key][bin], other.counts[key][bin]);
        }
      }
    }
  }

  private static final class Calibration {
    private final long sourceSnapshotId;
    private final long[][] cumulative;
    private final long[] totals;
    private final String hash;

    private Calibration(long sourceSnapshotId, long[][] cumulative, long[] totals, String hash) {
      this.sourceSnapshotId = sourceSnapshotId;
      this.cumulative = cumulative;
      this.totals = totals;
      this.hash = hash;
    }

    private static Calibration empty() {
      return new Calibration(-1L, new long[HistogramKey.values().length][0], new long[0], null);
    }

    private static Calibration from(long sourceSnapshotId, Histogram histogram) {
      long[][] cumulative = new long[HistogramKey.values().length][HISTOGRAM_BINS];
      long[] totals = new long[HistogramKey.values().length];
      for (HistogramKey key : HistogramKey.values()) {
        long running = 0L;
        for (int bin = 0; bin < HISTOGRAM_BINS; bin++) {
          running = Math.addExact(running, histogram.counts[key.ordinal()][bin]);
          cumulative[key.ordinal()][bin] = running;
        }
        totals[key.ordinal()] = running;
      }
      return new Calibration(
          sourceSnapshotId, cumulative, totals, DecisionAdaptiveExploration.hash(histogram));
    }

    private Lookup lookup(Kind kind, boolean response, float uncertainty, double tieFraction) {
      if (isEmpty()) {
        return Lookup.none();
      }
      Lookup exact = lookup(key(kind), LookupSource.KIND, uncertainty, tieFraction);
      if (exact.available()) {
        return exact;
      }
      HistogramKey family = response ? HistogramKey.RESPONSE : HistogramKey.TURN;
      Lookup familyLookup =
          lookup(
              family,
              response ? LookupSource.RESPONSE : LookupSource.TURN,
              uncertainty,
              tieFraction);
      if (familyLookup.available()) {
        return familyLookup;
      }
      return lookup(HistogramKey.GLOBAL, LookupSource.GLOBAL, uncertainty, tieFraction);
    }

    private Lookup lookup(
        HistogramKey key, LookupSource source, float uncertainty, double tieFraction) {
      long total = totals[key.ordinal()];
      if (total < MINIMUM_CALIBRATION_SAMPLES) {
        return Lookup.none();
      }
      int bin = bin(uncertainty);
      long atOrBelow = cumulative[key.ordinal()][bin];
      long below = bin == 0 ? 0L : cumulative[key.ordinal()][bin - 1];
      float percentile = (float) ((below + tieFraction * (atOrBelow - below)) / total);
      return new Lookup(clamp01(percentile), source);
    }

    private boolean isEmpty() {
      return sourceSnapshotId < 0L;
    }

    private long sourceSnapshotId() {
      return sourceSnapshotId;
    }

    private String hash() {
      return hash;
    }

    private long samples() {
      return isEmpty() ? 0L : totals[HistogramKey.GLOBAL.ordinal()];
    }
  }

  private static final class LocalAccumulator {
    private final Histogram histogram = new Histogram();
    private final MutableStats overall = new MutableStats();
    private final MutableStats[] byKind = statsArray(Kind.values().length);
    private final MutableStats[][] byKindAndDecile = decileStats();

    private void selection(
        Observation observation, float[] rolloutPolicy, float[] behaviorPolicy, int selectedSlot) {
      if (rolloutPolicy.length != behaviorPolicy.length
          || selectedSlot < 0
          || selectedSlot >= rolloutPolicy.length) {
        throw new IllegalArgumentException("exploration telemetry policy shape mismatch");
      }
      double totalVariation = 0.0;
      double behaviorToRolloutKl = 0.0;
      float selectedRollout = rolloutPolicy[selectedSlot];
      float selectedBehavior = behaviorPolicy[selectedSlot];
      float rolloutTop = 0.0f;
      int rolloutTopSlot = 0;
      int rank = 1;
      if (observation.measurement() != null) {
        for (int slot = 0; slot < rolloutPolicy.length; slot++) {
          double rollout = Math.max(PROBABILITY_EPSILON, rolloutPolicy[slot]);
          double behavior = Math.max(PROBABILITY_EPSILON, behaviorPolicy[slot]);
          totalVariation += Math.abs(behavior - rollout);
          behaviorToRolloutKl += behavior * Math.log(behavior / rollout);
          if (rolloutPolicy[slot] > rolloutTop) {
            rolloutTop = rolloutPolicy[slot];
            rolloutTopSlot = slot;
          }
          if (rolloutPolicy[slot] > selectedRollout) {
            rank++;
          }
        }
      }
      // 同じ分布の診断値は一度だけ計算し、全体・種別・十分位へ同じ順序で加算する。
      double halfTotalVariation = 0.5 * totalVariation;
      float selectedRatio = selectedRollout / selectedBehavior;
      float topOneMassChange = behaviorPolicy[rolloutTopSlot] - rolloutTop;
      overall.selection(
          observation,
          halfTotalVariation,
          behaviorToRolloutKl,
          selectedRollout,
          selectedBehavior,
          selectedRatio,
          rank,
          topOneMassChange);
      byKind[observation.kind().ordinal()].selection(
          observation,
          halfTotalVariation,
          behaviorToRolloutKl,
          selectedRollout,
          selectedBehavior,
          selectedRatio,
          rank,
          topOneMassChange);
      if (observation.decile() >= 0) {
        byKindAndDecile[observation.kind().ordinal()][observation.decile()].selection(
            observation,
            halfTotalVariation,
            behaviorToRolloutKl,
            selectedRollout,
            selectedBehavior,
            selectedRatio,
            rank,
            topOneMassChange);
      }
    }

    private void learningRole(Kind kind, int decile, DecisionLearningRole role) {
      overall.learningRole(role);
      byKind[kind.ordinal()].learningRole(role);
      if (decile >= 0) {
        byKindAndDecile[kind.ordinal()][decile].learningRole(role);
      }
    }
  }

  private static final class MacroAggregate {
    private final Histogram histogram = new Histogram();
    private final MutableStats overall = new MutableStats();
    private final MutableStats[] byKind = statsArray(Kind.values().length);
    private final MutableStats[][] byKindAndDecile = decileStats();

    private void merge(LocalAccumulator accumulator) {
      histogram.merge(accumulator.histogram);
      overall.merge(accumulator.overall);
      for (Kind kind : Kind.values()) {
        byKind[kind.ordinal()].merge(accumulator.byKind[kind.ordinal()]);
        for (int decile = 0; decile < 10; decile++) {
          byKindAndDecile[kind.ordinal()][decile].merge(
              accumulator.byKindAndDecile[kind.ordinal()][decile]);
        }
      }
    }

    private Histogram histogram() {
      return histogram;
    }

    private Report toReport(
        AdaptiveExplorationSettings settings,
        Long sourceSnapshotId,
        String sourceHash,
        long sourceSamples,
        long collectedSnapshotId,
        String collectedHash,
        long collectedSamples,
        double unweightedConfiguredNodeMassMean) {
      ArrayList<KindReport> kindReports = new ArrayList<>(Kind.values().length);
      ArrayList<DecileReport> decileReports = new ArrayList<>();
      for (Kind kind : Kind.values()) {
        kindReports.add(new KindReport(kind, byKind[kind.ordinal()].snapshot()));
        for (int decile = 0; decile < 10; decile++) {
          MutableStats stats = byKindAndDecile[kind.ordinal()][decile];
          if (stats.selections > 0L || stats.roles > 0L) {
            decileReports.add(new DecileReport(kind, decile, stats.snapshot()));
          }
        }
      }
      Summary overallSummary = overall.snapshot();
      return new Report(
          settings.schedule().name(),
          HISTOGRAM_BINS,
          MINIMUM_CALIBRATION_SAMPLES,
          sourceSnapshotId,
          sourceHash,
          sourceSamples,
          collectedSnapshotId,
          collectedHash,
          collectedSamples,
          unweightedConfiguredNodeMassMean,
          unweightedConfiguredNodeMassMean * overallSummary.meanMultiplier(),
          overallSummary,
          List.copyOf(kindReports),
          List.copyOf(decileReports));
    }
  }

  private static final class MutableStats {
    private long selections;
    private long measuredSelections;
    private long calibratedSelections;
    private long kindCalibrations;
    private long turnCalibrations;
    private long responseCalibrations;
    private long globalCalibrations;
    private long uncalibratedSelections;
    private long nonTopOneSelections;
    private long roles;
    private long forcedRoles;
    private long causalRoles;
    private long preemptedRoles;
    private double entropySum;
    private double ambiguitySum;
    private double uncertaintySum;
    private double percentileSum;
    private double multiplierSum;
    private double totalVariationSum;
    private double behaviorToRolloutKlSum;
    private double selectedRolloutProbabilitySum;
    private double selectedBehaviorProbabilitySum;
    private double selectedRolloutToBehaviorRatioSum;
    private double selectedRolloutRankSum;
    private double topOneMassChangeSum;

    private void selection(
        Observation observation,
        double totalVariation,
        double behaviorToRolloutKl,
        float selectedRollout,
        float selectedBehavior,
        float selectedRatio,
        int rank,
        float topOneMassChange) {
      selections++;
      if (observation.measurement() != null) {
        measuredSelections++;
        multiplierSum += observation.multiplier();
        entropySum += observation.measurement().normalizedEntropy();
        ambiguitySum += observation.measurement().topTwoAmbiguity();
        uncertaintySum += observation.measurement().uncertainty();
        if (observation.lookup().available()) {
          calibratedSelections++;
          percentileSum += observation.lookup().percentile();
          switch (observation.lookup().source()) {
            case KIND -> kindCalibrations++;
            case TURN -> turnCalibrations++;
            case RESPONSE -> responseCalibrations++;
            case GLOBAL -> globalCalibrations++;
            case NONE -> throw new AssertionError("available lookup cannot have NONE source");
          }
        } else {
          uncalibratedSelections++;
        }
      } else {
        return;
      }
      totalVariationSum += totalVariation;
      behaviorToRolloutKlSum += behaviorToRolloutKl;
      selectedRolloutProbabilitySum += selectedRollout;
      selectedBehaviorProbabilitySum += selectedBehavior;
      selectedRolloutToBehaviorRatioSum += selectedRatio;
      selectedRolloutRankSum += rank;
      if (rank > 1) {
        nonTopOneSelections++;
      }
      topOneMassChangeSum += topOneMassChange;
    }

    private void learningRole(DecisionLearningRole role) {
      roles++;
      switch (role) {
        case FORCED -> forcedRoles++;
        case CAUSAL -> causalRoles++;
        case PREEMPTED -> preemptedRoles++;
      }
    }

    private void merge(MutableStats other) {
      selections = Math.addExact(selections, other.selections);
      measuredSelections = Math.addExact(measuredSelections, other.measuredSelections);
      calibratedSelections = Math.addExact(calibratedSelections, other.calibratedSelections);
      kindCalibrations = Math.addExact(kindCalibrations, other.kindCalibrations);
      turnCalibrations = Math.addExact(turnCalibrations, other.turnCalibrations);
      responseCalibrations = Math.addExact(responseCalibrations, other.responseCalibrations);
      globalCalibrations = Math.addExact(globalCalibrations, other.globalCalibrations);
      uncalibratedSelections = Math.addExact(uncalibratedSelections, other.uncalibratedSelections);
      nonTopOneSelections = Math.addExact(nonTopOneSelections, other.nonTopOneSelections);
      roles = Math.addExact(roles, other.roles);
      forcedRoles = Math.addExact(forcedRoles, other.forcedRoles);
      causalRoles = Math.addExact(causalRoles, other.causalRoles);
      preemptedRoles = Math.addExact(preemptedRoles, other.preemptedRoles);
      entropySum += other.entropySum;
      ambiguitySum += other.ambiguitySum;
      uncertaintySum += other.uncertaintySum;
      percentileSum += other.percentileSum;
      multiplierSum += other.multiplierSum;
      totalVariationSum += other.totalVariationSum;
      behaviorToRolloutKlSum += other.behaviorToRolloutKlSum;
      selectedRolloutProbabilitySum += other.selectedRolloutProbabilitySum;
      selectedBehaviorProbabilitySum += other.selectedBehaviorProbabilitySum;
      selectedRolloutToBehaviorRatioSum += other.selectedRolloutToBehaviorRatioSum;
      selectedRolloutRankSum += other.selectedRolloutRankSum;
      topOneMassChangeSum += other.topOneMassChangeSum;
    }

    private Summary snapshot() {
      return new Summary(
          selections,
          measuredSelections,
          calibratedSelections,
          kindCalibrations,
          turnCalibrations,
          responseCalibrations,
          globalCalibrations,
          uncalibratedSelections,
          nonTopOneSelections,
          forcedRoles,
          causalRoles,
          preemptedRoles,
          mean(entropySum, measuredSelections),
          mean(ambiguitySum, measuredSelections),
          mean(uncertaintySum, measuredSelections),
          mean(percentileSum, calibratedSelections),
          mean(multiplierSum, measuredSelections),
          mean(totalVariationSum, measuredSelections),
          mean(behaviorToRolloutKlSum, measuredSelections),
          mean(selectedRolloutProbabilitySum, measuredSelections),
          mean(selectedBehaviorProbabilitySum, measuredSelections),
          mean(selectedRolloutToBehaviorRatioSum, measuredSelections),
          mean(selectedRolloutRankSum, measuredSelections),
          mean(topOneMassChangeSum, measuredSelections));
    }
  }

  private static MutableStats[] statsArray(int size) {
    MutableStats[] result = new MutableStats[size];
    for (int index = 0; index < size; index++) {
      result[index] = new MutableStats();
    }
    return result;
  }

  private static MutableStats[][] decileStats() {
    MutableStats[][] result = new MutableStats[Kind.values().length][10];
    for (int kind = 0; kind < result.length; kind++) {
      result[kind] = statsArray(10);
    }
    return result;
  }

  private static HistogramKey key(Kind kind) {
    return switch (kind) {
      case AGARI_OR_ABORT -> HistogramKey.AGARI_OR_ABORT;
      case CALL -> HistogramKey.CALL;
      case KAN -> HistogramKey.KAN;
      case RIICHI -> HistogramKey.RIICHI;
      case DISCARD -> HistogramKey.DISCARD;
      case FORCED -> throw new IllegalArgumentException("forced decisions do not enter the CDF");
    };
  }

  private static int bin(float uncertainty) {
    if (!Float.isFinite(uncertainty) || uncertainty < 0.0f || uncertainty > 1.0f) {
      throw new IllegalArgumentException("uncertainty must be finite and in [0, 1]");
    }
    return Math.min(HISTOGRAM_BINS - 1, (int) (uncertainty * HISTOGRAM_BINS));
  }

  private static String hash(Histogram histogram) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
      for (HistogramKey key : HistogramKey.values()) {
        for (long count : histogram.counts[key.ordinal()]) {
          buffer.clear();
          buffer.putLong(count);
          digest.update(buffer.array());
        }
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError("SHA-256 is required by the Java platform", impossible);
    }
  }

  private static double mean(double sum, long count) {
    return count == 0L ? 0.0 : sum / count;
  }
}
