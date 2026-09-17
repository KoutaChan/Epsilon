package com.epsilon.ai.decision.duel;

import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.runtime.InferenceAdmission;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 事前に決めた牌山数と判定時点への有意水準の配分に従い、候補モデルを採用・不採用・判定保留に分類する。 */
public final class EpsilonDecisionWallDuelEvaluator {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionWallDuelEvaluator.class);
  private static final double RANK_DELTA_LOWER_BOUND = -2.0;
  private static final double RANK_DELTA_UPPER_BOUND = 2.0;
  private static final double LCB_95_Z = 1.959963984540054;
  private static final double BOUNDARY_COMPARISON_ULPS = 8.0;

  private EpsilonDecisionWallDuelEvaluator() {}

  /** 事前計画した牌山区間を実行し、各判定時点の信頼区間から判定する。 */
  public static Result evaluate(
      DuelEvaluationSource source,
      EvaluationPlan plan,
      long seedBase,
      long duelSequence,
      double promotionMargin,
      double harmfulMargin,
      EpsilonUtilityProfile objectiveProfile,
      DecisionEvalVsSettings settings) {
    UtilityBounds objectiveBounds = utilityBounds(objectiveProfile);
    validate(duelSequence, promotionMargin, harmfulMargin, objectiveBounds);
    EpsilonDecisionFixedSampleConfidenceInterval utilityConfidenceInterval =
        new EpsilonDecisionFixedSampleConfidenceInterval(
            objectiveBounds.lower(), objectiveBounds.upper());
    EpsilonDecisionFixedSampleConfidenceInterval rankConfidenceInterval =
        new EpsilonDecisionFixedSampleConfidenceInterval(
            RANK_DELTA_LOWER_BOUND, RANK_DELTA_UPPER_BOUND);
    ArrayList<DuelEvaluation> segments = new ArrayList<>();
    int[] progressUpdates = {0};
    Decision decision = Decision.UNRESOLVED;
    double decisionAlpha = Double.NaN;
    int completedLooks = 0;

    long duelSeed = EpsilonDecisionSeeds.promotionAttempt(seedBase, duelSequence);
    for (int lookIndex = 0; lookIndex < plan.looks().size(); lookIndex++) {
      Look look = plan.looks().get(lookIndex);
      int segmentStartWalls = utilityConfidenceInterval.samples();
      int segmentWalls = look.cumulativeWallSeeds() - segmentStartWalls;
      int lookNumber = lookIndex + 1;
      DuelEvaluation segment =
          source.evaluateWalls(
              segmentWalls,
              duelSeed,
              segmentStartWalls,
              settings.gamesInFlight(),
              outcome -> {
                utilityConfidenceInterval.add(outcome.pairedConfiguredUtilityDelta());
                rankConfidenceInterval.add(outcome.pairedRankDelta());
                int completedWalls = utilityConfidenceInterval.samples();
                boolean lookBoundary = completedWalls == look.cumulativeWallSeeds();
                if (!lookBoundary && completedWalls % settings.progressIntervalWallSeeds() != 0) {
                  return;
                }
                progressUpdates[0]++;
                reportProgress(
                    duelSequence,
                    plan.mode(),
                    progressUpdates[0],
                    lookNumber,
                    plan.looks().size(),
                    completedWalls,
                    plan.maximumWallSeeds(),
                    look.alpha(),
                    lookBoundary,
                    objectiveProfile,
                    utilityConfidenceInterval.estimate(look.alpha()));
              });
      segments.add(segment);
      if (utilityConfidenceInterval.samples() != look.cumulativeWallSeeds()) {
        throw new IllegalStateException(
            "duel look completed "
                + utilityConfidenceInterval.samples()
                + " of "
                + look.cumulativeWallSeeds()
                + " walls");
      }

      completedLooks = lookNumber;
      decisionAlpha = look.alpha();
      EpsilonDecisionFixedSampleConfidenceInterval.Estimate lookEstimate =
          utilityConfidenceInterval.estimate(decisionAlpha);
      decision = decide(lookEstimate, promotionMargin, harmfulMargin);
      log.info(
          "Decision champion duel look complete: sequence={} mode={} look={}/{} "
              + "wallSeeds={}/{} utilityProfile={} alpha={} mean={} lower={} upper={} "
              + "decision={}",
          duelSequence,
          plan.mode(),
          lookNumber,
          plan.looks().size(),
          utilityConfidenceInterval.samples(),
          plan.maximumWallSeeds(),
          objectiveProfile,
          decisionAlpha,
          lookEstimate.mean(),
          lookEstimate.lower(),
          lookEstimate.upper(),
          decision);
      if (decision != Decision.UNRESOLVED || lookNumber == plan.looks().size()) {
        break;
      }
    }

    EpsilonDecisionFixedSampleConfidenceInterval.Estimate utilityEstimate =
        utilityConfidenceInterval.estimate(decisionAlpha);
    EpsilonDecisionFixedSampleConfidenceInterval.Estimate rankEstimate =
        rankConfidenceInterval.estimate(decisionAlpha);
    DuelEvaluation.Result descriptiveResult = combineResults(segments, rankEstimate);
    DuelEvaluation.Metrics metrics = combineMetrics(segments);
    log.info(
        "Decision champion duel complete: sequence={} mode={} looks={} wallSeeds={}/{} "
            + "games={} gamesInFlight={} progressUpdates={} inferenceBatches={} "
            + "avgInferenceBatch={} totalAlpha={} decisionAlpha={} mean={} lower={} upper={} "
            + "utilityProfile={} promotionMargin={} harmfulMargin={} decision={}",
        duelSequence,
        plan.mode(),
        completedLooks,
        utilityConfidenceInterval.samples(),
        plan.maximumWallSeeds(),
        descriptiveResult.games(),
        metrics.gamesInFlight(),
        progressUpdates[0],
        metrics.inferenceBatches(),
        metrics.averageInferenceBatch(),
        plan.totalAlpha(),
        decisionAlpha,
        utilityEstimate.mean(),
        utilityEstimate.lower(),
        utilityEstimate.upper(),
        objectiveProfile,
        promotionMargin,
        harmfulMargin,
        decision);
    return new Result(
        source.candidateCheckpoint(),
        source.opponentCheckpoint(),
        descriptiveResult.games(),
        utilityConfidenceInterval.samples(),
        plan.maximumWallSeeds(),
        progressUpdates[0],
        completedLooks,
        plan.mode(),
        duelSequence,
        plan.totalAlpha(),
        decisionAlpha,
        objectiveProfile,
        utilityEstimate.mean(),
        utilityEstimate.lower(),
        utilityEstimate.upper(),
        rankEstimate.mean(),
        rankEstimate.lower(),
        rankEstimate.upper(),
        promotionMargin,
        harmfulMargin,
        decision,
        descriptiveResult);
  }

  private static DuelEvaluation.Result combineResults(
      List<DuelEvaluation> segments,
      EpsilonDecisionFixedSampleConfidenceInterval.Estimate estimate) {
    if (segments.isEmpty()) {
      throw new IllegalStateException("duel produced no evaluation segments");
    }
    DuelEvaluation.Result first = segments.get(0).result();
    int games = 0;
    double scoreAdvantage = 0.0;
    double candidateAverageRank = 0.0;
    double opponentAverageRank = 0.0;
    double candidateTopRate = 0.0;
    double opponentTopRate = 0.0;
    double candidateLastRate = 0.0;
    double opponentLastRate = 0.0;
    double[] utilityProfileAdvantages = new double[first.utilityProfileAdvantages().length];
    for (DuelEvaluation segment : segments) {
      DuelEvaluation.Result result = segment.result();
      int segmentGames = result.games();
      games = Math.addExact(games, segmentGames);
      scoreAdvantage += result.scoreAdvantage() * segmentGames;
      candidateAverageRank += result.candidateAverageRank() * segmentGames;
      opponentAverageRank += result.opponentAverageRank() * segmentGames;
      candidateTopRate += result.candidateTopRate() * segmentGames;
      opponentTopRate += result.opponentTopRate() * segmentGames;
      candidateLastRate += result.candidateLastRate() * segmentGames;
      opponentLastRate += result.opponentLastRate() * segmentGames;
      for (int profile = 0; profile < utilityProfileAdvantages.length; profile++) {
        utilityProfileAdvantages[profile] +=
            result.utilityProfileAdvantages()[profile] * segmentGames;
      }
    }
    for (int profile = 0; profile < utilityProfileAdvantages.length; profile++) {
      utilityProfileAdvantages[profile] /= games;
    }
    return new DuelEvaluation.Result(
        first.candidateCheckpoint(),
        first.opponentCheckpoint(),
        games,
        estimate.samples(),
        estimate.mean(),
        estimate.standardError(),
        estimate.mean() - LCB_95_Z * estimate.standardError(),
        scoreAdvantage / games,
        candidateAverageRank / games,
        opponentAverageRank / games,
        candidateTopRate / games,
        opponentTopRate / games,
        candidateLastRate / games,
        opponentLastRate / games,
        utilityProfileAdvantages);
  }

  private static DuelEvaluation.Metrics combineMetrics(List<DuelEvaluation> segments) {
    int games = 0;
    int gamesInFlight = 0;
    int completedGames = 0;
    long inferenceBatches = 0L;
    long inferenceRequests = 0L;
    int maxInferenceBatch = 0;
    InferenceAdmission.Metrics batching = InferenceAdmission.Metrics.empty();
    for (DuelEvaluation segment : segments) {
      DuelEvaluation.Metrics metrics = segment.metrics();
      games = Math.addExact(games, metrics.games());
      gamesInFlight = Math.max(gamesInFlight, metrics.gamesInFlight());
      completedGames = Math.addExact(completedGames, metrics.completedGames());
      inferenceBatches = Math.addExact(inferenceBatches, metrics.inferenceBatches());
      inferenceRequests = Math.addExact(inferenceRequests, metrics.inferenceRequests());
      maxInferenceBatch = Math.max(maxInferenceBatch, metrics.maxInferenceBatch());
      batching = batching.plus(metrics.batching());
    }
    double averageInferenceBatch =
        inferenceBatches == 0L ? 0.0 : inferenceRequests / (double) inferenceBatches;
    return new DuelEvaluation.Metrics(
        games,
        gamesInFlight,
        completedGames,
        inferenceBatches,
        inferenceRequests,
        averageInferenceBatch,
        maxInferenceBatch,
        batching);
  }

  private static void reportProgress(
      long duelSequence,
      Mode mode,
      int progressUpdate,
      int look,
      int plannedLooks,
      int completedWalls,
      int maximumWallSeeds,
      double alpha,
      boolean decisionEligible,
      EpsilonUtilityProfile objectiveProfile,
      EpsilonDecisionFixedSampleConfidenceInterval.Estimate estimate) {
    log.info(
        "Decision champion duel progress: sequence={} mode={} progress={} look={}/{} "
            + "wallSeeds={}/{} alpha={} decisionEligible={} utilityProfile={} "
            + "mean={} lower={} upper={}",
        duelSequence,
        mode,
        progressUpdate,
        look,
        plannedLooks,
        completedWalls,
        maximumWallSeeds,
        alpha,
        decisionEligible,
        objectiveProfile,
        estimate.mean(),
        estimate.lower(),
        estimate.upper());
    log.info(
        "V20_FIXED_DUEL_STAGE"
            + " stage="
            + progressUpdate
            + " wallSeeds="
            + completedWalls
            + "/"
            + maximumWallSeeds
            + " mean="
            + estimate.mean()
            + " lower="
            + estimate.lower()
            + " upper="
            + estimate.upper()
            + " look="
            + look
            + "/"
            + plannedLooks
            + " decisionEligible="
            + decisionEligible
            + " mode="
            + mode
            + " utilityProfile="
            + objectiveProfile);
  }

  static Decision decide(
      EpsilonDecisionFixedSampleConfidenceInterval.Estimate estimate,
      double promotionMargin,
      double harmfulMargin) {
    if (strictlyGreaterThanBoundary(estimate.lower(), promotionMargin)) {
      return Decision.PROMOTED;
    }
    return strictlyGreaterThanBoundary(harmfulMargin, estimate.upper())
        ? Decision.HARMFUL
        : Decision.UNRESOLVED;
  }

  /** 浮動小数の丸めだけを同値として扱い、統計的な判定幅は導入しない。 */
  private static boolean strictlyGreaterThanBoundary(double value, double boundary) {
    double scale = Math.max(1.0, Math.max(Math.abs(value), Math.abs(boundary)));
    return value - boundary > BOUNDARY_COMPARISON_ULPS * Math.ulp(scale);
  }

  private static void validate(
      long duelSequence, double promotionMargin, double harmfulMargin, UtilityBounds bounds) {
    if (duelSequence <= 0L) {
      throw new IllegalArgumentException("duelSequence must be positive");
    }
    if (!Double.isFinite(promotionMargin)
        || !Double.isFinite(harmfulMargin)
        || promotionMargin < bounds.lower()
        || promotionMargin >= bounds.upper()
        || harmfulMargin <= bounds.lower()
        || harmfulMargin > bounds.upper()
        || harmfulMargin > promotionMargin) {
      throw new IllegalArgumentException(
          "duel margins must be inside the configured utility range");
    }
  }

  public static UtilityBounds utilityBounds(EpsilonUtilityProfile profile) {
    if (!profile.rankBased()) {
      throw new IllegalArgumentException("champion duel requires a rank utility profile");
    }
    double minimum = profile.utilityForRank(0);
    double maximum = minimum;
    for (int rank = 1; rank < 4; rank++) {
      double utility = profile.utilityForRank(rank);
      minimum = Math.min(minimum, utility);
      maximum = Math.max(maximum, utility);
    }
    double range = maximum - minimum;
    return new UtilityBounds(-range, range);
  }

  /** 対戦評価に使用する牌山数と判定時点の計画。 */
  public enum Mode {
    FIXED_HORIZON,
    TWO_LOOK_ALPHA_SPENDING
  }

  /** 固定標本数の信頼区間を使った、比較対局によるモデルの採用判定。 */
  public enum Decision {
    PROMOTED,
    HARMFUL,
    UNRESOLVED
  }

  public record Look(int cumulativeWallSeeds, double alpha) {}

  public record UtilityBounds(double lower, double upper) {}

  public record EvaluationPlan(
      Mode mode, int maximumWallSeeds, double totalAlpha, List<Look> looks) {
    public EvaluationPlan {
      looks = List.copyOf(looks);
      if (maximumWallSeeds <= 0 || looks.isEmpty()) {
        throw new IllegalArgumentException(
            "duel plan requires positive walls and at least one look");
      }
      if (!Double.isFinite(totalAlpha) || totalAlpha <= 0.0 || totalAlpha >= 1.0) {
        throw new IllegalArgumentException("duel total alpha must be in (0, 1)");
      }
      int previousWalls = 0;
      double allocatedAlpha = 0.0;
      for (Look look : looks) {
        if (look.cumulativeWallSeeds() <= previousWalls
            || look.cumulativeWallSeeds() > maximumWallSeeds) {
          throw new IllegalArgumentException("duel looks must be increasing and within maximum");
        }
        if (!Double.isFinite(look.alpha()) || look.alpha() <= 0.0 || look.alpha() >= 1.0) {
          throw new IllegalArgumentException("duel look alpha must be in (0, 1)");
        }
        previousWalls = look.cumulativeWallSeeds();
        allocatedAlpha += look.alpha();
      }
      if (previousWalls != maximumWallSeeds || allocatedAlpha > totalAlpha + 1.0e-15) {
        throw new IllegalArgumentException(
            "duel plan must end at maximum walls without exceeding total alpha");
      }
    }

    public static EvaluationPlan fixed(int wallSeeds, double alpha) {
      return new EvaluationPlan(
          Mode.FIXED_HORIZON, wallSeeds, alpha, List.of(new Look(wallSeeds, alpha)));
    }

    public static EvaluationPlan twoLook(
        int interimWallSeeds, int maximumWallSeeds, double alpha, double interimAlphaFraction) {
      if (interimWallSeeds <= 0 || interimWallSeeds >= maximumWallSeeds) {
        throw new IllegalArgumentException(
            "two-look wall seeds must satisfy 0 < interim < maximum");
      }
      if (!Double.isFinite(interimAlphaFraction)
          || interimAlphaFraction <= 0.0
          || interimAlphaFraction >= 1.0) {
        throw new IllegalArgumentException("interim alpha fraction must be in (0, 1)");
      }
      double interimAlpha = alpha * interimAlphaFraction;
      return new EvaluationPlan(
          Mode.TWO_LOOK_ALPHA_SPENDING,
          maximumWallSeeds,
          alpha,
          List.of(
              new Look(interimWallSeeds, interimAlpha),
              new Look(maximumWallSeeds, alpha - interimAlpha)));
    }
  }

  /**
   * @param wallSeeds 実際に消費した牌山数
   * @param plannedWallSeeds 事前計画した最大牌山数
   * @param progressUpdates 対戦環境を止めずに発行した進捗通知数
   * @param completedLooks 実行した統計判定時点数
   * @param alpha 比較対局全体へ配分した両側誤差率
   * @param decisionAlpha 最終判定を行った判定時点の両側誤差率
   * @param utilityProfile 採用モデル判定に用いた効用値設定
   * @param pairedUtilityDeltaMean 対応する対局間の効用値の差の標本平均
   * @param pairedUtilityDeltaLower 対応する対局間の効用値の差の信頼区間下限
   * @param pairedUtilityDeltaUpper 対応する対局間の効用値の差の信頼区間上限
   * @param pairedRankDeltaMean 診断用に集計する、対応する対局間の平均順位差の標本平均
   * @param pairedRankDeltaLower 診断用に集計する、対応する対局間の平均順位差の信頼区間下限
   * @param pairedRankDeltaUpper 診断用に集計する、対応する対局間の平均順位差の信頼区間上限
   */
  public record Result(
      Path candidateCheckpoint,
      Path parentCheckpoint,
      int games,
      int wallSeeds,
      int plannedWallSeeds,
      int progressUpdates,
      int completedLooks,
      Mode mode,
      long duelSequence,
      double alpha,
      double decisionAlpha,
      EpsilonUtilityProfile utilityProfile,
      double pairedUtilityDeltaMean,
      double pairedUtilityDeltaLower,
      double pairedUtilityDeltaUpper,
      double pairedRankDeltaMean,
      double pairedRankDeltaLower,
      double pairedRankDeltaUpper,
      double promotionMargin,
      double harmfulMargin,
      Decision decision,
      DuelEvaluation.Result descriptiveResult) {}
}
