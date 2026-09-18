package com.epsilon.ai.decision.duel;

import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.ai.decision.EpsilonDecisionSeeds;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.spi.BatchedPolicy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 同じ牌山で候補モデルの席を4席に入れ替えて対戦し、席順の差を抑えた評価指標を集計する。 */
public final class EpsilonDecisionDuelArena {

  private static final Logger LOG = LoggerFactory.getLogger(EpsilonDecisionDuelArena.class);

  private static final int SEAT_ROTATIONS = GameState.NUM_PLAYERS;
  private static final double LCB_95_Z = 1.96;

  private EpsilonDecisionDuelArena() {}

  /**
   * {@code firstWallFamilyId}から始まる新しく生成した牌山範囲で評価する。
   *
   * <p>段階評価間で同じ乱数シードを再利用しないためのオフセット付き境界である。
   *
   * @param candidateCheckpoint 候補識別情報として記録するチェックポイントパス
   * @param opponentCheckpoint 対戦相手識別情報として記録するチェックポイントパス
   * @param candidateEvaluator 候補行動を選ぶ推論器
   * @param opponentEvaluator 残り3席の行動を選ぶ推論器
   * @param games 要求する半荘数。完全な4席を入れ替えた対局へ切り上げる
   * @param seedBase 同一牌山の対局組乱数シードの基点
   * @param firstWallFamilyId 最初に使う非負同一牌山の対局組オフセット
   * @param gamesInFlight 同時進行する半荘数
   * @return 対応をそろえた着順差・点差と推論バッチ指標
   */
  public static Evaluation evaluateDuel(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      BatchedPolicy candidateEvaluator,
      BatchedPolicy opponentEvaluator,
      int games,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      EpsilonUtilityProfile utilityProfile,
      DecisionDuelArenaSettings settings,
      InferenceBatchingSettings batchingSettings) {
    DuelRun run =
        runDuel(
            candidateCheckpoint,
            opponentCheckpoint,
            candidateEvaluator,
            opponentEvaluator,
            games,
            seedBase,
            firstWallFamilyId,
            gamesInFlight,
            true,
            ignored -> {},
            utilityProfile,
            settings,
            batchingSettings);
    return new Evaluation(run.result(), run.metrics(), run.rotationOutcomes(), run.wallOutcomes());
  }

  /**
   * 観測リストを保持せず、完了した4席を入れ替えた対局を逐次集約する。
   *
   * <p>長時間の固定牌山評価では全ゲームを一つのスケジューラーへ投入し、完了したゲーム枠を直ちに補充する。{@code wallOutcomeSink}
   * は4席が揃った牌山ごとにスケジューラースレッドから呼ばれるため、進捗表示や固定標本統計を対局実行処理の停止なしで更新できる。
   */
  public static DuelEvaluation evaluateDuelStreaming(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      BatchedPolicy candidateEvaluator,
      BatchedPolicy opponentEvaluator,
      int games,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<DuelEvaluation.WallOutcome> wallOutcomeSink,
      EpsilonUtilityProfile utilityProfile,
      DecisionDuelArenaSettings settings,
      InferenceBatchingSettings batchingSettings) {
    DuelRun run =
        runDuel(
            candidateCheckpoint,
            opponentCheckpoint,
            candidateEvaluator,
            opponentEvaluator,
            games,
            seedBase,
            firstWallFamilyId,
            gamesInFlight,
            false,
            wallOutcomeSink,
            utilityProfile,
            settings,
            batchingSettings);
    return new DuelEvaluation(run.result(), run.metrics());
  }

  private static DuelRun runDuel(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      BatchedPolicy candidateEvaluator,
      BatchedPolicy opponentEvaluator,
      int games,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      boolean collectOutcomes,
      Consumer<DuelEvaluation.WallOutcome> wallOutcomeSink,
      EpsilonUtilityProfile utilityProfile,
      DecisionDuelArenaSettings settings,
      InferenceBatchingSettings batchingSettings) {
    if (firstWallFamilyId < 0L) {
      throw new IllegalArgumentException("firstWallFamilyId must be non-negative");
    }
    if (gamesInFlight <= 0) {
      throw new IllegalArgumentException("gamesInFlight must be positive");
    }
    int totalGames = totalDuelGames(games);
    LOG.info(
        "Decision duel started: selectionMode={} games={} gamesInFlight={}",
        DecisionSelectionMode.POLICY_GREEDY,
        totalGames,
        gamesInFlight);
    Totals totals = new Totals(collectOutcomes, wallOutcomeSink, utilityProfile);
    EpsilonDecisionDuelScheduler.Execution execution =
        EpsilonDecisionDuelScheduler.run(
            candidateEvaluator,
            opponentEvaluator,
            totalGames,
            seedBase,
            firstWallFamilyId,
            gamesInFlight,
            totals::add,
            settings,
            batchingSettings);
    EpsilonDecisionDuelScheduler.Metrics schedulerMetrics = execution.metrics();
    return new DuelRun(
        totals.toResult(candidateCheckpoint, opponentCheckpoint, totalGames),
        new DuelEvaluation.Metrics(
            totalGames,
            Math.min(gamesInFlight, totalGames),
            schedulerMetrics.completedGames(),
            schedulerMetrics.inferenceBatches(),
            schedulerMetrics.inferenceRequests(),
            average(schedulerMetrics.inferenceRequests(), schedulerMetrics.inferenceBatches()),
            schedulerMetrics.maximumInferenceBatch(),
            schedulerMetrics.batching()),
        totals.rotationOutcomes(),
        totals.wallOutcomes());
  }

  private static double average(long total, long count) {
    return count == 0 ? 0.0 : total / (double) count;
  }

  static int totalDuelGames(int requestedGames) {
    if (requestedGames <= 0) {
      throw new IllegalArgumentException("requestedGames must be positive");
    }
    return ((requestedGames + SEAT_ROTATIONS - 1) / SEAT_ROTATIONS) * SEAT_ROTATIONS;
  }

  static Rotation rotationForGame(long gameIndex, long seedBase) {
    if (gameIndex < 0) {
      throw new IllegalArgumentException("gameIndex must be non-negative");
    }
    long wallIndex = gameIndex / SEAT_ROTATIONS;
    int candidateSeat = (int) (gameIndex % SEAT_ROTATIONS);
    return new Rotation(
        wallIndex, EpsilonDecisionSeeds.evalVsGame(seedBase, wallIndex), candidateSeat);
  }

  static double pairedGameRankDelta(int candidateRank, int... opponentRanks) {
    if (candidateRank < 1
        || candidateRank > GameState.NUM_PLAYERS
        || opponentRanks == null
        || opponentRanks.length != GameState.NUM_PLAYERS - 1) {
      throw new IllegalArgumentException(
          "A paired rank delta requires one candidate and 3 opponents");
    }
    double opponentRankTotal = 0.0;
    for (int opponentRank : opponentRanks) {
      if (opponentRank < 1 || opponentRank > GameState.NUM_PLAYERS) {
        throw new IllegalArgumentException("rank must be 1-4: " + opponentRank);
      }
      opponentRankTotal += opponentRank;
    }
    return opponentRankTotal / opponentRanks.length - candidateRank;
  }

  static double pairedWallRankDelta(double... rotationRankDeltas) {
    if (rotationRankDeltas == null || rotationRankDeltas.length != SEAT_ROTATIONS) {
      throw new IllegalArgumentException("A paired wall requires all 4 candidate-seat rotations");
    }
    double sum = 0.0;
    for (double delta : rotationRankDeltas) {
      if (!Double.isFinite(delta)) {
        throw new IllegalArgumentException("rotation paired rank delta must be finite: " + delta);
      }
      sum += delta;
    }
    return sum / SEAT_ROTATIONS;
  }

  private static final class Totals {
    private final boolean collectOutcomes;
    private final Consumer<DuelEvaluation.WallOutcome> wallOutcomeSink;
    private final Map<Long, WallOutcomeBuilder> pendingWalls = new HashMap<>();
    private final ArrayList<RotationOutcome> rotationOutcomes = new ArrayList<>();
    private final ArrayList<DuelEvaluation.WallOutcome> wallOutcomes = new ArrayList<>();
    private final int configuredProfile;
    private double scoreAdvantage;
    private double candidateRankTotal;
    private double opponentRankTotal;
    private int candidateTop;
    private int opponentTop;
    private int candidateLast;
    private int opponentLast;
    private final double[] profileAdvantages = new double[EpsilonUtilityProfile.values().length];
    private int completedWalls;
    private double wallMean;
    private double wallSquaredDeviation;

    private Totals(
        boolean collectOutcomes,
        Consumer<DuelEvaluation.WallOutcome> wallOutcomeSink,
        EpsilonUtilityProfile profile) {
      configuredProfile = profile.ordinal();
      this.collectOutcomes = collectOutcomes;
      this.wallOutcomeSink = wallOutcomeSink;
    }

    private void add(EpsilonDecisionDuelScheduler.CompletedGame completed) {
      add(
          completed.wallIndex(),
          completed.wallSeed(),
          completed.candidateSeat(),
          completed.finalScores());
    }

    private void add(long wallIndex, long wallSeed, int candidateSeat, int[] scores) {
      int[] ranks = ScoreRanking.byScoreThenSeat(scores);
      int candidateScore = 0;
      int opponentScore = 0;
      int candidateRank = 0;
      int candidateTopForGame = 0;
      int candidateLastForGame = 0;
      int opponentTopForGame = 0;
      int opponentLastForGame = 0;
      int[] opponentRanks = new int[GameState.NUM_PLAYERS - 1];
      int opponentRankIndex = 0;
      double[] candidateTotals = new double[EpsilonUtilityProfile.values().length];
      double[] opponentTotals = new double[EpsilonUtilityProfile.values().length];
      for (int seat = 0; seat < scores.length; seat++) {
        if (seat == candidateSeat) {
          candidateScore += scores[seat];
          candidateRank = ranks[seat] + 1;
          candidateRankTotal += candidateRank;
          if (ranks[seat] == 0) {
            candidateTop++;
            candidateTopForGame++;
          } else if (ranks[seat] == scores.length - 1) {
            candidateLast++;
            candidateLastForGame++;
          }
          addProfileUtility(candidateTotals, seat, ranks, scores);
        } else {
          opponentScore += scores[seat];
          int opponentRank = ranks[seat] + 1;
          opponentRanks[opponentRankIndex++] = opponentRank;
          opponentRankTotal += opponentRank;
          if (ranks[seat] == 0) {
            opponentTop++;
            opponentTopForGame++;
          } else if (ranks[seat] == scores.length - 1) {
            opponentLast++;
            opponentLastForGame++;
          }
          addProfileUtility(opponentTotals, seat, ranks, scores);
        }
      }
      double gameScoreAdvantage = candidateScore - opponentScore / 3.0;
      double gameRankAdvantage = pairedGameRankDelta(candidateRank, opponentRanks);
      double gameUtilityAdvantage =
          candidateTotals[configuredProfile] - opponentTotals[configuredProfile] / 3.0;
      scoreAdvantage += gameScoreAdvantage;
      RotationOutcome rotationOutcome =
          new RotationOutcome(
              wallIndex,
              wallSeed,
              candidateSeat,
              gameRankAdvantage,
              gameUtilityAdvantage,
              gameScoreAdvantage,
              candidateTopForGame - opponentTopForGame / 3.0,
              (1 - candidateLastForGame) - (3 - opponentLastForGame) / 3.0);
      if (collectOutcomes) {
        rotationOutcomes.add(rotationOutcome);
      }
      WallOutcomeBuilder wall =
          pendingWalls.computeIfAbsent(
              wallIndex, ignored -> new WallOutcomeBuilder(wallIndex, wallSeed));
      wall.add(rotationOutcome);
      if (wall.complete()) {
        pendingWalls.remove(wallIndex);
        addWallOutcome(wall.finish());
      }
      for (int i = 0; i < profileAdvantages.length; i++) {
        profileAdvantages[i] += candidateTotals[i] - opponentTotals[i] / 3.0;
      }
    }

    private void addWallOutcome(DuelEvaluation.WallOutcome outcome) {
      completedWalls++;
      double delta = outcome.pairedRankDelta() - wallMean;
      wallMean += delta / completedWalls;
      wallSquaredDeviation += delta * (outcome.pairedRankDelta() - wallMean);
      if (collectOutcomes) {
        wallOutcomes.add(outcome);
      }
      wallOutcomeSink.accept(outcome);
    }

    private List<RotationOutcome> rotationOutcomes() {
      rotationOutcomes.sort(
          Comparator.comparingLong(RotationOutcome::wallIndex)
              .thenComparingInt(RotationOutcome::candidateSeat));
      return rotationOutcomes;
    }

    private List<DuelEvaluation.WallOutcome> wallOutcomes() {
      wallOutcomes.sort(Comparator.comparingLong(DuelEvaluation.WallOutcome::wallIndex));
      return wallOutcomes;
    }

    private DuelEvaluation.Result toResult(
        Path candidateCheckpoint, Path opponentCheckpoint, int games) {
      int candidateSeatGames = games;
      int opponentSeatGames = games * (GameState.NUM_PLAYERS - 1);
      int expectedWalls = games / SEAT_ROTATIONS;
      if (!pendingWalls.isEmpty() || completedWalls != expectedWalls) {
        throw new IllegalStateException(
            "Incomplete paired wall rotations: expected="
                + expectedWalls
                + " actual="
                + completedWalls);
      }
      double pairedRankStandardError =
          completedWalls <= 1
              ? 0.0
              : Math.sqrt((wallSquaredDeviation / (completedWalls - 1)) / completedWalls);
      return new DuelEvaluation.Result(
          candidateCheckpoint,
          opponentCheckpoint,
          games,
          completedWalls,
          wallMean,
          pairedRankStandardError,
          wallMean - LCB_95_Z * pairedRankStandardError,
          scoreAdvantage / games,
          candidateRankTotal / candidateSeatGames,
          opponentRankTotal / opponentSeatGames,
          candidateTop / (double) candidateSeatGames,
          opponentTop / (double) opponentSeatGames,
          candidateLast / (double) candidateSeatGames,
          opponentLast / (double) opponentSeatGames,
          divide(profileAdvantages, games));
    }

    private static void addProfileUtility(double[] totals, int seat, int[] ranks, int[] scores) {
      for (EpsilonUtilityProfile profile : EpsilonUtilityProfile.values()) {
        totals[profile.ordinal()] +=
            profile == EpsilonUtilityProfile.SCORE
                ? ((scores[seat] - 25000) / 1000.0) / 50.0
                : profile.utilityForRank(ranks[seat]);
      }
    }

    private static double[] divide(double[] values, int denominator) {
      double[] out = values.clone();
      for (int i = 0; i < out.length; i++) {
        out[i] /= denominator;
      }
      return out;
    }
  }

  /**
   * 対戦評価の統計結果と、その根拠となる席順を入れ替えた対局／牌山観測。
   *
   * @param result 集約した対応をそろえた対戦評価指標
   * @param metrics 対局実行処理のバッチ処理・推論指標
   * @param rotationOutcomes 牌山と候補席ごとの観測
   * @param wallOutcomes 4席を入れ替えた対局をまとめた独立観測
   */
  public record Evaluation(
      DuelEvaluation.Result result,
      DuelEvaluation.Metrics metrics,
      List<RotationOutcome> rotationOutcomes,
      List<DuelEvaluation.WallOutcome> wallOutcomes) {
    /** 観測リストを不変リストに固定して評価結果を構築する。 */
    public Evaluation {
      rotationOutcomes = List.copyOf(rotationOutcomes);
      wallOutcomes = List.copyOf(wallOutcomes);
    }
  }

  private record DuelRun(
      DuelEvaluation.Result result,
      DuelEvaluation.Metrics metrics,
      List<RotationOutcome> rotationOutcomes,
      List<DuelEvaluation.WallOutcome> wallOutcomes) {}

  /**
   * 1つの牌山と候補の席に対応する対戦結果。計測した指標を保持する。
   *
   * @param wallIndex 評価列内の牌山インデックス
   * @param wallSeed 牌山を再現する乱数シード
   * @param candidateSeat この席順を入れ替えた対局で候補を配置した席
   * @param pairedRankDelta 同じ牌山の候補と比較元の平均順位差
   * @param pairedConfiguredUtilityDelta 有効な効用の定義による候補と比較元の差
   * @param meanScoreDelta 候補視点の平均点差
   * @param firstPlaceRateDelta 候補と比較元の1着率差
   * @param lastPlaceAvoidanceDelta 候補と比較元のラス回避率差
   */
  public record RotationOutcome(
      long wallIndex,
      long wallSeed,
      int candidateSeat,
      double pairedRankDelta,
      double pairedConfiguredUtilityDelta,
      double meanScoreDelta,
      double firstPlaceRateDelta,
      double lastPlaceAvoidanceDelta) {}

  private static final class WallOutcomeBuilder {
    private final long wallIndex;
    private final long wallSeed;
    private final double[] rankDeltaBySeat = new double[SEAT_ROTATIONS];
    private final double[] utilityDeltaBySeat = new double[SEAT_ROTATIONS];
    private final boolean[] present = new boolean[SEAT_ROTATIONS];
    private int size;

    private WallOutcomeBuilder(long wallIndex, long wallSeed) {
      this.wallIndex = wallIndex;
      this.wallSeed = wallSeed;
    }

    private void add(RotationOutcome outcome) {
      if (outcome.wallSeed() != wallSeed) {
        throw new IllegalArgumentException(
            "wall family contains multiple wall seeds: wallIndex="
                + wallIndex
                + " expected="
                + wallSeed
                + " actual="
                + outcome.wallSeed());
      }
      int seat = outcome.candidateSeat();
      if (seat < 0 || seat >= SEAT_ROTATIONS) {
        throw new IllegalArgumentException("candidateSeat must be 0-3: " + seat);
      }
      if (present[seat]) {
        throw new IllegalArgumentException(
            "duplicate candidate seat in wall family: wallIndex=" + wallIndex + " seat=" + seat);
      }
      if (!Double.isFinite(outcome.pairedRankDelta())
          || !Double.isFinite(outcome.pairedConfiguredUtilityDelta())) {
        throw new IllegalArgumentException("paired duel deltas must be finite");
      }
      present[seat] = true;
      rankDeltaBySeat[seat] = outcome.pairedRankDelta();
      utilityDeltaBySeat[seat] = outcome.pairedConfiguredUtilityDelta();
      size++;
    }

    private boolean complete() {
      return size == SEAT_ROTATIONS;
    }

    private DuelEvaluation.WallOutcome finish() {
      if (size != SEAT_ROTATIONS) {
        throw new IllegalArgumentException(
            "wall family must contain all 4 candidate seats: wallIndex="
                + wallIndex
                + " actual="
                + size);
      }
      return new DuelEvaluation.WallOutcome(
          wallIndex,
          wallSeed,
          pairedWallRankDelta(rankDeltaBySeat),
          pairedWallRankDelta(utilityDeltaBySeat));
    }
  }

  record Rotation(long wallIndex, long wallSeed, int candidateSeat) {}
}
