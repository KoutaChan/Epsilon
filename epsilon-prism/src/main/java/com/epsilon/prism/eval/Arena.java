package com.epsilon.prism.eval;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.RoundResult;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.policy.PolicyService;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 物差しとの対局。候補の方策を1席、相手の方策を3席に置き、候補の席は半荘 ID で回す。牌山は半荘 ID だけで決まるので、同じ基準 ID なら反復をまたいで
 * 同じ牌山の対比較になる。収集と同じドライバを通し、行も写しも作らない（{@link Sampling#OFF}）。手は各席の σ から引く（行動方策 b は使わない。
 * 網の方策の σ は貪欲）。
 */
public final class Arena {
  /** 3×Heuristic の評価の牌山の種の塩と基準の半荘 ID（train の評価と evaluate が同じ牌山で測る）。 */
  public static final long EVALUATION_SALT = 0x4556414cL, HEURISTIC_BASE = 1L << 40;

  public record Result(
      String opponent,
      int games,
      double meanRank,
      double rankError,
      double meanUtility,
      double utilityError,
      double[] rankShares,
      double seconds) {}

  /** 1半荘の候補の着順（1〜4）と効用。 */
  public record Game(long gameId, int rank, double utility) {}

  /** 集計と、半荘ごとの結果（半荘 ID の順）。 */
  public record Detailed(Result result, Game[] games) {}

  private Arena() {}

  public static Result duel(
      CollectionDriver driver,
      PolicyService candidate,
      PolicyService opponent,
      String opponentName,
      int games,
      long seed,
      long gameIdBase,
      EpsilonUtilityProfile profile) {
    return duel(driver, candidate, opponent, opponentName, games, seed, gameIdBase, profile, null);
  }

  /**
   * @param watcher 局と判断の診断（null なら使わない）。候補の席は {@code Math.floorMod(gameId, 4)}
   */
  public static Result duel(
      CollectionDriver driver,
      PolicyService candidate,
      PolicyService opponent,
      String opponentName,
      int games,
      long seed,
      long gameIdBase,
      EpsilonUtilityProfile profile,
      CollectionDriver.GameObserver watcher) {
    return detailed(
            driver, candidate, opponent, opponentName, games, seed, gameIdBase, profile, watcher)
        .result();
  }

  /**
   * {@link #duel} と同じ対局の集計に、半荘ごとの候補の着順と効用を添えたもの。同じ seed と基準 ID なら、候補を替えても
   * 同じ牌山・同じ席の半荘ごとの対比較になる（版どうしや設定どうしの差を、牌山の雑音を除いて測る）。
   */
  public static Detailed detailed(
      CollectionDriver driver,
      PolicyService candidate,
      PolicyService opponent,
      String opponentName,
      int games,
      long seed,
      long gameIdBase,
      EpsilonUtilityProfile profile,
      CollectionDriver.GameObserver watcher) {
    long started = System.nanoTime();
    int[][] finals = play(driver, candidate, opponent, games, seed, gameIdBase, profile, watcher);
    Game[] played = new Game[games];
    double rankSum = 0, rankSquares = 0, utilitySum = 0, utilitySquares = 0;
    double[] shares = new double[4];
    for (int game = 0; game < games; game++) {
      long gameId = gameIdBase + game;
      int seat = Math.floorMod(gameId, 4);
      int rank = ScoreRanking.byScoreThenSeat(finals[game])[seat];
      double utility = EpsilonUtilityTargets.utilityDouble(profile, finals[game][seat], rank);
      played[game] = new Game(gameId, rank + 1, utility);
      rankSum += rank + 1;
      rankSquares += (rank + 1.0) * (rank + 1.0);
      utilitySum += utility;
      utilitySquares += utility * utility;
      shares[rank]++;
    }
    double meanRank = rankSum / games, meanUtility = utilitySum / games;
    for (int r = 0; r < 4; r++) shares[r] /= games;
    var result =
        new Result(
            opponentName,
            games,
            meanRank,
            Math.sqrt(Math.max(0, rankSquares / games - meanRank * meanRank) / games),
            meanUtility,
            Math.sqrt(Math.max(0, utilitySquares / games - meanUtility * meanUtility) / games),
            shares,
            (System.nanoTime() - started) / 1e9);
    return new Detailed(result, played);
  }

  /** 半荘ごとの最終の点数（半荘 ID の順）。 */
  private static int[][] play(
      CollectionDriver driver,
      PolicyService candidate,
      PolicyService opponent,
      int games,
      long seed,
      long gameIdBase,
      EpsilonUtilityProfile profile,
      CollectionDriver.GameObserver watcher) {
    var finals = new ConcurrentHashMap<Long, int[]>();
    CollectionDriver.GameObserver observer =
        watcher == null
            ? finals::put
            : new CollectionDriver.GameObserver() {
              @Override
              public void gameEnded(long gameId, int[] finalScores) {
                finals.put(gameId, finalScores);
                watcher.gameEnded(gameId, finalScores);
              }

              @Override
              public void decided(
                  long gameId, GameState state, EngineDecisionPoint point, Action action) {
                watcher.decided(gameId, state, point, action);
              }

              @Override
              public void handEnded(long gameId, GameState state, RoundResult result) {
                watcher.handEnded(gameId, state, result);
              }

              @Override
              public void handValued(long gameId, float[] nextRootPrefix, int[] scores) {
                watcher.handValued(gameId, nextRootPrefix, scores);
              }

              @Override
              public GameRecorder recorder(long gameId) {
                return watcher.recorder(gameId);
              }
            };
    List<Long> ids = new ArrayList<>(games);
    for (int game = 0; game < games; game++) ids.add(gameIdBase + game);
    var cohort =
        new CollectionDriver.Cohort(
            seed,
            seed,
            ids,
            profile,
            Sampling.OFF,
            null,
            gameId -> {
              PolicyService[] seats = {opponent, opponent, opponent, opponent};
              seats[Math.floorMod(gameId, 4)] = candidate;
              return seats;
            },
            observer);
    try (var phi = HandValue.scoreRank(profile)) {
      driver.collect(cohort, candidate, phi);
    }
    int[][] scores = new int[games][];
    for (int game = 0; game < games; game++) {
      scores[game] = finals.get(gameIdBase + game);
      if (scores[game] == null)
        throw new IllegalStateException("Missing result for game " + (gameIdBase + game));
    }
    return scores;
  }
}
