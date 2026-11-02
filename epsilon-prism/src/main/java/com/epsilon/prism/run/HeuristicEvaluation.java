package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.eval.PlayStats;
import com.epsilon.prism.policy.PlayerPolicy;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 版の網の 3×Heuristic の評価（train の評価）: 版の網を 1 席、Heuristic を 3 席に置き、各席の σ（網の
 * 方策は貪欲）で evalGames 局打つ（{@link Arena#duel}。牌山の種は seed ^ {@link Arena#EVALUATION_SALT}、半荘 ID は {@link
 * Arena#HEURISTIC_BASE} から）。
 *
 * @param heuristic 対局の集計
 * @param play 対局の内訳（{@link PlayStats#report}）
 */
public record HeuristicEvaluation(Arena.Result heuristic, Map<String, Object> play) {
  /** 内訳のログは 1 行。 */
  private static final Gson LINES = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  /**
   * 版 snapshot の網 net（打つ網か学習の網）を評価する（{@link Snapshot#open}）。
   *
   * @param devices 推論のデバイス
   */
  public static HeuristicEvaluation run(
      PrismConfig config,
      Path snapshot,
      Path net,
      CollectionDriver driver,
      PolicyExecutionContext execution,
      List<Device> devices,
      EpsilonUtilityProfile profile)
      throws IOException {
    var stats = new PlayStats();
    Arena.Result result;
    try (var candidate =
            Snapshot.open(
                snapshot,
                net,
                devices,
                execution,
                config.inferBatch(),
                config.priorSettings(),
                config.inference());
        var opponent = PlayerPolicy.heuristic()) {
      result =
          Arena.duel(
              driver,
              candidate,
              opponent,
              "heuristic",
              config.evalGames(),
              config.seed() ^ Arena.EVALUATION_SALT,
              Arena.HEURISTIC_BASE,
              profile,
              stats);
    }
    return new HeuristicEvaluation(result, stats.report());
  }

  /** evaluations.jsonl の値（heuristic と play）。 */
  public Map<String, Object> record() {
    var map = new LinkedHashMap<String, Object>();
    map.put("heuristic", heuristic);
    map.put("play", play);
    return map;
  }

  /**
   * ログの集計の 1 行。
   *
   * @param label 版の番号に添える名前（打つ網は空）
   */
  public String summary(int version, String label) {
    return String.format(
        "version %d%s vs 3xHeuristic: games=%d rank=%.3f±%.3f utility=%.4f±%.4f"
            + " shares=%.3f/%.3f/%.3f/%.3f %.1fs",
        version,
        label,
        heuristic.games(),
        heuristic.meanRank(),
        heuristic.rankError(),
        heuristic.meanUtility(),
        heuristic.utilityError(),
        heuristic.rankShares()[0],
        heuristic.rankShares()[1],
        heuristic.rankShares()[2],
        heuristic.rankShares()[3],
        heuristic.seconds());
  }

  /**
   * ログの内訳の 1 行。
   *
   * @param label 版の番号に添える名前（打つ網は空）
   */
  public String playSummary(int version, String label) {
    return "version " + version + label + " play " + LINES.toJson(play);
  }
}
