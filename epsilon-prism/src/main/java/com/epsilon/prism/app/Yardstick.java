package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.Seating;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.prism.eval.EvaluationPlan;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/** 同じ牌山を全ての席で回した、候補 1 席の実戦卓の候補を測る。標準誤差の標本単位は牌山。 */
@Named(
    value = "yardstick",
    doc =
        "Report placement, Tenhou points, and pairwise Elo for a completed rotated one-candidate"
            + " table")
public record Yardstick(
    @Key(
            doc = "Completed evaluate job directory containing job.json and result.json",
            presence = Presence.REQUIRED)
        Path job)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws IOException {
    var saved = RecordFile.read(JobFile.class, job.resolve(Jobs.JOB)).value();
    if (saved.state() != JobFile.State.DONE) {
      throw new IOException("The evaluation job is incomplete: " + job);
    }
    var setup = RecordFile.read(EvaluationPlan.class, saved.plan()).value().duel();
    if (setup == null) throw new IOException("The yardstick requires a duel plan: " + job);
    int seats = setup.variant().seats();
    if (setup.utility() != RankUtility.TENHOU
        || setup.seating() != Seating.EVERY_PLACEMENT
        || !setup.splits().equals(List.of("1v" + (seats - 1)))
        || setup.wallSet() != WallSet.FIXED
        || !(setup.stop() instanceof StopRule.Fixed)) {
      throw new IOException(
          "The yardstick requires a fixed Tenhou table with one candidate in each seat");
    }
    var result = Jobs.result(job, DuelResult.class);
    if (result.walls() < 2
        || result.walls() != setup.walls()
        || result.games() != (long) seats * result.walls()) {
      throw new IOException(
          "The yardstick requires at least two complete wall clusters of one game per seat: "
              + job);
    }
    var candidate = result.slots().getFirst();
    String summary =
        String.format(
            Locale.ROOT,
            "%s vs %s, walls=%d hanchan=%d, placement=%.7f +/- %.7f, Tenhou pt=%+.6f +/- %.6f,"
                + " Elo=%s (1 wall-clustered SE)",
            candidate.label(),
            result.slots().get(1).label(),
            result.walls(),
            result.games(),
            candidate.meanRank(),
            candidate.rankError(),
            RankUtility.TENHOU_POINTS * candidate.meanUtility(),
            RankUtility.TENHOU_POINTS * candidate.utilityError(),
            elo(seats, candidate.meanRank(), candidate.rankError()));
    context.out().println(summary);
    return Outcome.ok("Measured " + saved.name());
  }

  /**
   * 平均順位から、相手 1 人との勝率（下に置いた相手の割合）を通した Elo の差を求める。
   *
   * @param seats 卓の席の数
   * @param rank 候補の平均順位（1 始まり）
   * @param standardError 平均順位の標準誤差
   */
  static String elo(int seats, double rank, double standardError) {
    if (!(rank >= 1 && rank <= seats && standardError >= 0 && Double.isFinite(standardError))) {
      throw new IllegalArgumentException("Invalid mean placement or standard error");
    }
    if (rank == 1) return "+infinity (SE undefined)";
    if (rank == seats) return "-infinity (SE undefined)";
    int opponents = seats - 1;
    double win = (seats - rank) / opponents;
    double rating = 400 * Math.log10(win / (1 - win));
    double error = 400 * standardError / (opponents * Math.log(10) * win * (1 - win));
    return String.format(Locale.ROOT, "%+.4f +/- %.4f", rating, error);
  }
}
