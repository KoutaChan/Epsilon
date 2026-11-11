package com.epsilon.prism.eval;

import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.PlayStats;
import com.epsilon.agent.eval.Placements;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Stopped;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallMean;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.util.SeedMixer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.testng.Assert;

/**
 * 答えの分かる合成の決闘。三人打ちの 1v2 の卓で、牌山ごと・半荘ごとに決めた候補の着順から games.tsv の行を書き、決闘の結果（δ・υ の平均と
 * 標準誤差）も同じ行から作る。1v2 の全ての割り当てを打つので牌山 1 つで 3 半荘になり、牌山の δ は 3 半荘での候補の着順の平均 − 2 になる。
 *
 * <p>評価のジョブ（{@link EvaluationJobs}）の計画の作り方と結果の反映はそのまま使い、打ち手を開かずに、この行と結果を返す。
 */
final class SyntheticDuels {

  /** 着順ごとの点（1 位・2 位・3 位）。 */
  private static final int[] POINTS = {50_000, 35_000, 20_000};

  /** 結果の信頼区間に使う臨界値（両側 1%）。 */
  static final double CRITICAL = 2.5758293035489004;

  private SyntheticDuels() {}

  /** 候補の着順（0 始まり）を決める関数。 */
  @FunctionalInterface
  interface Places {

    /**
     * @param job ジョブ
     * @param wall 計画の中で何番目の牌山か（0 始まり）
     * @param game 牌山の中の半荘の番号
     * @return 候補の着順（0 始まり）
     */
    int place(JobFile job, int wall, int game);
  }

  /**
   * 牌山ごとの δ の期待値が目標の値 t になるように、着順を乱数で選ぶ。1 位の確率を 1/3 − t/2、3 位の確率を 1/3 + t/2 にする。乱数はジョブの
   * 名前・牌山・半荘から作るので、何度実行しても同じ着順になる。
   *
   * @param target ジョブから目標の値 t を返す関数（|t| ≤ 2/3）
   */
  static Places drawn(Function<JobFile, Double> target) {
    return (job, wall, game) -> {
      double t = target.apply(job);
      long state = SeedMixer.mix64(0x5eedL ^ wall);
      for (int i = 0; i < job.name().length(); i++) {
        state = SeedMixer.mix64(state ^ job.name().charAt(i));
      }
      state = SeedMixer.mix64(state ^ game);
      double u = (state >>> 11) * 0x1.0p-53;
      double first = 1 / 3.0 - t / 2;
      return u < first ? 0 : u < first + 1 / 3.0 ? 1 : 2;
    };
  }

  /**
   * 1v2 の卓の games.tsv の行を作る。候補の席には着順の点を、ほかの 2 席には残りの点を席の順に置く。
   *
   * @param duel 決闘の計画
   * @param job ジョブ
   * @param places 候補の着順を決める関数
   * @return games.tsv の行（牌山の順。牌山の中は半荘の順）
   */
  static List<String> lines(DuelSetup duel, JobFile job, Places places) {
    Placements layout = duel.layout();
    int[][] slotTable = layout.slotTable();
    int seats = duel.variant().seats();
    List<String> lines = new ArrayList<>();
    for (int index = 0; index < duel.walls(); index++) {
      long wall = duel.wallBase() + index;
      for (int game = 0; game < layout.gamesPerWall(); game++) {
        int placement = layout.placement(wall, game);
        int[] slots = slotTable[placement];
        int place = places.place(job, index, game);
        int[] scores = new int[seats];
        int next = 0;
        StringBuilder seatSlots = new StringBuilder();
        for (int seat = 0; seat < seats; seat++) {
          seatSlots.append(slots[seat]);
          if (slots[seat] == 0) {
            scores[seat] = POINTS[place];
          } else {
            next += next == place ? 1 : 0;
            scores[seat] = POINTS[next++];
          }
        }
        StringBuilder line = new StringBuilder();
        line.append(wall).append('\t').append(placement).append('\t').append(seatSlots);
        for (int score : scores) {
          line.append('\t').append(score);
        }
        line.append('\t').append((double) (place + 1));
        line.append('\t').append(duel.utility().value(duel.variant(), place));
        new PlayStats(2).appendTo(line);
        lines.add(line.toString());
      }
    }
    return lines;
  }

  /** 牌山ごとの値の平均と標準誤差。 */
  static Estimate estimate(double[] values) {
    WallMean mean = new WallMean();
    for (double value : values) {
      mean.add(value);
    }
    return Estimate.of(mean.mean(), mean.error(), CRITICAL);
  }

  /** games.tsv の行から、1v2 の卓で計画の牌山を全て打った決闘の結果を作る。 */
  static DuelResult result(DuelSetup duel, List<String> lines) {
    WallValues values = WallValues.of(duel, lines);
    Estimate rank = estimate(values.rank());
    Estimate utility = estimate(values.utility());
    List<Double> shares = List.of(1 / 3.0, 1 / 3.0, 1 / 3.0);
    return new DuelResult(
        values.walls(),
        lines.size(),
        Verdict.EVEN,
        Stopped.LIMIT,
        duel.score(),
        CRITICAL,
        rank,
        utility,
        List.of(new DuelResult.SplitValue("1v2", rank, utility)),
        List.of(
            new DuelResult.SlotValue(duel.candidate().label(), 2 + rank.mean(), rank.error(), 0, 0, shares),
            new DuelResult.SlotValue(duel.opponent().label(), 2, 0, 0, 0, shares)),
        new PlayStats.Rates(0, 0, 0, 0, 0),
        0,
        0,
        0);
  }

  /**
   * 王者の卓の、判定が決まった結果。δ は候補が良いなら −0.2、悪いなら 0.2、互角なら 0。
   *
   * @param walls 使った牌山の数
   * @param error δ の標準誤差
   */
  static DuelResult decided(Verdict verdict, int walls, double error) {
    double mean =
        switch (verdict) {
          case BETTER -> -0.2;
          case WORSE -> 0.2;
          default -> 0;
        };
    Estimate rank = Estimate.of(mean, error, CRITICAL);
    List<Double> shares = List.of(1 / 3.0, 1 / 3.0, 1 / 3.0);
    return new DuelResult(
        walls,
        walls * 6L,
        verdict,
        Stopped.DECIDED,
        Score.RANK,
        CRITICAL,
        rank,
        new Estimate(0, 0, 0, 0),
        List.of(),
        List.of(
            new DuelResult.SlotValue("candidate", 2 + mean, error, 0, 0, shares),
            new DuelResult.SlotValue("champion", 2, 0, 0, 0, shares)),
        new PlayStats.Rates(0, 0, 0, 0, 0),
        0,
        0,
        0);
  }

  /** 王者の卓が無い run で使う、王者の卓の結果。呼ばれたらテストを落とす。 */
  static DuelResult noChampion(JobFile job) {
    throw new AssertionError("Unexpected champion duel " + job.name());
  }

  /**
   * 計画の作り方と結果の反映は評価のジョブのまま使い、打ち手を開かずに合成の行と結果を返す。打たずに候補を最初の王者に
   * するジョブは、評価のジョブのまま開く。王者の卓は {@code champion} が返す結果を、ほかの卓は {@code places} の着順から作った行と結果を返す。
   */
  record Runner(EvaluationJobs jobs, Function<JobFile, DuelResult> champion, Places places) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) throws Exception {
      return jobs.plan(job);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws Exception {
      EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, job.plan()).value();
      if (plan.crowns()) {
        return jobs.open(job, directory);
      }
      TableSpec table = RecordFile.read(TableSpec.class, job.definition()).value();
      DuelResult result;
      List<String> lines;
      if (TableSpec.kind(table.opponent()) == TableSpec.Opponent.CHAMPION) {
        result = champion.apply(job);
        lines = List.of();
      } else {
        Assert.assertEquals(plan.duel().splits(), List.of("1v2"), job.name());
        lines = lines(plan.duel(), job, places);
        result = result(plan.duel(), lines);
      }
      return new JobWork() {
        @Override
        public Chunk next(long walls, JobOutputs outputs) throws Exception {
          for (String line : lines) {
            outputs.game(line);
          }
          return Chunk.decided(result.walls(), result);
        }

        @Override
        public void close() {}
      };
    }

    @Override
    public void settle(JobFile job, Node.Table result) throws Exception {
      jobs.settle(job, result);
    }
  }
}
