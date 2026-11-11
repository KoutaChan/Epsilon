package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.FinalRoundReference;
import com.epsilon.grp.FinalRoundWorkspace;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.grp.RankPermutations;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 最後の局の数え上げの速さを、実際の卓とモデルで測る（{@code native} のグループ）。名前が *Test で終わらないので、
 * -Dtest=FinalRoundBench で指定したときだけ走る。卓は、-Dlate.games に渡した 1 コホートの主経路の記録（GameLog の形式 1）から取る。
 * そのうち数え上げる局末から、等間隔に -Dlate.tables 卓（既定 2,000）を選ぶ。葉の GRP は -Dlate.leaf、局の結果のネットワークは
 * -Dlate.outcomes に渡したチェックポイントから読む。
 *
 * <p>1 スレッドで、打ち切りの値ごとに次をログに書く。1 回の呼び出しの µs（平均・中央値・99%）。呼び出し 1 回あたりの、精算した組の
 * 数、葉のネットワークに通した葉の数、葉のメモに当たった数、半荘が終わる葉の数。埋めた確率（平均・99%・最大）と、半荘が終わる組の
 * 確率の平均。内訳として、局の結果のネットワークの順伝播の µs と、葉を一様な定数に替えた参照実装（FinalRoundReference）の 1 回の
 * µs（順伝播・数え上げ・精算・足し込み）も書く。葉のネットワークだけにかかる時間は、1 卓の µs で書く。葉にネットワークを使った
 * 参照実装の結果が、FinalRound とビット単位で違う卓の数も書く。打ち切り 1e-5 では、葉のメモの大きさごとの当たりと µs も書く。
 * 合否は判定しない。
 */
public class FinalRoundBench {

  private static final Logger LOG = LoggerFactory.getLogger(FinalRoundBench.class);

  private static final double[] CUTOFFS = {1e-4, 1e-5, 1e-6, 1e-7};
  private static final int DEFAULT_TABLES = 2_000;

  /** 葉のネットワークだけの時間を測るために記録する、葉の卓の数の上限。 */
  private static final int RECORDED_LEAVES = 200_000;

  private static final double MEMO_CUTOFF = 1e-5;
  private static final int[] LEAF_SLOTS = {1 << 10, 1 << 12, 1 << 14};

  @Test(groups = "native")
  public void measure() throws IOException {
    String games = System.getProperty("late.games");
    String leafPath = System.getProperty("late.leaf");
    String outcomesPath = System.getProperty("late.outcomes");
    if (games == null || leafPath == null || outcomesPath == null) {
      throw new SkipException("Pass -Dlate.games, -Dlate.leaf and -Dlate.outcomes to measure on recorded tables");
    }
    int wanted = Integer.getInteger("late.tables", DEFAULT_TABLES);
    GameCohort cohort = OldGameLog.read(Path.of(games));
    List<GrpState> covered = new ArrayList<>();
    long handEnds = 0;
    for (GameRecord game : cohort.games()) {
      for (GrpState state : game.handEnds()) {
        handEnds++;
        if (FinalRound.covers(state)) {
          covered.add(state);
        }
      }
    }
    int stride = Math.max(1, covered.size() / wanted);
    List<GrpState> sample = new ArrayList<>();
    for (int index = 0; index < covered.size() && sample.size() < wanted; index += stride) {
      sample.add(covered.get(index));
    }
    GrpNetwork leaf = network(GrpCheckpoint.read(Path.of(leafPath)), leafPath);
    OutcomeNetwork outcomes = GrpCheckpoint.readOutcomes(Path.of(outcomesPath));
    LOG.info(String.format(Locale.ROOT,
        "LATE-BENCH cohort %d: %d games, %d hand ends, %d covered (%.2f%%), %d sampled; leaf %s, outcomes %s",
        cohort.cohort(), cohort.games().size(), handEnds, covered.size(), 100.0 * covered.size() / handEnds,
        sample.size(), leaf.shape(), outcomes.shape()));
    measureOutcomes(outcomes, sample);
    for (double cutoff : CUTOFFS) {
      FinalRound round = FinalRound.of(outcomes, leaf, cutoff);
      measureParts(round, sample, measureCalls(round, sample));
    }
    FinalRound round = FinalRound.of(outcomes, leaf, MEMO_CUTOFF);
    for (int slots : LEAF_SLOTS) {
      measureMemo(round, sample, slots);
    }
  }

  /**
   * 1 回の呼び出しの µs、呼び出し 1 回あたりの組と葉の数、埋めた確率、半荘が終わる組の確率を測ってログに書く。
   *
   * @return 卓ごとの p(π|s)（並びは [卓 × S! ＋ 順列]）
   */
  private static double[] measureCalls(FinalRound round, List<GrpState> sample) {
    int count = RankPermutations.of(round.variant()).count();
    int tables = sample.size();
    double[] p = new double[count];
    FinalRoundWorkspace warm = FinalRoundWorkspace.of(round);
    for (GrpState state : sample) {
      round.permutations(state, warm, p);
    }
    double[] values = new double[tables * count];
    double[] filled = new double[tables];
    double terminal = 0;
    double[] micros = new double[tables];
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round);
    for (int index = 0; index < tables; index++) {
      long start = System.nanoTime();
      filled[index] = round.permutations(sample.get(index), workspace, p);
      micros[index] = (System.nanoTime() - start) / 1e3;
      terminal += workspace.terminalMass();
      System.arraycopy(p, 0, values, index * count, count);
    }
    double[] sortedFilled = filled.clone();
    Arrays.sort(sortedFilled);
    LOG.info(String.format(Locale.ROOT,
        "LATE-BENCH cutoff %.0e: %s us/call; settled %.0f, leaf network %.1f, leaf memo hits %.1f,"
            + " terminal leaves %.0f per call; filled mass mean %.3e p99 %.3e max %.3e; terminal mass mean %.4f",
        round.cutoff(), spread(micros), workspace.settled() / (double) tables,
        workspace.networkLeaves() / (double) tables, workspace.memoHits() / (double) tables,
        workspace.terminalLeaves() / (double) tables, mean(filled), percentile(sortedFilled, 0.99),
        sortedFilled[tables - 1], terminal / tables));
    return values;
  }

  /**
   * 内訳を測る。葉を一様な定数に替えた参照実装の 1 回の µs と、葉のネットワークだけにかかる 1 卓の µs を書く。後者は、葉に
   * ネットワークを使う参照実装を動かして葉の卓を記録し、その卓をネットワークに通し直して測る。その参照実装の値は、FinalRound の値と
   * ビット単位で比べる。
   *
   * @param values FinalRound の卓ごとの p(π|s)
   */
  private static void measureParts(FinalRound round, List<GrpState> sample, double[] values) {
    GrpNetwork leaf = round.leaf();
    int count = RankPermutations.of(round.variant()).count();
    double[] p = new double[count];
    List<GrpState> leaves = new ArrayList<>();
    GrpWorkspace recording = GrpWorkspace.of(leaf.shape());
    FinalRoundReference recorder = FinalRoundReference.of(round, (table, out) -> {
      if (leaves.size() < RECORDED_LEAVES) {
        leaves.add(GrpState.of(table));
      }
      leaf.permutations(table, recording, out);
    });
    int differing = 0;
    for (int index = 0; index < sample.size(); index++) {
      recorder.permutations(sample.get(index), p);
      boolean same = true;
      for (int k = 0; k < count; k++) {
        same &= Double.doubleToRawLongBits(p[k]) == Double.doubleToRawLongBits(values[index * count + k]);
      }
      differing += same ? 0 : 1;
    }
    double uniform = 1.0 / count;
    FinalRoundReference constant = FinalRoundReference.of(round, (table, out) -> Arrays.fill(out, uniform));
    for (GrpState state : sample) {
      constant.permutations(state, p);
    }
    long constantStart = System.nanoTime();
    for (GrpState state : sample) {
      constant.permutations(state, p);
    }
    double constantMicros = (System.nanoTime() - constantStart) / 1e3 / sample.size();
    GrpWorkspace workspace = GrpWorkspace.of(leaf.shape());
    double leafMicros = 0;
    for (int pass = 0; pass < 2; pass++) {
      long start = System.nanoTime();
      for (GrpState state : leaves) {
        leaf.permutations(state, workspace, p);
      }
      leafMicros = (System.nanoTime() - start) / 1e3 / leaves.size();
    }
    double perCall = recorder.networkLeaves() / (double) sample.size();
    LOG.info(String.format(Locale.ROOT,
        "LATE-BENCH cutoff %.0e parts: constant leaves %.1f us/call (outcome forward, enumeration, settlement and"
            + " sums); leaf network %.2f us/table over %d recorded tables x %.1f network leaves per call = %.1f us/call;"
            + " copy with network leaves differs from FinalRound in %d of %d tables",
        round.cutoff(), constantMicros, leafMicros, leaves.size(), perCall, perCall * leafMicros, differing,
        sample.size()));
  }

  /** 葉のメモの大きさを指定して、1 回の呼び出しの µs と、メモに当たった割合を測る。 */
  private static void measureMemo(FinalRound round, List<GrpState> sample, int slots) {
    double[] p = new double[RankPermutations.of(round.variant()).count()];
    FinalRoundWorkspace warm = FinalRoundWorkspace.of(round, slots);
    for (GrpState state : sample) {
      round.permutations(state, warm, p);
    }
    FinalRoundWorkspace workspace = FinalRoundWorkspace.of(round, slots);
    long start = System.nanoTime();
    for (GrpState state : sample) {
      round.permutations(state, workspace, p);
    }
    double micros = (System.nanoTime() - start) / 1e3 / sample.size();
    long lookups = workspace.memoHits() + workspace.networkLeaves();
    LOG.info(String.format(Locale.ROOT,
        "LATE-BENCH cutoff %.0e leaf memo %d slots: %.1f us/call, %.1f%% of %.1f leaf lookups per call hit",
        round.cutoff(), slots, micros, 100.0 * workspace.memoHits() / lookups, lookups / (double) sample.size()));
  }

  /** 局の結果のネットワークで、1 回の順伝播にかかる µs を測る。 */
  private static void measureOutcomes(OutcomeNetwork outcomes, List<GrpState> sample) {
    OutcomeWorkspace workspace = OutcomeWorkspace.of(outcomes.shape());
    double micros = 0;
    for (int pass = 0; pass < 3; pass++) {
      long start = System.nanoTime();
      for (GrpState state : sample) {
        outcomes.forward(state, workspace);
      }
      micros = (System.nanoTime() - start) / 1e3 / sample.size();
    }
    LOG.info(String.format(Locale.ROOT, "LATE-BENCH outcome network forward: %.2f us/table", micros));
  }

  private static GrpNetwork network(GrpModel model, String path) throws IOException {
    if (!(model instanceof GrpNetwork network)) {
      throw new IOException(path + " holds " + model + ", not a GRP network");
    }
    return network;
  }

  /** 平均・中央値・99% の値を「平均 (median 中央値, p99 99% の値)」の形の文字列にする。 */
  private static String spread(double[] values) {
    double[] sorted = values.clone();
    Arrays.sort(sorted);
    return String.format(Locale.ROOT, "%.1f (median %.1f, p99 %.1f)", mean(values), percentile(sorted, 0.5),
        percentile(sorted, 0.99));
  }

  private static double mean(double[] values) {
    double sum = 0;
    for (double value : values) {
      sum += value;
    }
    return sum / values.length;
  }

  /** 昇順に並べた値から、下から割合 q の位置にある値を返す。 */
  private static double percentile(double[] sorted, double q) {
    return sorted[Math.min(sorted.length - 1, (int) (q * sorted.length))];
  }
}
