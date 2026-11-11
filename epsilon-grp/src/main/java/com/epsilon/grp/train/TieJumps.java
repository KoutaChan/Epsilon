package com.epsilon.grp.train;

import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.RankPermutations;
import com.epsilon.grp.Seats;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 同点のまわりの段差（TieJump）を測る。実測はデータの全ての組で、予測は点差が ±500 点以内の組から乱数の種で選んだ最大 4,096 組で測る。 */
public final class TieJumps {

  /** 点差の刻みと、見る点差の上限。 */
  private static final int STEP = 100;

  private static final int REACH = 500;
  private static final int BINS = 2 * REACH / STEP + 1;
  private static final int ZERO = REACH / STEP;

  /** 予測に使う組の数と、組を選ぶ乱数の種。 */
  private static final int SAMPLES = 4_096;

  private static final long SEED = 0x7E1E_5EEDL;

  /** 今の順位が隣り合う 2 席。1 つ目は起家に近い席（同点なら上位になる席）。 */
  private record NearTie(int game, int handEnd, int first, int second) {}

  private TieJumps() {}

  /**
   * ネットワークが予測する同点のまわりの段差と、局末のデータでの段差を測る。ネットワークと局末でルール（四人打ち・三人打ち）が
   * 違えば IllegalArgumentException を投げる。
   */
  public static TieJump measure(HandEnds ends, GrpNetwork network, GrpWorkspace workspace) {
    if (network.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + network.variant() + " network for " + ends.variant + " hand ends");
    }
    int seats = ends.seats;
    AdjacentBins realized = new AdjacentBins(1, BINS);
    List<NearTie> ties = new ArrayList<>();
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        GrpState state = ends.states[h];
        for (int place = 0; place < seats - 1; place++) {
          int upper = ends.seatAt(h, place);
          int lower = ends.seatAt(h, place + 1);
          int first = Math.min(upper, lower);
          int second = Math.max(upper, lower);
          int difference = state.score(first) - state.score(second);
          if (Math.abs(difference) > REACH) {
            continue;
          }
          boolean above = ends.finalRank(h, first) < ends.finalRank(h, second);
          realized.add(0, Math.floorDiv(difference, STEP) + ZERO, above ? 1 : 0);
          ties.add(new NearTie(game, h, first, second));
        }
      }
      realized.endGame();
    }
    Paired realizedJump = realized.differences()[0][ZERO - 1];
    double[] realizedCurve = new double[BINS];
    for (int bin = 0; bin < BINS; bin++) {
      realizedCurve[bin] = realized.count(0, bin) > 0 ? realized.mean(0, bin) : Double.NaN;
    }
    RankPermutations table = RankPermutations.of(ends.variant);
    double[] probabilities = new double[table.count()];
    double[] curve = new double[BINS];
    double[] predictedCurve = new double[BINS];
    Clusters predicted = new Clusters(1);
    int[] chosen = Samples.choose(ties.size(), SAMPLES, SEED);
    int currentGame = chosen.length == 0 ? -1 : ties.get(chosen[0]).game();
    for (int index : chosen) {
      NearTie tie = ties.get(index);
      if (tie.game() != currentGame) {
        predicted.endGame();
        currentGame = tie.game();
      }
      for (int bin = 0; bin < BINS; bin++) {
        int difference = (bin - ZERO) * STEP;
        curve[bin] = above(network, workspace, table, ends.states[tie.handEnd()], tie, difference, probabilities);
        predictedCurve[bin] += curve[bin] / chosen.length;
      }
      predicted.add(0, curve[ZERO] - curve[ZERO - 1]);
    }
    predicted.endGame();
    if (chosen.length == 0) {
      Arrays.fill(predictedCurve, Double.NaN);
    }
    Paired predictedJump = predicted.paired(0);
    return new TieJump(
        ties.size(),
        predictedJump.mean(),
        predictedJump.standardError(),
        realizedJump.mean(),
        realizedJump.standardError(),
        predictedCurve,
        realizedCurve);
  }

  /**
   * 2 席の点差を指定した値に変えた卓（2 席の点の和は変えない）で、1 つ目の席が最終順位で 2 つ目の席より上になる確率を
   * ネットワークで求める。
   */
  private static double above(
      GrpNetwork network,
      GrpWorkspace workspace,
      RankPermutations table,
      GrpState state,
      NearTie tie,
      int difference,
      double[] probabilities) {
    int seats = table.variant().seats();
    int[] scores = state.scores();
    int total = scores[tie.first()] + scores[tie.second()];
    scores[tie.first()] = (total + difference) / 2;
    scores[tie.second()] = total - scores[tie.first()];
    GrpState moved = new GrpState(state.variant(), state.round(), state.honba(), state.sticks(), scores);
    network.permutations(moved, workspace, probabilities);
    int firstRelative = Seats.relative(state.dealer(), tie.first(), seats);
    int secondRelative = Seats.relative(state.dealer(), tie.second(), seats);
    double probability = 0;
    for (int permutation = 0; permutation < probabilities.length; permutation++) {
      if (table.rank(permutation, firstRelative) < table.rank(permutation, secondRelative)) {
        probability += probabilities[permutation];
      }
    }
    return probability;
  }
}
