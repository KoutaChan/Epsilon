package com.epsilon.agent.eval;

import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * 2 スロットの卓で、牌山 1 つについて候補と相手を比べた値。席の数を S とする。分け方 n（スロット 0 の候補が n 席）の割り当て p の半荘で、
 * 候補の席の平均順位を r̄(p)、候補の席の平均の効用を ū(p)、席あたりの効用の平均（順位ごとの効用の平均）を ū とすると、次のようになる。
 *
 * <pre>
 * 順位: δ_n = (S − 1)/(S − n) × 平均_p [ r̄(p) − (S + 1)/2 ]   （負が良い）
 * 効用: υ_n = (S − 1)/(S − n) × 平均_p [ ū(p) − ū ]            （正が良い）
 * 牌山の値: δ = その牌山で打った分け方の δ_n の平均（分け方ごとに同じ重み）。υ も同じ
 * </pre>
 *
 * 平均_p は、その牌山で打った分け方 n の割り当てについての平均。順位は最終の持ち点の順で、同点なら席の順（{@link Placement}）。席ごとの
 * 強さの差が足し算で効くなら、係数を掛けることで、どの分け方の値も候補 1 席・相手 S − 1 席の卓の値に換算できる。|δ| ≤ (S − 1)/2。
 *
 * <p>候補と相手が同じ方策なら、どの割り当ての半荘も席を入れ替えただけの同じ半荘になる。そのため分け方の全ての割り当てを通すと、候補の席は
 * 各順位に同じ回数ずつ入る。順位は整数の和から、効用は順位ごとの席の割合と 1/S の差から求めるので、このとき全ての牌山で δ と υ は丸め誤差
 * なしに 0 になる。
 *
 * <p>値は次に {@link #measure} を呼ぶまで有効。同時に使えるのは 1 つのスレッドだけ。
 */
public final class PairScore {

  private final Placements placements;
  private final int seats;

  /** [割り当て][席] → スロット。 */
  private final int[][] slotOfSeat;

  /** 割り当てごとの分け方の番号。 */
  private final int[] splitOf;

  /** 分け方ごとの候補の席の数 n と、係数 (S − 1)/(S − n)。 */
  private final int[] candidateSeats;

  private final double[] scale;

  /** 順位ごとの効用。 */
  private final double[] utilityOfPlace;

  /** 分け方ごとの、候補の席の順位（1 始まり）の和、順位ごとの候補の席の数（添字は [分け方 × S ＋ 順位]）、打った割り当ての数。 */
  private final long[] rankSum;

  private final int[] places, games;
  private final double[] rankBySplit, utilityBySplit;
  private double rank, utility;

  /**
   * @param plan 2 スロットの卓の計画
   * @throws IllegalArgumentException スロットが 2 つでないとき
   */
  public PairScore(ArenaPlan plan) {
    this(plan.layout(), plan.utility());
  }

  /**
   * 打ち手のモデルを読み込まずに、席の割り当てから作る。ジョブの記録を読むときに使う。
   *
   * @param placements 2 スロットの卓の席の割り当て
   * @param utility 順位ごとの効用
   * @throws IllegalArgumentException スロットが 2 つでないとき
   */
  public PairScore(Placements placements, RankUtility utility) {
    if (placements.slots() != 2) {
      throw new IllegalArgumentException("A paired score compares two slots, not " + placements.slots());
    }
    this.placements = placements;
    Variant variant = placements.variant();
    seats = variant.seats();
    slotOfSeat = placements.slotTable();
    splitOf = new int[slotOfSeat.length];
    for (int placement = 0; placement < splitOf.length; placement++) {
      splitOf[placement] = placements.splitOf(placement);
    }
    int splits = placements.splits().size();
    candidateSeats = new int[splits];
    scale = new double[splits];
    for (int split = 0; split < splits; split++) {
      candidateSeats[split] = placements.splits().get(split).seatsOf(0);
      scale[split] = (seats - 1) / (double) (seats - candidateSeats[split]);
    }
    utilityOfPlace = new double[seats];
    for (int place = 0; place < seats; place++) {
      utilityOfPlace[place] = utility.value(variant, place);
    }
    rankSum = new long[splits];
    places = new int[splits * seats];
    games = new int[splits];
    rankBySplit = new double[splits];
    utilityBySplit = new double[splits];
  }

  /**
   * 1 つの牌山の値を求める。
   *
   * @param wall 牌山の番号
   * @param scores [牌山の中の半荘][席] の最終の持ち点（{@link WallSink#accept} と同じ順）
   */
  public void measure(long wall, int[][] scores) {
    if (scores.length != placements.gamesPerWall()) {
      throw new IllegalArgumentException(
          "Wall " + wall + " has " + scores.length + " games, the plan " + placements.gamesPerWall());
    }
    Arrays.fill(rankSum, 0);
    Arrays.fill(places, 0);
    Arrays.fill(games, 0);
    for (int game = 0; game < scores.length; game++) {
      int placement = placements.placement(wall, game);
      int split = splitOf[placement];
      int[] slots = slotOfSeat[placement];
      for (int seat = 0; seat < seats; seat++) {
        if (slots[seat] == 0) {
          int place = Placement.rank(scores[game], seat);
          rankSum[split] += place + 1;
          places[split * seats + place]++;
        }
      }
      games[split]++;
    }
    double middle = (seats + 1) / 2.0;
    double share = 1.0 / seats;
    rank = 0;
    utility = 0;
    int played = 0;
    for (int split = 0; split < games.length; split++) {
      if (games[split] == 0) {
        rankBySplit[split] = Double.NaN;
        utilityBySplit[split] = Double.NaN;
        continue;
      }
      double candidates = (double) candidateSeats[split] * games[split];
      double gap = 0;
      for (int place = 0; place < seats; place++) {
        gap += (places[split * seats + place] / candidates - share) * utilityOfPlace[place];
      }
      rankBySplit[split] = scale[split] * (rankSum[split] / candidates - middle);
      utilityBySplit[split] = scale[split] * gap;
      rank += rankBySplit[split];
      utility += utilityBySplit[split];
      played++;
    }
    rank /= played;
    utility /= played;
  }

  /** @return この牌山の δ */
  public double rank() {
    return rank;
  }

  /** @return この牌山の υ */
  public double utility() {
    return utility;
  }

  /** @return 判定に使う量で見た、この牌山の値 */
  public double value(Score score) {
    return switch (score) {
      case RANK -> rank;
      case UTILITY -> utility;
    };
  }

  /**
   * @param split 分け方の番号
   * @return その分け方の δ_n（この牌山で打たなかった分け方なら NaN）
   */
  public double rank(int split) {
    return rankBySplit[split];
  }

  /**
   * @param split 分け方の番号
   * @return その分け方の υ_n（この牌山で打たなかった分け方なら NaN）
   */
  public double utility(int split) {
    return utilityBySplit[split];
  }
}
