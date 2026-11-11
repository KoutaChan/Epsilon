package com.epsilon.agent.eval;

import java.util.List;

/**
 * 終わった決闘の games.tsv（列は {@link DuelTally} に書いた形）から、牌山ごとの順位と効用の対の量（{@link PairScore}）を求める。打ち手のモデルを
 * 開かずに読めるので、別のジョブの結果と同じ牌山どうしで対にするときに使う。決闘が判定に足した値と同じ点から同じ式で求めるため、
 * 求めた値はその値とビット単位で一致する。
 */
public final class WallValues {

  private final double[] rank;
  private final double[] utility;

  private WallValues(double[] rank, double[] utility) {
    this.rank = rank;
    this.utility = utility;
  }

  /**
   * games.tsv の行を読む。
   *
   * @param setup ジョブの計画
   * @param lines games.tsv の行（牌山の番号の順、牌山の中は半荘の順）
   * @return 計画の最初の牌山から、行が書かれた最後の牌山までの値
   * @throws IllegalArgumentException 行の数が牌山 1 つ分の半荘の数の倍数でないとき、または計画の牌山の数を超えるとき。行が計画の牌山・
   *     割り当て・席ごとのスロットと合わないとき。列の数か数値の書き方が誤っているとき
   */
  public static WallValues of(DuelSetup setup, List<String> lines) {
    Placements placements = setup.layout();
    int seats = setup.variant().seats();
    int perWall = placements.gamesPerWall();
    if (lines.size() % perWall != 0 || lines.size() / perWall > setup.walls()) {
      throw new IllegalArgumentException(
          "games.tsv holds " + lines.size() + " games, not whole walls of " + perWall + " games within "
              + setup.walls() + " walls");
    }
    int walls = lines.size() / perWall;
    int columns = DuelTally.columns(seats, placements.slots());
    String[] seatSlots = DuelTally.seatSlots(placements.slotTable());
    PairScore pair = new PairScore(placements, setup.utility());
    int[][] scores = new int[perWall][seats];
    double[] rank = new double[walls];
    double[] utility = new double[walls];
    for (int index = 0; index < walls; index++) {
      long wall = setup.wallBase() + index;
      for (int game = 0; game < perWall; game++) {
        int number = index * perWall + game;
        int placement = placements.placement(wall, game);
        String[] fields =
            DuelTally.fields(lines.get(number), number, wall, game, placement, seatSlots[placement], columns);
        for (int seat = 0; seat < seats; seat++) {
          scores[game][seat] = Integer.parseInt(fields[DuelTally.SCORE_COLUMN + seat]);
        }
      }
      pair.measure(wall, scores);
      rank[index] = pair.rank();
      utility[index] = pair.utility();
    }
    return new WallValues(rank, utility);
  }

  /** @return 読んだ牌山の数 */
  public int walls() {
    return rank.length;
  }

  /** @return 牌山ごとの順位の対の量（牌山の番号の順。負が良い） */
  public double[] rank() {
    return rank.clone();
  }

  /** @return 牌山ごとの効用の対の量（牌山の番号の順。正が良い） */
  public double[] utility() {
    return utility.clone();
  }

  /**
   * @param score 判定に使う量
   * @return 牌山ごとの値。大きいほど候補が良い（着順なら順位の対の量にマイナスを付けた値、効用なら効用の対の量）
   */
  public double[] better(Score score) {
    return switch (score) {
      case RANK -> {
        double[] values = new double[rank.length];
        for (int index = 0; index < values.length; index++) {
          values[index] = -rank[index];
        }
        yield values;
      }
      case UTILITY -> utility.clone();
    };
  }
}
