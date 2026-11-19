package com.epsilon.prism.grp;

import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** 次局、精算後の順位と点差で区分した、最終順位の予測と実測。順位は 1 始まり。 */
public final class RankCalibration {
  /** 次局。南場の最後の局の 1 つ前（四人打ちは南 3 局、三人打ちは南 2 局）、南場の最後の局、西場。 */
  public enum Round {
    BEFORE_LAST,
    LAST,
    WEST
  }

  /**
   * 隣の順位の席との点差。区切りは子の和了の打点の目安で、2 翻 30 符のロン（2,000）、3 翻 30 符のロン（3,900 を丸めた 4,000）、満貫
   * （8,000）。点差がこの打点より小さければ、ほかの席から 1 回和了るだけで順位が入れ替わる。
   */
  public enum Gap {
    BELOW_2000(Integer.MIN_VALUE),
    FROM_2000(2000),
    FROM_4000(4000),
    FROM_8000(8000);

    /** この区分の点差の下限（含む）。 */
    private final int from;

    Gap(int from) {
      this.from = from;
    }

    static Gap of(int points) {
      Gap gap = BELOW_2000;
      for (Gap candidate : values()) if (points >= candidate.from) gap = candidate;
      return gap;
    }
  }

  public enum Direction {
    ABOVE,
    BELOW
  }

  public record Line(
      Round round,
      Direction direction,
      Gap gap,
      int rank,
      long seatHands,
      Double[] predicted,
      Double[] observed,
      Double predictedUtility,
      Double observedUtility) {}

  private static final class Cell {
    final double[] probability;
    final long[] outcomes;
    double estimated, realized;
    long count;

    Cell(int seats) {
      probability = new double[seats];
      outcomes = new long[seats];
    }

    void add(HandEndValues values, int h, int seat) {
      count++;
      estimated += values.phi(h, seat);
      realized += values.ends().utility(h, seat);
      outcomes[values.ends().finalRank(h, seat)]++;
      for (int r = 0; r < probability.length; r++) probability[r] += values.marginal(h, seat, r);
    }
  }

  private RankCalibration() {}

  public static List<Line> read(GrpFiles files, PhiSetup setup, int cohort) throws IOException {
    if (!files.recorded(cohort))
      throw new IOException("No settled game records for cohort " + cohort);
    var data = PhiScorer.recorded(files, setup, cohort);
    var ends = data.ends();
    var late = ends.select(h -> ends.state(h).round() >= ends.variant().lastRound() - 1);
    var values = HandEndValues.of(late, HandEndPhi.openSingleSlot(data.value(), setup.utility()));
    return measure(values);
  }

  static List<Line> measure(HandEndValues values) {
    HandEnds ends = values.ends();
    int seats = ends.seats();
    Cell[][][][] cells =
        new Cell[Round.values().length][Direction.values().length][Gap.values().length][seats];
    for (var rounds : cells)
      for (var directions : rounds)
        for (var gaps : directions)
          for (int rank = 0; rank < seats; rank++) gaps[rank] = new Cell(seats);
    int[] seatAt = new int[seats];
    for (int h = 0; h < ends.handEnds(); h++) {
      var state = ends.state(h);
      int next = state.round();
      if (next < ends.variant().lastRound() - 1) continue;
      Round round =
          next > ends.variant().lastRound()
              ? Round.WEST
              : next == ends.variant().lastRound() ? Round.LAST : Round.BEFORE_LAST;
      for (int seat = 0; seat < seats; seat++) seatAt[ends.rank(h, seat)] = seat;
      for (int rank = 1; rank < seats; rank++) {
        int upper = seatAt[rank - 1], lower = seatAt[rank];
        Gap gap = Gap.of(state.score(upper) - state.score(lower));
        cells[round.ordinal()][Direction.BELOW.ordinal()][gap.ordinal()][rank - 1].add(
            values, h, upper);
        cells[round.ordinal()][Direction.ABOVE.ordinal()][gap.ordinal()][rank].add(
            values, h, lower);
      }
    }
    List<Line> lines = new ArrayList<>();
    for (Round round : Round.values())
      for (Direction direction : Direction.values())
        for (Gap gap : Gap.values()) {
          int first = direction == Direction.ABOVE ? 1 : 0;
          for (int rank = first; rank < first + seats - 1; rank++) {
            Cell cell = cells[round.ordinal()][direction.ordinal()][gap.ordinal()][rank];
            Double[] predicted = new Double[seats], observed = new Double[seats];
            for (int r = 0; r < seats; r++) {
              predicted[r] = mean(cell.probability[r], cell.count);
              observed[r] = mean(cell.outcomes[r], cell.count);
            }
            lines.add(
                new Line(
                    round,
                    direction,
                    gap,
                    rank + 1,
                    cell.count,
                    predicted,
                    observed,
                    mean(cell.estimated, cell.count),
                    mean(cell.realized, cell.count)));
          }
        }
    return lines;
  }

  private static Double mean(double sum, long n) {
    return n == 0 ? null : sum / n;
  }
}
