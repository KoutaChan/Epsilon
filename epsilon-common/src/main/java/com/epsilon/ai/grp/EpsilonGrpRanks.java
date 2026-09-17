package com.epsilon.ai.grp;

import com.epsilon.core.GameState;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 4人の最終順位の可逆符号化と、各席が各順位になる確率の配列操作を行う。 */
public final class EpsilonGrpRanks {

  /** 対局に参加する席数。 */
  public static final int SEAT_COUNT = GameState.NUM_PLAYERS;

  /** 予測する順位の種類数。 */
  public static final int RANK_COUNT = GameState.NUM_PLAYERS;

  /** 席ごとにまとめた 4x4 周辺確率の要素数。 */
  public static final int MATRIX_SIZE = SEAT_COUNT * RANK_COUNT;

  /** 4人の順位順列総数。 */
  public static final int CODE_COUNT = 24;

  private static final float MIN_RANK_PROBABILITY_MASS = 1.0e-8f;
  private static final float MARGINAL_SUM_TOLERANCE = 1.0e-4f;
  private static final int[] FACTORIAL = {1, 1, 2, 6, 24};

  private EpsilonGrpRanks() {}

  /**
   * 席ごとの最終順位の順列を、レーマー符号を使って0～23の整数に変換する。
   *
   * @param ranksBySeat 席ごとの 0始まりの最終順位
   * @return 0～23 の可逆な順位コード
   */
  public static int encode(int[] ranksBySeat) {
    validateRanks(ranksBySeat);
    int code = 0;
    for (int seat = 0; seat < SEAT_COUNT; seat++) {
      int smallerToRight = 0;
      for (int next = seat + 1; next < SEAT_COUNT; next++) {
        if (ranksBySeat[next] < ranksBySeat[seat]) {
          smallerToRight++;
        }
      }
      code += smallerToRight * FACTORIAL[SEAT_COUNT - 1 - seat];
    }
    return code;
  }

  /**
   * レーマー符号で表した0～23の整数から、席ごとの最終順位を復元する。
   *
   * @param code 0～23 の順位順列のコード
   * @return 席ごとの 0始まりの最終順位
   */
  public static int[] decode(int code) {
    requireCode(code);
    ArrayList<Integer> remaining = new ArrayList<>(List.of(0, 1, 2, 3));
    int[] ranksBySeat = new int[SEAT_COUNT];
    int rest = code;
    for (int seat = 0; seat < SEAT_COUNT; seat++) {
      int factor = FACTORIAL[SEAT_COUNT - 1 - seat];
      int index = rest / factor;
      rest %= factor;
      ranksBySeat[seat] = remaining.remove(index);
    }
    return ranksBySeat;
  }

  /**
   * 順位コードを4行4列の教師配列に変換する。各席に対応する行では、正解順位の要素を1、それ以外を0とする。
   *
   * @param code 0～23 の順位順列のコード
   * @return 席ごとにまとめた、正解の要素だけが1である教師配列
   */
  public static float[] oneHotMarginals(int code) {
    int[] ranks = decode(code);
    float[] target = new float[MATRIX_SIZE];
    for (int seat = 0; seat < SEAT_COUNT; seat++) {
      target[index(seat, ranks[seat])] = 1.0f;
    }
    return target;
  }

  /**
   * 席ごとにまとめた周辺確率配列で指定席・順位に対応するインデックスを返す。
   *
   * @param seat 0始まりの席
   * @param rank 0始まりの順位
   * @return {@code seat * RANK_COUNT + rank}
   */
  public static int index(int seat, int rank) {
    requireSeat(seat);
    if (rank < 0 || rank >= RANK_COUNT) {
      throw new IllegalArgumentException("rank must be 0-3: " + rank);
    }
    return seat * RANK_COUNT + rank;
  }

  /**
   * 席ごとにまとめた 4x4 周辺確率から、指定席の4順位分布をコピーする。
   *
   * @param marginals 席ごとにまとめた4x4順位周辺確率
   * @param seat 取り出す0始まりの席
   * @return 1着から4着の順に並ぶ独立配列
   */
  public static float[] seatMarginal(float[] marginals, int seat) {
    if (marginals == null || marginals.length != MATRIX_SIZE) {
      throw new IllegalArgumentException("marginals size must be " + MATRIX_SIZE);
    }
    requireSeat(seat);
    int offset = seat * RANK_COUNT;
    return Arrays.copyOfRange(marginals, offset, offset + RANK_COUNT);
  }

  /** 4順位に対応する値が有限かつ非負で、総和が正であることを検証し、正規化前の総和を返す。 */
  public static double requireRankProbabilityMass(float[] values) {
    if (values == null || values.length != RANK_COUNT) {
      throw new IllegalArgumentException("rank distribution must have length " + RANK_COUNT);
    }
    double sum = 0.0;
    for (int rank = 0; rank < RANK_COUNT; rank++) {
      float value = values[rank];
      if (!Float.isFinite(value) || value < 0.0f) {
        throw new IllegalArgumentException(
            "rank distribution must be finite and non-negative: rank=" + rank + " value=" + value);
      }
      sum += value;
    }
    if (!Double.isFinite(sum) || sum < MIN_RANK_PROBABILITY_MASS) {
      throw new IllegalArgumentException("rank distribution must have positive mass: " + sum);
    }
    return sum;
  }

  /** 呼び出し側が所有する4順位周辺確率を検証し、その場で総和1へ正規化する。 */
  public static float[] normalizeRankProbabilitiesInPlace(float[] values) {
    float inverse = (float) (1.0 / requireRankProbabilityMass(values));
    for (int rank = 0; rank < RANK_COUNT; rank++) {
      values[rank] *= inverse;
    }
    return values;
  }

  /** 4行4列の各要素が正で、行和と列和が1に近いことを検証し、入力から独立した配列を返す。 */
  public static float[] requirePositiveMarginals(float[] values, String label) {
    return requireMarginals(values == null ? null : values.clone(), label, true);
  }

  /** 呼び出し側の4行4列の配列を複製せず、各要素が正で行和と列和が1に近いことを検証する。 */
  public static float[] requirePositiveMarginalsInPlace(float[] values, String label) {
    return requireMarginals(values, label, true);
  }

  /** 4行4列の各要素が非負で、行和と列和が1に近いことを検証し、入力から独立した配列を返す。 */
  public static float[] requireNonNegativeMarginals(float[] values, String label) {
    return requireMarginals(values == null ? null : values.clone(), label, false);
  }

  /**
   * 終局順位のコードが有効範囲にあることを要求する。
   *
   * @param code 検証する順位コード
   * @throws IllegalArgumentException 順位コードが0～23 でない場合
   */
  public static void requireCode(int code) {
    if (!isValidCode(code)) {
      throw new IllegalArgumentException("final ranks code must be 0-23: " + code);
    }
  }

  /**
   * 終局順位のコードが有効範囲にあるかを返す。
   *
   * @param code 検査する順位コード
   * @return 0～23 なら {@code true}
   */
  public static boolean isValidCode(int code) {
    return code >= 0 && code < CODE_COUNT;
  }

  private static float[] requireMarginals(float[] values, String label, boolean requirePositive) {
    if (values == null || values.length != MATRIX_SIZE) {
      throw new IllegalArgumentException(label + " size must be " + MATRIX_SIZE);
    }
    double[] rowSums = new double[SEAT_COUNT];
    double[] columnSums = new double[RANK_COUNT];
    for (int seat = 0; seat < SEAT_COUNT; seat++) {
      for (int rank = 0; rank < RANK_COUNT; rank++) {
        float value = values[index(seat, rank)];
        if (!Float.isFinite(value) || (requirePositive ? value <= 0.0f : value < 0.0f)) {
          throw new IllegalArgumentException(
              label
                  + (requirePositive
                      ? " must be finite and positive"
                      : " must be finite and non-negative"));
        }
        rowSums[seat] += value;
        columnSums[rank] += value;
      }
    }
    for (int seat = 0; seat < SEAT_COUNT; seat++) {
      requireUnitSum(rowSums[seat], label + " row " + seat);
    }
    for (int rank = 0; rank < RANK_COUNT; rank++) {
      requireUnitSum(columnSums[rank], label + " column " + rank);
    }
    return values;
  }

  private static void requireUnitSum(double sum, String label) {
    if (Math.abs(sum - 1.0) > MARGINAL_SUM_TOLERANCE) {
      throw new IllegalArgumentException(label + " sum must be 1: " + sum);
    }
  }

  private static void validateRanks(int[] ranksBySeat) {
    if (ranksBySeat == null || ranksBySeat.length != SEAT_COUNT) {
      throw new IllegalArgumentException("ranksBySeat size mismatch");
    }
    boolean[] seen = new boolean[RANK_COUNT];
    for (int rank : ranksBySeat) {
      if (rank < 0 || rank >= RANK_COUNT || seen[rank]) {
        throw new IllegalArgumentException("final ranks must be a permutation of 0..3");
      }
      seen[rank] = true;
    }
  }

  private static void requireSeat(int seat) {
    if (seat < 0 || seat >= SEAT_COUNT) {
      throw new IllegalArgumentException("seat must be 0-3: " + seat);
    }
  }
}
