package com.epsilon.ai.grp;

import java.util.concurrent.CompletableFuture;

/** 局開始時点までの特徴量系列から、最終順位の確率を非同期に予測するインターフェース。 */
@FunctionalInterface
public interface EpsilonGrpRankPredictor {

  /**
   * 局開始時点までの GRP 特徴量系列から指定席の最終順位分布を非同期に予測する。
   *
   * @param sequence 一次元化済み GRP 特徴量系列
   * @param seat 予測対象の席
   * @return 順位 1～4 の確率を完了値に持つ非同期処理の結果。系列が空なら完了値は {@code null}
   */
  CompletableFuture<float[]> predictRankProbabilitiesAsync(float[] sequence, int seat);

  /**
   * 局開始時点までの GRP 特徴量系列から絶対席×順位の4x4 周辺確率を非同期に返す。
   *
   * <p>専用実装を持たない予測器では4席の既存APIを合成する。本番バッチ化処理はこのメソッドをオーバーライドし、一つのGRP 順伝播だけを実行する。
   *
   * @param sequence 一次元化済みGRP 特徴量系列
   * @return 席ごとにまとめた16確率を完了値に持つ非同期処理の結果
   */
  default CompletableFuture<float[]> predictMarginalProbabilitiesAsync(float[] sequence) {
    @SuppressWarnings("unchecked")
    CompletableFuture<float[]>[] seats =
        (CompletableFuture<float[]>[]) new CompletableFuture<?>[EpsilonGrpRanks.SEAT_COUNT];
    for (int seat = 0; seat < seats.length; seat++) {
      seats[seat] = predictRankProbabilitiesAsync(sequence, seat);
    }
    return CompletableFuture.allOf(seats)
        .thenApply(
            ignored -> {
              float[] marginals = new float[EpsilonGrpRanks.MATRIX_SIZE];
              for (int seat = 0; seat < seats.length; seat++) {
                float[] row = seats[seat].join();
                if (row == null) {
                  return null;
                }
                if (row.length != EpsilonGrpRanks.RANK_COUNT) {
                  throw new IllegalStateException(
                      "GRP seat marginal width must be " + EpsilonGrpRanks.RANK_COUNT);
                }
                System.arraycopy(
                    row,
                    0,
                    marginals,
                    seat * EpsilonGrpRanks.RANK_COUNT,
                    EpsilonGrpRanks.RANK_COUNT);
              }
              return marginals;
            });
  }
}
