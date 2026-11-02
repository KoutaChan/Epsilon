package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.core.ScoreRanking;
import java.util.function.Consumer;

/**
 * 局末の評価 Φ。局の途中では打ち切らず、局が終わった時点の次局 root（公開情報だけ）から4席の期待順位効用を返す。半荘が終わった局は最終順位の効用をそのまま使う。
 *
 * <p>{@link #request} はドライバのスレッドから呼び、completion もドライバのスレッドで呼ばれる。
 */
public interface HandValue extends AutoCloseable {
  /** 次局 root の公開系列を評価する。配列は呼び出し側が書き換えない。scores は精算後の点数。 */
  void request(float[] nextRootPrefix, int[] scores, Consumer<double[]> completion);

  /** 評価待ちの件数。 */
  int pending();

  /** 呼び出し・メモ化の命中・評価器の時間。 */
  Stats stats();

  @Override
  default void close() {}

  record Stats(long requests, long memoHits, long evaluated, long batches, long busyNanos) {}

  /** 局末の点数をそのまま最終順位とみなす代用の評価。GRP を使わない CPU の試験と計測に使う。 */
  static HandValue scoreRank(EpsilonUtilityProfile profile) {
    return new HandValue() {
      private long requests;

      @Override
      public void request(float[] prefix, int[] scores, Consumer<double[]> completion) {
        requests++;
        completion.accept(terminal(scores, profile));
      }

      @Override
      public int pending() {
        return 0;
      }

      @Override
      public Stats stats() {
        return new Stats(requests, 0, requests, 0, 0);
      }
    };
  }

  /** 半荘終了時の4席の効用。 */
  static double[] terminal(int[] scores, EpsilonUtilityProfile profile) {
    if (scores.length != 4) throw new IllegalArgumentException("Four scores required");
    int[] ranks = ScoreRanking.byScoreThenSeat(scores);
    double[] utilities = new double[4];
    for (int seat = 0; seat < 4; seat++)
      utilities[seat] = EpsilonUtilityTargets.utilityDouble(profile, scores[seat], ranks[seat]);
    return utilities;
  }
}
