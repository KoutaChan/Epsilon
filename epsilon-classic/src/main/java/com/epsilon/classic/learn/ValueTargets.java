package com.epsilon.classic.learn;

import com.epsilon.mahjong.rules.RankUtility;
import java.util.Arrays;

/**
 * 局の中で V-trace を使って求める価値の目標と、方策のアドバンテージ。ある席のある局での判断を、起きた順に t = 0, …, T − 1 とする。
 * その局末の価値を G（{@link GameTrajectory#endValue}）とする。局の中に報酬は無く、割引もしない。
 *
 * <pre>
 * V_T = v_T = G
 * q_t = (1 − λ)·V_{t+1} + λ·v_{t+1}
 * v_t = V_t + c_t·(q_t − V_t)     （価値の目標）
 * A_t = q_t − V_t                 （方策のアドバンテージ）
 * </pre>
 *
 * V_t は学習の前のネットワークの価値、c_t は選んだ手の補正 {@link #credit}（選び替えで局の進みが変わりえない判断では 1）、λ は次の判断の価値の代わりに
 * その判断の価値の目標を使う割合。
 */
public final class ValueTargets {

  private ValueTargets() {}

  /**
   * 選んだ手の補正。探索で選んだ手から、どれだけ学ぶかの割合。
   *
   * <pre>
   * c = min(1, α + (1 − α)·π(a)/μ(a))
   * </pre>
   *
   * @param alpha 混ぜる割合 α（0 より大きく 1 以下）
   * @param rollout 学習の前の方策で選んだ手を選ぶ確率 π(a)
   * @param behaviour 挙動方策（実際に打った方策）で選んだ手を選ぶ確率 μ(a)（0 より大きい）
   * @return 補正 c
   */
  public static double credit(double alpha, double rollout, double behaviour) {
    return Math.min(1, alpha + (1 - alpha) * rollout / behaviour);
  }

  /**
   * 全ての判断について、価値の目標とアドバンテージを書く。配列はどれも判断の通しの番号の順。
   *
   * @param batch 1 回の macro の軌跡
   * @param utility 局末の価値を求めるのに使う、順位の効用
   * @param values 学習の前のネットワークの価値 V_t
   * @param credits 選んだ手の補正 c_t
   * @param lambda 次の判断の価値の代わりにその判断の価値の目標を使う割合 λ（0 以上 1 以下）
   * @param targets 価値の目標 v_t を書く先
   * @param advantages アドバンテージ A_t を書く先
   */
  public static void compute(
      TrajectoryBatch batch,
      RankUtility utility,
      float[] values,
      float[] credits,
      double lambda,
      float[] targets,
      float[] advantages) {
    if (!(lambda >= 0 && lambda <= 1)) {
      throw new IllegalArgumentException("Lambda must be in [0, 1]: " + lambda);
    }
    int seats = batch.variant().seats();
    int[] activeHand = new int[seats];
    boolean[] chained = new boolean[seats];
    double[] nextValue = new double[seats];
    double[] nextTarget = new double[seats];
    for (int game = 0; game < batch.games().size(); game++) {
      GameTrajectory trajectory = batch.games().get(game);
      int start = batch.start(game);
      Arrays.fill(activeHand, -1);
      for (int decision = trajectory.size() - 1; decision >= 0; decision--) {
        int seat = trajectory.seat(decision);
        int hand = trajectory.hand(decision);
        if (activeHand[seat] != hand) {
          activeHand[seat] = hand;
          chained[seat] = false;
          nextTarget[seat] = trajectory.endValue(hand, seat, utility);
        }
        int at = start + decision;
        double value = values[at];
        double next = chained[seat] ? nextValue[seat] : nextTarget[seat];
        double lookahead = (1 - lambda) * next + lambda * nextTarget[seat];
        double target = value + credits[at] * (lookahead - value);
        targets[at] = (float) target;
        advantages[at] = (float) (lookahead - value);
        nextValue[seat] = value;
        nextTarget[seat] = target;
        chained[seat] = true;
      }
    }
  }
}
