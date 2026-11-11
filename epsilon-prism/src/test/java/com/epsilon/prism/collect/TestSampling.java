package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Openness;
import java.util.Arrays;

/** テスト用の収集の率。打牌の 3 クラス・確約のクラス・オーラスの和了の 3 つにまとめて率を決める。 */
public final class TestSampling {

  private TestSampling() {}

  /**
   * まとめて決めた率から収集の率を作る。全ての合法手を測るのは打牌の 3 クラスだけで、ラベル行の率は手の形に依らない。中立ロールインは
   * 全ての根の判断のクラスで ε_q・ρ_br・上限が同じで、d は候補集合の全ての手から引く。
   *
   * @param holdoutFraction ホールドアウトにする半荘の割合
   * @param labelRateDiscardEarly 脅威のない序盤の打牌のラベル行の率
   * @param labelRateDiscardMid 脅威のない中盤の打牌のラベル行の率
   * @param labelRateDiscardLate 脅威のない終盤の打牌のラベル行の率
   * @param labelRateCommit 確約のクラス（オーラスの和了を除く）のラベル行の率
   * @param labelRateFinalEnd オーラスの和了のラベル行の率
   * @param maxCandidates 候補集合の大きさの上限
   * @param measureAllDiscards 打牌の 3 クラスで全ての合法手を測るか
   * @param epsilonC 候補集合に一様に引いた 1 手を足す確率
   * @param rollinRateDiscard 打牌のクラスの中立ロールインの率
   * @param rollinRateCommit 確約のクラスの中立ロールインの率
   * @param epsilonQ 逸脱の分布に混ぜる一様分布の割合
   * @param rhoBranch 写しの中で逸脱した席の判断をラベル行にする率
   * @param maxCopyLabels 写しごとのラベル行の上限
   * @return 収集の率
   */
  public static Sampling grouped(
      double holdoutFraction,
      double labelRateDiscardEarly,
      double labelRateDiscardMid,
      double labelRateDiscardLate,
      double labelRateCommit,
      double labelRateFinalEnd,
      int maxCandidates,
      boolean measureAllDiscards,
      double epsilonC,
      double rollinRateDiscard,
      double rollinRateCommit,
      double epsilonQ,
      double rhoBranch,
      int maxCopyLabels) {
    double[] labelRates =
        byClass(labelRateDiscardEarly, labelRateDiscardMid, labelRateDiscardLate, labelRateCommit, labelRateFinalEnd);
    double[][] byHand = new double[Openness.COUNT][];
    Arrays.fill(byHand, labelRates);
    boolean[] measureAll = new boolean[DecisionClass.COUNT];
    double[] rollinRates =
        byClass(rollinRateDiscard, rollinRateDiscard, rollinRateDiscard, rollinRateCommit, rollinRateCommit);
    RollInRule[] rollIn = new RollInRule[DecisionClass.COUNT];
    for (DecisionClass c : DecisionClass.values()) {
      measureAll[c.ordinal()] = measureAllDiscards && !c.commitment();
      rollIn[c.ordinal()] =
          RollInRule.uniform(rollinRates[c.ordinal()], epsilonQ, DeviationTarget.ANY, rhoBranch, maxCopyLabels);
    }
    return new Sampling(holdoutFraction, byHand, measureAll, maxCandidates, epsilonC, rollIn);
  }

  /** 打牌の 3 クラス・確約のクラス・オーラスの和了の率を、クラスの番号の順の表にする。 */
  private static double[] byClass(double early, double mid, double late, double commit, double finalEnd) {
    double[] rates = new double[DecisionClass.COUNT];
    for (DecisionClass c : DecisionClass.values()) {
      rates[c.ordinal()] =
          switch (c) {
            case DISCARD_EARLY -> early;
            case DISCARD_MID -> mid;
            case DISCARD_LATE -> late;
            case THREAT, RIICHI, CALL, KAN, END, NUKI, NUKI_RESPONSE -> commit;
            case FINAL_END -> finalEnd;
          };
    }
    return rates;
  }
}
