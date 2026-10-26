package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;

/**
 * 収集の率（PRISM-min の M1・M2）。評価（{@link com.epsilon.prism.eval.Arena}）は {@link #OFF}（行も写しも作らない）。
 *
 * @param holdoutFraction 保留（HOLDOUT）にする半荘の割合
 * @param labelRateDiscardEarly 脅威のない序盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardMid 脅威のない中盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardLate 脅威のない終盤の打牌のラベル行の率 ρ
 * @param labelRateCommit 確約の類（{@link DecisionClass#commitment()}）のラベル行の率 ρ
 * @param maxCandidates 候補集合 C(o) の大きさの上限（双子をまとめた後）
 * @param measureAllDiscards 打牌の類（脅威のない序盤・中盤・終盤）では候補集合を全ての合法手（双子をまとめた代表）にするか
 * @param epsilonC 候補集合に一様な 1 手を足す確率 ε_C
 * @param rollinRateDiscard 打牌の類の中立ロールインの率 β
 * @param rollinRateCommit 確約の類の中立ロールインの率 β
 * @param epsilonQ 逸脱の分布 q の一様な床 ε_q
 * @param rhoBranch 写しの中の逸脱した席の判断をラベル行にする率 ρ_br
 * @param maxCopyLabels 写しごとのラベル行の上限
 */
public record Sampling(
    double holdoutFraction,
    double labelRateDiscardEarly,
    double labelRateDiscardMid,
    double labelRateDiscardLate,
    double labelRateCommit,
    int maxCandidates,
    boolean measureAllDiscards,
    double epsilonC,
    double rollinRateDiscard,
    double rollinRateCommit,
    double epsilonQ,
    double rhoBranch,
    int maxCopyLabels) {
  /** 行も写しも作らない（評価）。 */
  public static final Sampling OFF = new Sampling(0, 0, 0, 0, 0, 2, false, 0, 0, 0, 0, 0, 0);

  public Sampling {
    if (!(holdoutFraction >= 0 && holdoutFraction < 1))
      throw new IllegalArgumentException("holdoutFraction " + holdoutFraction);
    for (double rate :
        new double[] {
          labelRateDiscardEarly,
          labelRateDiscardMid,
          labelRateDiscardLate,
          labelRateCommit,
          epsilonC,
          rollinRateDiscard,
          rollinRateCommit,
          epsilonQ,
          rhoBranch
        })
      if (!(rate >= 0 && rate <= 1)) throw new IllegalArgumentException("Invalid rate " + rate);
    if (maxCandidates < 2 || maxCandidates > 32 || maxCopyLabels < 0)
      throw new IllegalArgumentException("Invalid candidate or copy limits");
  }

  /** 類 c の判断で全ての合法手を測るか（measureAllDiscards のときの打牌の類）。 */
  public boolean measuresAll(DecisionClass c) {
    return measureAllDiscards && !c.commitment();
  }

  /** 類 c のラベル行の率 ρ_c。 */
  public double labelRate(DecisionClass c) {
    return switch (c) {
      case DISCARD_EARLY -> labelRateDiscardEarly;
      case DISCARD_MID -> labelRateDiscardMid;
      case DISCARD_LATE -> labelRateDiscardLate;
      default -> labelRateCommit;
    };
  }

  /** 類 c の中立ロールインの率 β_c。 */
  public double rollinRate(DecisionClass c) {
    return c.commitment() ? rollinRateCommit : rollinRateDiscard;
  }
}
