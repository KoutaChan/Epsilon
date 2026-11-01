package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.Arena;
import java.util.Arrays;

/**
 * 収集の率。評価（{@link Arena}）は {@link #OFF}（行も写しも作らない）。
 *
 * @param holdoutFraction 保留（HOLDOUT）にする半荘の割合
 * @param labelRateDiscardEarly 脅威のない序盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardMid 脅威のない中盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardLate 脅威のない終盤の打牌のラベル行の率 ρ
 * @param labelRateCommit 確約の類（{@link DecisionClass#commitment()}）のラベル行の率 ρ（{@link DecisionClass#FINAL_END} のほか）
 * @param labelRateFinalEnd オーラスの和了（{@link DecisionClass#FINAL_END}）のラベル行の率 ρ
 * @param maxCandidates 候補集合 C(o) の大きさの上限（双子をまとめた後）
 * @param measureAllDiscards 打牌の類（脅威のない序盤・中盤・終盤）では候補集合を全ての合法手（双子をまとめた代表）にするか
 * @param epsilonC 候補集合に一様な 1 手を足す確率 ε_C
 * @param rollinRateDiscard 打牌の類の中立ロールインの率 β
 * @param rollinRateCommit 確約の類の中立ロールインの率 β
 * @param epsilonQ 逸脱の分布 q の一様な床 ε_q
 * @param rhoBranch 写しの中の逸脱した席の判断をラベル行にする率 ρ_br
 * @param maxCopyLabels 写しごとのラベル行の上限
 * @param rollinOverrides 類（{@link DecisionClass} の番号）ごとの中立ロールインの率の上書き（NaN の類は rollinRateDiscard か
 *     rollinRateCommit。続きの罠が出た鳴きの類だけ写しを厚くするため）
 * @param eventLedger 分岐と主経路のラベルの枠ごとに運の精算の帳簿（{@link EventLedger}）を付けて行に書くか（false なら行は帳簿なし）
 */
public record Sampling(
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
    int maxCopyLabels,
    double[] rollinOverrides,
    boolean eventLedger) {
  /** 行も写しも作らない（評価）。 */
  public static final Sampling OFF = new Sampling(0, 0, 0, 0, 0, 0, 2, false, 0, 0, 0, 0, 0, 0);

  public Sampling {
    if (!(holdoutFraction >= 0 && holdoutFraction < 1))
      throw new IllegalArgumentException("holdoutFraction " + holdoutFraction);
    for (double rate :
        new double[] {
          labelRateDiscardEarly,
          labelRateDiscardMid,
          labelRateDiscardLate,
          labelRateCommit,
          labelRateFinalEnd,
          epsilonC,
          rollinRateDiscard,
          rollinRateCommit,
          epsilonQ,
          rhoBranch
        })
      if (!(rate >= 0 && rate <= 1)) throw new IllegalArgumentException("Invalid rate " + rate);
    if (maxCandidates < 2 || maxCandidates > 32 || maxCopyLabels < 0)
      throw new IllegalArgumentException("Invalid candidate or copy limits");
    if (rollinOverrides.length != DecisionClass.COUNT)
      throw new IllegalArgumentException("rollinOverrides needs one rate per class");
    for (double rate : rollinOverrides)
      if (!Double.isNaN(rate) && !(rate >= 0 && rate <= 1))
        throw new IllegalArgumentException("Invalid roll-in rate " + rate);
    rollinOverrides = rollinOverrides.clone();
  }

  /** 類ごとの上書きがなく、帳簿を付けない率。 */
  public Sampling(
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
    this(
        holdoutFraction,
        labelRateDiscardEarly,
        labelRateDiscardMid,
        labelRateDiscardLate,
        labelRateCommit,
        labelRateFinalEnd,
        maxCandidates,
        measureAllDiscards,
        epsilonC,
        rollinRateDiscard,
        rollinRateCommit,
        epsilonQ,
        rhoBranch,
        maxCopyLabels,
        noOverrides(),
        false);
  }

  /** この率の、運の精算の帳簿を付けるかだけを替えた写し。 */
  public Sampling withEventLedger(boolean eventLedger) {
    return new Sampling(
        holdoutFraction,
        labelRateDiscardEarly,
        labelRateDiscardMid,
        labelRateDiscardLate,
        labelRateCommit,
        labelRateFinalEnd,
        maxCandidates,
        measureAllDiscards,
        epsilonC,
        rollinRateDiscard,
        rollinRateCommit,
        epsilonQ,
        rhoBranch,
        maxCopyLabels,
        rollinOverrides,
        eventLedger);
  }

  /** 全ての類で上書きしない表（NaN）。 */
  public static double[] noOverrides() {
    double[] rates = new double[DecisionClass.COUNT];
    Arrays.fill(rates, Double.NaN);
    return rates;
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
      case THREAT, RIICHI, CALL, KAN, END -> labelRateCommit;
      case FINAL_END -> labelRateFinalEnd;
    };
  }

  /** 類 c の中立ロールインの率 β_c。 */
  public double rollinRate(DecisionClass c) {
    double override = rollinOverrides[c.ordinal()];
    if (!Double.isNaN(override)) return override;
    return c.commitment() ? rollinRateCommit : rollinRateDiscard;
  }
}
