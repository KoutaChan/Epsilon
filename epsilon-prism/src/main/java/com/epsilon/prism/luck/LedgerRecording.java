package com.epsilon.prism.luck;

import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LedgerSettings;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * 収集で使う運の精算の設定。計算しない組の列は 0 のまま行に書く。
 *
 * @param compute 計算する組（行の帳簿の列を持つ 5 つの組から選んだ、空でない部分集合）
 * @param pointClip 点の列を切る幅（千点）
 * @param policyClip 方策の列で s を切る幅（効用。増分の絶対値はこの 2 倍まで）
 */
public record LedgerRecording(Set<LedgerGroup> compute, double pointClip, double policyClip) {

  public LedgerRecording {
    compute = groups(compute);
    if (compute.isEmpty() || !LedgerGroup.computed().containsAll(compute)) {
      throw new IllegalArgumentException(
          "A ledger computes a non-empty subset of " + LedgerGroup.computed() + ": " + compute);
    }
    if (!(pointClip > 0 && policyClip > 0 && Double.isFinite(pointClip) && Double.isFinite(policyClip))) {
      throw new IllegalArgumentException("Ledger clips must be positive and finite: " + pointClip + ", " + policyClip);
    }
  }

  /**
   * 設定の帳簿のセクションから作る。
   *
   * @return 帳簿を書かないモードでは null
   */
  public static LedgerRecording of(LedgerSettings ledger) {
    return ledger instanceof LedgerSettings.Recording recording
        ? new LedgerRecording(recording.compute(), recording.pointClip(), recording.policyClip())
        : null;
  }

  /** 組を宣言の順に並べ、変更できない集合にして返す。 */
  static Set<LedgerGroup> groups(Set<LedgerGroup> groups) {
    EnumSet<LedgerGroup> ordered = EnumSet.noneOf(LedgerGroup.class);
    ordered.addAll(groups);
    return Collections.unmodifiableSet(ordered);
  }
}
