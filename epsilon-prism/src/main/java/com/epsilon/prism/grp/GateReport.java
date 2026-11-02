package com.epsilon.prism.grp;

import java.util.List;
import java.util.Map;

/**
 * 関門の報告（refits.jsonl の gate）。
 *
 * @param mode 関門の表の列
 * @param result PASS か、落ちた関門のうち表の順で最初のもの
 * @param failures 落ちた全ての関門（表の順。PASS なら空）
 * @param measures 関門の数値（名前 → 値。対の MSE の差と SE・NLL・ECE・ΔΦ・偶奇の RMS・逆に動く率・傾きの比・分岐の見本など）
 * @param active 取り置きのコホートの使用中の Φ の較正の測り
 * @param candidate 同じく候補の測り
 */
public record GateReport(
    GateMode mode,
    GateResult result,
    List<GateResult> failures,
    Map<String, Double> measures,
    CellReport active,
    CellReport candidate) {

  /** 合格したなら true。 */
  public boolean passed() {
    return result == GateResult.PASS;
  }
}
