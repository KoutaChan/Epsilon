package com.epsilon.grp.train;

import java.util.List;

/**
 * 関門の判定。
 *
 * @param kind 関門の種類
 * @param failures 落ちた理由（宣言の順）。空なら通った
 * @param versusUsed 関門のコホートでの、対応のある MSE の差（候補 − 使用中のモデル）
 * @param versusAnchor 関門のコホートでの、対応のある MSE の差（候補 − 戻したモデル）。監視でモデルを戻してから 8 反復の間で
 *     なければ null
 * @param monotoneRate 候補の Φ が逆向きに動く率
 * @param monotoneLine その率のしきい値
 */
public record GateReport(
    GateKind kind,
    List<GateFailure> failures,
    Paired versusUsed,
    Paired versusAnchor,
    double monotoneRate,
    double monotoneLine) {

  public GateReport {
    failures = List.copyOf(failures);
  }

  public boolean passed() {
    return failures.isEmpty();
  }
}
