package com.epsilon.prism.grp;

import java.util.List;

/**
 * 関門の入力（{@link GateEvaluator#evaluate}）。使用中の Φ は使っているとおり（温度 → 表）、候補は表なしで評価する。
 *
 * @param mode 関門の表の列
 * @param holdout 取り置きのコホートの半荘（学習に使っていない最新の封印済みのコホート）
 * @param active 使用中の Φ
 * @param candidate 候補の Φ
 * @param twin 半荘の偶奇のもう半分で学んだ候補（FIRST_REPLACEMENT の安定の関門だけ。REFIT では null）
 * @param candidateValid 候補の checkpoint が {@code EpsilonGrpValidationGate.structurallyValid} で構造が grp-epsilon-v3 なら true
 * @param windowGames 学習の窓の半荘の数
 * @param trustGames 最初の置き換えに要る窓の半荘の数（grpTrustGames）
 */
public record GateInput(
    GateMode mode,
    List<RecordedGame> holdout,
    HandEndPhi active,
    HandEndPhi candidate,
    HandEndPhi twin,
    boolean candidateValid,
    long windowGames,
    int trustGames) {}
