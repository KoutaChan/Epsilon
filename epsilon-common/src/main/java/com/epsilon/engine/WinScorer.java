package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.TurnEvent;

/**
 * 局面の和了を、実対局と同じ和了・精算の規則（{@link WinLegality} と {@code WinSettlementCalculator}）で採点する。持ち主が使い回し、
 * 複数のスレッドから同時に使えない。
 */
public final class WinScorer {
  private final HandScoreEvaluator scores = new HandScoreEvaluator();
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandScoreBuffer score = new HandScoreBuffer();

  /**
   * winner の和了を採点し精算する。手番の出来事が winner のツモならツモ和了、他家の打牌か加槓ならロン。裏ドラは局面で分かっている表示牌の分だけ
   * 数え、点の移動に供託は入れない。
   *
   * @param honba この和了が受け取る本場（同じ打牌への 2 人目以降のロンは 0）
   * @return 和了の明細。和了の形でないか、役が無いか、振聴なら null
   */
  public WinClaim claim(GameState state, int winner, int honba) {
    Hand hand = state.hand(winner);
    boolean riichi = state.isRiichi(winner);
    TurnEvent event = state.getTurnEvent();
    if (event instanceof TurnEvent.Draw draw && draw.player() == winner) {
      return WinLegality.evaluateTsumoInto(state, winner, draw, score, scores, shapes)
          ? WinSettlementCalculator.createTsumoClaim(
              winner, state.getOya(), honba, score, hand, riichi)
          : null;
    }
    if (event instanceof TurnEvent.ResponseSource source && source.player() != winner) {
      return WinLegality.evaluateRonInto(state, winner, source, score, scores, shapes).canRon()
          ? WinSettlementCalculator.createRonClaim(
              winner, source.player(), state.getOya(), honba, score, hand, riichi)
          : null;
    }
    return null;
  }
}
