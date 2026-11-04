package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.GameState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.List;

/**
 * 判断の局面に当てる変更。点数・本場・供託は判断の局面で見えている値で、省いた欄は元のまま（配列は空、数は null）。
 *
 * @param scores 4 席の点数。空なら変えない
 */
public record ScenarioEdits(
    List<RiverEdit> rivers,
    List<DoraEdit> dora,
    List<HandEdit> hand,
    List<Integer> scores,
    Integer honba,
    Integer kyotaku) {
  public ScenarioEdits {
    rivers = rivers == null ? List.of() : rivers;
    dora = dora == null ? List.of() : dora;
    hand = hand == null ? List.of() : hand;
    scores = scores == null ? List.of() : scores;
    if (!scores.isEmpty() && scores.size() != GameState.NUM_PLAYERS)
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST, "edits.scores must contain four scores.");
  }
}
