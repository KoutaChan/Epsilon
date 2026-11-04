package com.epsilon.reviewer.api;

import com.epsilon.core.GameState;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;

/**
 * もしもの要求（protocol/commands.ts の ScenarioRequest = {resultId} &amp; ScenarioQuery）。変更を局面に
 * 照らす確かめは engine が行う。
 *
 * @param edits 変更（必須。変えなければ {}）
 * @param call 鳴いたとするチー・ポンの行動 ID。無ければ null
 */
public record ScenarioRequest(
    String resultId, int round, int eventIndex, int seat, ScenarioEdits edits, Integer call) {
  public ScenarioRequest {
    if (edits == null) throw invalid("edits is required.");
    if (round < 0 || eventIndex < 0) throw invalid("round and eventIndex must not be negative.");
    if (seat < 0 || seat >= GameState.NUM_PLAYERS) throw invalid("seat must be between 0 and 3.");
  }

  public ScenarioQuery query() {
    return new ScenarioQuery(round, eventIndex, seat, edits, call);
  }

  private static ReviewerException invalid(String message) {
    return new ReviewerException(ErrorCode.INVALID_REQUEST, message);
  }
}
