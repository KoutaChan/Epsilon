package com.epsilon.reviewer.api;

import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;

/**
 * もしもの要求（protocol/commands.ts の ScenarioRequest = {resultId} &amp; ScenarioQuery）。読むときは、変更が渡されたか、局とイベントの位置が負でないか、
 * 席が 4 人の卓に収まるかだけを確かめる。牌譜の人数と合うか、変更が局面に合うかは engine が確かめる。
 *
 * @param edits 加える変更。必須で、何も変えないときは {} を渡す
 * @param call 鳴いたことにするチー・ポンの行動 ID。無ければ null
 */
public record ScenarioRequest(
    String resultId, int round, int eventIndex, int seat, ScenarioEdits edits, Integer call) {
  /** 読むときに受け付ける席の数。席の番号はこれより小さくなければならない。 */
  private static final int MAX_SEATS = 4;

  public ScenarioRequest {
    if (edits == null) throw invalid("edits is required.");
    if (round < 0 || eventIndex < 0) throw invalid("round and eventIndex must not be negative.");
    if (seat < 0 || seat >= MAX_SEATS)
      throw invalid("seat must be between 0 and " + (MAX_SEATS - 1) + ".");
  }

  public ScenarioQuery query() {
    return new ScenarioQuery(round, eventIndex, seat, edits, call);
  }

  private static ReviewerException invalid(String message) {
    return new ReviewerException(ErrorCode.INVALID_REQUEST, message);
  }
}
