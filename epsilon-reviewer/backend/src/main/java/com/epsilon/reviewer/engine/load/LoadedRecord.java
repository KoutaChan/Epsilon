package com.epsilon.reviewer.engine.load;

import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayRecord;
import java.util.List;

/**
 * 読み込んだ牌譜。
 *
 * @param roundStarts 局ごとの StartKyoku の出来事の位置（読む時に 1 回だけ数える）
 */
public record LoadedRecord(RecordSummary summary, ReplayRecord replay, int[] roundStarts) {
  /** 局 round の終わりの次の出来事の位置（次の局の始めか牌譜の終わり）。 */
  public int roundEnd(int round) {
    return round + 1 < roundStarts.length ? roundStarts[round + 1] : replay.events().size();
  }

  /** 出来事の列の局の始めの位置。 */
  public static int[] roundStarts(List<ReplayEvent> events) {
    int rounds = 0;
    for (ReplayEvent event : events) if (event instanceof StartKyoku) rounds++;
    int[] starts = new int[rounds];
    int round = 0;
    for (int index = 0; index < events.size(); index++)
      if (events.get(index) instanceof StartKyoku) starts[round++] = index;
    return starts;
  }
}
