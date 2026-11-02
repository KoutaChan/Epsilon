package com.epsilon.prism.play;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.Player;
import com.epsilon.prism.core.Greedy;
import java.util.List;

/**
 * 公開した版の s（{@link SnapshotScorer}、牌譜解析と同じ値）の貪欲で打つ。収集の ε の床は使わない。同点は番号の小さい手（{@link Greedy}。学んだ網で
 * s が同じになることはまずない）。
 *
 * @param scorer 版の採点（呼び出し元が持ち、閉じる）
 */
public record GreedyPlayer(SnapshotScorer scorer) implements Player {
  @Override
  public Action selectAction(GameState state, int playerIndex, List<Action> legalActions) {
    float[] scores = scorer.scores(new PublicObservation(state, playerIndex), legalActions);
    return legalActions.get(Greedy.argmax(scores, null, legalActions.size()));
  }
}
