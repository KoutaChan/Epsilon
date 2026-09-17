package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import java.util.List;

/** 再生の判断とイベント適用を同期で受け取る。状態と合法手は通知中だけ読み取り専用で借用する。 */
@FunctionalInterface
public interface ReplayObserver {
  /** 適用前の判断を受け取る。必要な特徴量・表示値だけをここで生成し、借用値を保持・変更・非同期転送しない。 全席の手牌を含む状態を方策へ渡さず、推論には当該席の公開観測を使う。 */
  void onDecision(DecisionPoint point, GameState state, int player, List<Action> legalActions);

  /** 一つの元イベントを適用した状態。判断と同じstateIdなら同じ更新時点の状態を表す。 */
  default void onEventApplied(int eventIndex, int stateId, ReplayEvent event, GameState state) {}
}
