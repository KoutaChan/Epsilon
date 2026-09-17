package com.epsilon.client.tenhou;

import com.epsilon.core.GameState;
import com.epsilon.engine.Player;
import java.util.List;

/** 天鳳の局イベントを観測状態へ反映し、必要な自家応答を返す。 */
public final class TenhouGameController {

  private final TenhouRoundState round = new TenhouRoundState();
  private final TenhouDecisionCoordinator decisions;

  /** 選択中だけ自家の観測状態と合法手をプレイヤーに貸し出す。プレイヤーの資源は呼び出し側が所有する。 */
  public TenhouGameController(Player player) {
    decisions = new TenhouDecisionCoordinator(player);
  }

  List<String> handleEvent(TenhouEvent.GameEvent event) {
    return decisions.respond(round.apply(event), round);
  }

  /**
   * 受信済みイベントを反映した現在の観測状態を返す。
   *
   * @return 自家から見える現在局の状態
   */
  public GameState getState() {
    return round.state();
  }
}
