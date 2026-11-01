package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.Player;
import com.epsilon.engine.player.HeuristicPlayer;
import java.util.List;

/**
 * 規則の打ち手（{@link HeuristicPlayer}）を inline の方策として使う（物差し専用。学習データには入れない）。σ は打ち手の手の one-hot。打ち手は
 * スレッド安全でないので、進行ワーカーごとに持つ。
 */
public final class PlayerPolicy implements PolicyService {
  private final ThreadLocal<Player> players = ThreadLocal.withInitial(HeuristicPlayer::new);

  private PlayerPolicy() {}

  public static PlayerPolicy heuristic() {
    return new PlayerPolicy();
  }

  @Override
  public int version() {
    return 0;
  }

  @Override
  public boolean inline() {
    return true;
  }

  @Override
  public void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
    List<Action> legal = request.legal();
    int n = legal.size();
    Action chosen = players.get().selectAction(state, point.player(), legal);
    float[] sigma = new float[n];
    for (int a = 0; a < n; a++) if (legal.get(a).equals(chosen)) sigma[a] = 1f;
    request.sigma = sigma;
  }
}
