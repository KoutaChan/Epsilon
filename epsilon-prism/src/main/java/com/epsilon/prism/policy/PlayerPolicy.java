package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.Player;
import com.epsilon.engine.player.HeuristicPlayer;
import java.util.List;

/**
 * 規則の打ち手（{@link HeuristicPlayer}）を inline の方策として使う（物差し専用。学習データには入れない）。打ち手はスレッド安全でないので、
 * 進行ワーカーごとに持つ。
 *
 * <p>q &gt; 0 なら M_q = (1−q)·δ_player + q·一様。
 */
public final class PlayerPolicy implements PolicyService {
  private final ThreadLocal<Player> players = ThreadLocal.withInitial(HeuristicPlayer::new);
  private final double uniformMix;

  private PlayerPolicy(double uniformMix) {
    if (!(uniformMix >= 0 && uniformMix <= 1)) throw new IllegalArgumentException("q");
    this.uniformMix = uniformMix;
  }

  public static PlayerPolicy heuristic(double uniformMix) {
    return new PlayerPolicy(uniformMix);
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
    float uniform = (float) (uniformMix / n);
    for (int a = 0; a < n; a++) {
      sigma[a] = uniform;
      if (legal.get(a).equals(chosen)) sigma[a] += (float) (1 - uniformMix);
    }
    request.sigma = sigma;
  }
}
