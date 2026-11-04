package com.epsilon.reviewer.engine.view;

import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Ryukyoku;

/** 局の結果。和了の牌は適用の直後に残る手番の出来事（ツモ・打牌・槍槓の加槓）の物理牌から取り、赤と複数のロンを保つ。 */
public final class OutcomeProjector {
  private OutcomeProjector() {}

  /** hora・ryukyoku でない出来事は null。 */
  public static RoundOutcome of(ReplayEvent event, GameState state) {
    return switch (event) {
      case Hora hora -> {
        var source = (TurnEvent.TileEvent) state.turnEvent();
        yield new RoundOutcome.Hora(
            hora.actor(),
            hora.target(),
            TileCode.of(source.tileType(), source.isAkaTile()),
            hora.details());
      }
      case Ryukyoku ignored -> new RoundOutcome.Ryukyoku();
      default -> null;
    };
  }
}
