package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineActionBuffer;
import com.epsilon.engine.MahjongTransition;
import com.epsilon.engine.PostCallDahaiRestriction;
import com.epsilon.engine.WinLegality.RonStatus;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import java.util.Arrays;
import java.util.List;

/** 再生ランタイムが所有する状態・合法手バッファ・麻雀遷移。通知先や入力形式は保持しない。 */
final class ReplayState {

  private static final int NUM_PLAYERS = GameState.NUM_PLAYERS;
  private final EngineActionBuffer turnActionsBuf = new EngineActionBuffer();
  private final EngineActionBuffer callDahaiActionsBuf = new EngineActionBuffer();
  private final EngineActionBuffer[] responseActionsByPlayer = new EngineActionBuffer[NUM_PLAYERS];
  private final ActionGenerator actionGenerator = new ActionGenerator();

  private final GameState state = new GameState();
  private final MahjongTransition transition = new MahjongTransition(state);
  private final boolean[] reachPending = new boolean[NUM_PLAYERS];
  private final int[][] decodedWalls;
  private PostCallDahaiRestriction restriction;
  private final int[] lastKnownScores = new int[NUM_PLAYERS];
  private boolean hasLastKnownScores;
  private int kyokuIndex;

  ReplayState(int[][] decodedWalls) {
    this.decodedWalls = decodedWalls;
    for (int player = 0; player < NUM_PLAYERS; player++) {
      responseActionsByPlayer[player] = new EngineActionBuffer();
    }
  }

  void startMatch() {
    for (int p = 0; p < NUM_PLAYERS; p++) {
      state.setScore(p, 25000);
    }
  }

  void applyRiichiDeclaration(Reach reach) {
    reachPending[reach.actor()] = true;
  }

  void applyRiichiPayment(ReachAccepted accepted) {
    int actor = accepted.actor();
    state.setScore(actor, state.getScore(actor) - 1000);
    state.setKyotakuCount(state.getKyotakuCount() + 1);
    if (hasLastKnownScores) {
      lastKnownScores[actor] -= 1000;
    }
  }

  void startRound(StartKyoku sk) {
    clearKanRobState();
    if (sk.scores() != null) {
      for (int p = 0; p < NUM_PLAYERS; p++) {
        lastKnownScores[p] = sk.scores()[p];
        state.setScore(p, sk.scores()[p]);
      }
      hasLastKnownScores = true;
    }
    int absoluteKyokuIndex = (sk.wind() - Tile.TON) * NUM_PLAYERS + sk.kyoku() - 1;
    state.startRoundForReconstruction(absoluteKyokuIndex, sk.oya(), sk.honba(), sk.kyotaku());
    if (decodedWalls != null && kyokuIndex < decodedWalls.length) {
      state.initializeWall(decodedWalls[kyokuIndex++]);
    } else {
      state.initializeWallForReconstruction(new int[0], 52);
      state.revealDoraIndicator(sk.doraIndicatorTileType(), sk.doraIndicatorAka());
    }
    for (int p = 0; p < NUM_PLAYERS; p++) {
      for (int physicalTileId : sk.initialHandPhysicalTileIds()[p]) {
        state.hand(p).addPhysicalTile(physicalTileId);
      }
    }
    Arrays.fill(reachPending, false);
    clearRestriction();
  }

  void applyDraw(Tsumo tsumo) {
    int physicalTileId = tsumo.physicalTileId();
    TurnEvent event = state.getTurnEvent();
    boolean rinshan =
        event instanceof TurnEvent.KanAttempt
            || event instanceof TurnEvent.Draw draw && draw.isRinshanDraw();
    transition.reconstructDraw(tsumo.actor(), physicalTileId, rinshan);
  }

  Action discardAction(Dahai discard) {
    int type = Tile.typeOf(discard.physicalTileId());
    boolean red = Tile.isAka(discard.physicalTileId());
    return reachPending[discard.actor()]
        ? Action.riichiDahai(type, red, discard.tsumogiri())
        : Action.dahai(type, red, discard.tsumogiri());
  }

  List<Action> discardActions(int player) {
    if (isCallDahai()) {
      ActionGenerator.generateCallDahaiActionsInto(
          callDahaiActionsBuf, state.hand(player), restriction);
      return callDahaiActionsBuf;
    }
    return turnActions(player);
  }

  List<Action> turnActions(int player) {
    actionGenerator.generateTurnActionsInto(turnActionsBuf, state, player, currentDraw());
    return turnActionsBuf;
  }

  void applyDiscard(Dahai discard, Action action) {
    reachPending[discard.actor()] = false;
    transition.reconstructDiscard(discard.actor(), action, !isCallDahai());
    clearRestriction();
  }

  static Action chiAction(Chi chi) {
    int calledType = Tile.typeOf(chi.calledPhysicalTileId());
    int[] consumed = chi.consumedPhysicalTileIds();
    int baseType =
        Math.min(calledType, Math.min(Tile.typeOf(consumed[0]), Tile.typeOf(consumed[1])));
    return Action.chiSequence(baseType, calledType, consumedTilesContainAka(consumed));
  }

  static Action ponAction(Pon pon) {
    return Action.pon(
        Tile.typeOf(pon.calledPhysicalTileId()),
        consumedTilesContainAka(pon.consumedPhysicalTileIds()));
  }

  void applyClosedKan(Ankan kan) {
    transition.completeAnkan(kan.actor(), Tile.typeOf(kan.consumedPhysicalTileIds()[0]));
    afterSelfKan(kan.actor());
  }

  void applyAddedKan(Kakan kan) {
    int tileType = Tile.typeOf(kan.addedPhysicalTileId());
    transition.declareKakan(kan.actor(), tileType);
    prepareKanRobState(kan.actor(), tileType, Tile.isAka(kan.addedPhysicalTileId()));
  }

  void collectResponses(ResponseWindow window) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
      if (player == window.sourcePlayer()) continue;
      List<Action> legal = responseActionsByPlayer[player];
      RonStatus status =
          actionGenerator.generateResponseActionsInto(
              legal, state, player, (TurnEvent.ResponseSource) state.getTurnEvent());
      if (status.createsImmediateFuriten()) state.enterTemporaryFuriten(player);
      window.put(player, legal, status);
    }
  }

  void applyChi(Chi chi) {
    int calledTileType = Tile.typeOf(chi.calledPhysicalTileId());
    int[] consumedPhysicalTileIds = chi.consumedPhysicalTileIds();
    int sequenceBaseTileType =
        Math.min(
            calledTileType,
            Math.min(
                Tile.typeOf(consumedPhysicalTileIds[0]), Tile.typeOf(consumedPhysicalTileIds[1])));
    transition.applyChi(
        chi.actor(),
        (TurnEvent.Discard) state.getTurnEvent(),
        Action.chiSequence(
            sequenceBaseTileType,
            calledTileType,
            consumedTilesContainAka(consumedPhysicalTileIds)));
    afterCall(chi.actor(), PostCallDahaiRestriction.afterChi(sequenceBaseTileType, calledTileType));
  }

  void applyPon(Pon pon) {
    int tileType = Tile.typeOf(pon.calledPhysicalTileId());
    int[] consumedPhysicalTileIds = pon.consumedPhysicalTileIds();
    transition.applyPon(
        pon.actor(),
        (TurnEvent.Discard) state.getTurnEvent(),
        consumedTilesContainAka(consumedPhysicalTileIds));
    afterCall(pon.actor(), PostCallDahaiRestriction.afterPon(tileType));
  }

  void applyOpenKan(Daiminkan kan) {
    transition.completeDaiminkan(kan.actor(), (TurnEvent.Discard) state.getTurnEvent());
    transition.beginReconstructionRinshan(kan.actor());
    clearRestriction();
  }

  void revealDora(Dora dora) {
    state.revealDoraIndicator(dora.indicatorTileType(), dora.indicatorAka());
  }

  void settleWin(Hora hora) {
    applyDeltas(hora.deltas());
    state.setKyotakuCount(0);
  }

  void settleDraw(Ryukyoku ryukyoku) {
    applyDeltas(ryukyoku.deltas());
  }

  void finishRound() {
    clearKanRobState();
  }

  void finishMatch(EndGame endGame) {
    if (endGame.scores() != null) {
      for (int player = 0; player < NUM_PLAYERS; player++) {
        lastKnownScores[player] = endGame.scores()[player];
        state.setScore(player, endGame.scores()[player]);
      }
      hasLastKnownScores = true;
    }
  }

  boolean enterTemporaryFuriten(int player) {
    boolean changed = !state.isTemporaryFuriten(player);
    state.enterTemporaryFuriten(player);
    return changed;
  }

  boolean completeResponses(ResponseWindow window, boolean robbedByRon) {
    if (window.isKakanResponse()) {
      if (robbedByRon) {
        clearKanRobState();
      } else {
        completeAddedKan(window);
      }
      return true;
    }
    return false;
  }

  private void completeAddedKan(ResponseWindow window) {
    clearKanRobState();
    afterSelfKan(window.sourcePlayer());
  }

  private void clearKanRobState() {
    if (state.getTurnEvent() instanceof TurnEvent.KanAttempt) {
      state.clearTurnEvent();
    }
  }

  private void prepareKanRobState(int sourcePlayer, int tileType, boolean isAkaTile) {
    state.recordKanAttempt(sourcePlayer, tileType, isAkaTile, TurnEvent.KanKind.KAKAN);
  }

  private TurnEvent.Draw currentDraw() {
    return (TurnEvent.Draw) state.getTurnEvent();
  }

  private void applyDeltas(int[] deltas) {
    if (deltas == null) {
      return;
    }
    if (!hasLastKnownScores) {
      for (int p = 0; p < NUM_PLAYERS; p++) {
        lastKnownScores[p] = state.getScore(p);
      }
      hasLastKnownScores = true;
    }
    for (int p = 0; p < Math.min(deltas.length, NUM_PLAYERS); p++) {
      lastKnownScores[p] += deltas[p];
      state.setScore(p, lastKnownScores[p]);
    }
  }

  private void afterCall(int actor, PostCallDahaiRestriction restriction) {
    state.clearAllIppatsu();
    state.setCurrentPlayer(actor);
    state.clearTurnEvent();
    this.restriction = restriction;
  }

  private void afterSelfKan(int actor) {
    transition.beginReconstructionRinshan(actor);
    clearRestriction();
  }

  private boolean isCallDahai() {
    return restriction != null;
  }

  private void clearRestriction() {
    restriction = null;
  }

  GameState borrowedState() {
    return state;
  }

  int[] finalScores() {
    if (!hasLastKnownScores) {
      for (int player = 0; player < NUM_PLAYERS; player++) {
        lastKnownScores[player] = state.getScore(player);
      }
      hasLastKnownScores = true;
    }
    return lastKnownScores;
  }

  private static boolean consumedTilesContainAka(int... physicalTileIds) {
    for (int physicalTileId : physicalTileIds) {
      if (Tile.isAka(physicalTileId)) {
        return true;
      }
    }
    return false;
  }
}
