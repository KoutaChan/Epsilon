package com.epsilon.prism.eval;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.RoundResult;
import com.epsilon.prism.collect.CollectionDriver;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * 対局の内訳（診断用）。候補の席（{@code Math.floorMod(gameId, 4)}）と相手の席を分けて、局の結果と判断の傾向を数える。
 *
 * <p>判断の傾向は、他家が立直しているときの打牌が全立直者の現物か（現物があったときに切ったか）、鳴ける場面で鳴いたか、立直できる場面で立直したか。
 */
public final class PlayStats implements CollectionDriver.GameObserver {
  private static final int CANDIDATE = 0, OPPONENT = 1;
  private final Side[] sides = {new Side(), new Side()};

  private static final class Side {
    final LongAdder hands = new LongAdder(), wins = new LongAdder(), tsumo = new LongAdder();
    final LongAdder dealIns = new LongAdder(), dealInsToRiichi = new LongAdder();
    final LongAdder riichiHands = new LongAdder(), openHands = new LongAdder();
    final LongAdder draws = new LongAdder(), tenpaiAtDraw = new LongAdder();
    final LongAdder winPoints = new LongAdder(), dealInPoints = new LongAdder();
    final LongAdder netPoints = new LongAdder();
    final LongAdder threatDiscards = new LongAdder(), threatWithSafe = new LongAdder();
    final LongAdder threatChoseSafe = new LongAdder();
    final LongAdder callChances = new LongAdder(), callsTaken = new LongAdder();
    final LongAdder riichiChances = new LongAdder(), riichiTaken = new LongAdder();
  }

  private static int side(long gameId, int seat) {
    return seat == Math.floorMod(gameId, 4) ? CANDIDATE : OPPONENT;
  }

  @Override
  public void gameEnded(long gameId, int[] finalScores) {}

  @Override
  public void decided(long gameId, GameState state, EngineDecisionPoint point, Action action) {
    var legal = point.legalActions();
    if (legal.size() < 2) return;
    Side side = sides[side(gameId, point.player())];
    boolean canCall = false, canRiichi = false, canDiscard = false;
    for (Action candidate : legal) {
      switch (candidate.type()) {
        case CHI, PON -> canCall = true;
        case RIICHI_DAHAI -> canRiichi = true;
        case DAHAI -> canDiscard = true;
        default -> {}
      }
    }
    if (canCall) {
      side.callChances.increment();
      if (action.type() == Action.Type.CHI || action.type() == Action.Type.PON)
        side.callsTaken.increment();
    }
    if (canRiichi) {
      side.riichiChances.increment();
      if (action.type() == Action.Type.RIICHI_DAHAI) side.riichiTaken.increment();
    }
    int player = point.player();
    if (!canDiscard || state.isRiichi(player)) return;
    int threats = 0;
    for (int seat = 0; seat < 4; seat++) if (seat != player && state.isRiichi(seat)) threats |= 1 << seat;
    if (threats == 0) return;
    side.threatDiscards.increment();
    boolean safeAvailable = false;
    for (Action candidate : legal)
      if (candidate.type() == Action.Type.DAHAI && safe(state, threats, candidate.tileType()))
        safeAvailable = true;
    if (!safeAvailable) return;
    side.threatWithSafe.increment();
    if ((action.type() == Action.Type.DAHAI || action.type() == Action.Type.RIICHI_DAHAI)
        && safe(state, threats, action.tileType())) side.threatChoseSafe.increment();
  }

  /** 全立直者の現物（その立直者の河にある牌種）。 */
  private static boolean safe(GameState state, int threats, int tileType) {
    for (int seat = 0; seat < 4; seat++)
      if ((threats & (1 << seat)) != 0 && !state.river(seat).contains(tileType)) return false;
    return true;
  }

  @Override
  public void handEnded(long gameId, GameState state, RoundResult result) {
    int[] delta = result.pointDelta().toArray();
    for (int seat = 0; seat < 4; seat++) {
      Side side = sides[side(gameId, seat)];
      side.hands.increment();
      side.netPoints.add(delta[seat]);
      if (state.isRiichi(seat)) side.riichiHands.increment();
      var hand = state.hand(seat);
      for (int meld = 0; meld < hand.meldCount(); meld++)
        if (hand.meld(meld).type() != Meld.Type.ANKAN) {
          side.openHands.increment();
          break;
        }
    }
    switch (result) {
      case RoundResult.TsumoAgari tsumo -> {
        Side side = sides[side(gameId, tsumo.winner())];
        side.wins.increment();
        side.tsumo.increment();
        side.winPoints.add(delta[tsumo.winner()]);
      }
      case RoundResult.RonAgari ron -> {
        int loser = ron.loser();
        Side loserSide = sides[side(gameId, loser)];
        loserSide.dealIns.increment();
        loserSide.dealInPoints.add(-delta[loser]);
        boolean toRiichi = false;
        for (int seat = 0; seat < 4; seat++) {
          if (seat == loser || !ron.hasWinner(seat)) continue;
          Side side = sides[side(gameId, seat)];
          side.wins.increment();
          side.winPoints.add(delta[seat]);
          toRiichi |= state.isRiichi(seat);
        }
        if (toRiichi) loserSide.dealInsToRiichi.increment();
      }
      case RoundResult.ExhaustiveDraw draw -> {
        for (int seat = 0; seat < 4; seat++) {
          Side side = sides[side(gameId, seat)];
          side.draws.increment();
          if (draw.isTenpai(seat)) side.tenpaiAtDraw.increment();
        }
      }
      case RoundResult.AbortiveDraw ignored -> {}
    }
  }

  /** 席あたりの率。相手は3席分なので、どちらも「1席・1局あたり」にそろえる。 */
  public Map<String, Object> report() {
    var result = new LinkedHashMap<String, Object>();
    String[] names = {"candidate", "opponents"};
    for (int s = 0; s < 2; s++) {
      Side side = sides[s];
      double hands = Math.max(1, side.hands.sum());
      var row = new LinkedHashMap<String, Object>();
      row.put("seatHands", side.hands.sum());
      row.put("winRate", side.wins.sum() / hands);
      row.put("tsumoShareOfWins", side.tsumo.sum() / (double) Math.max(1, side.wins.sum()));
      row.put("dealInRate", side.dealIns.sum() / hands);
      row.put("dealInToRiichiRate", side.dealInsToRiichi.sum() / hands);
      row.put("riichiRate", side.riichiHands.sum() / hands);
      row.put("openHandRate", side.openHands.sum() / hands);
      row.put("tenpaiAtDraw", side.tenpaiAtDraw.sum() / (double) Math.max(1, side.draws.sum()));
      row.put("meanWinPoints", side.winPoints.sum() / (double) Math.max(1, side.wins.sum()));
      row.put("meanDealInPoints", side.dealInPoints.sum() / (double) Math.max(1, side.dealIns.sum()));
      row.put("netPointsPerHand", side.netPoints.sum() / hands);
      row.put("threatDiscards", side.threatDiscards.sum());
      row.put(
          "safeChosenWhenAvailable",
          side.threatChoseSafe.sum() / (double) Math.max(1, side.threatWithSafe.sum()));
      row.put(
          "safeAvailableUnderThreat",
          side.threatWithSafe.sum() / (double) Math.max(1, side.threatDiscards.sum()));
      row.put("callTakeRate", side.callsTaken.sum() / (double) Math.max(1, side.callChances.sum()));
      row.put(
          "riichiTakeRate", side.riichiTaken.sum() / (double) Math.max(1, side.riichiChances.sum()));
      result.put(names[s], row);
    }
    return result;
  }
}
