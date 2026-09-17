package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;

/** 和了時の点棒移動を計算する。 */
final class WinSettlementCalculator {

  private WinSettlementCalculator() {}

  /** 候補の評価や入力の符号化に使うツモ和了の点棒移動を、和了明細や支払いオブジェクトを生成せずに書き込む。 */
  static void tsumoDeltaInto(
      int winner,
      int oya,
      int honba,
      HandScoreBuffer agari,
      HandView winnerHand,
      PointDeltaBuffer out) {
    int bonus = honba * 100;
    int base = agari.basePoints();
    int fromDealer;
    int fromChild;
    if (winner == oya) {
      fromDealer = 0;
      fromChild = ScorePayments.tsumoFromDealer(base) + bonus;
    } else {
      fromDealer = ScorePayments.tsumoFromDealer(base) + bonus;
      fromChild = ScorePayments.tsumoFromChild(base, false) + bonus;
    }
    int total = winner == oya ? fromChild * 3 : fromDealer + fromChild * 2;
    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    int pao = paoPlayer(winner, winnerHand, agari);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int delta;
      if (seat == winner) {
        delta = total;
      } else if (pao >= 0) {
        delta = seat == pao ? -total : 0;
      } else {
        delta = -(winner == oya ? fromChild : seat == oya ? fromDealer : fromChild);
      }
      if (seat == 0) d0 = delta;
      else if (seat == 1) d1 = delta;
      else if (seat == 2) d2 = delta;
      else d3 = delta;
    }
    out.bind(d0, d1, d2, d3);
  }

  /** 候補の評価や入力の符号化に使うロン和了の点棒移動を、和了明細や支払いオブジェクトを生成せずに書き込む。 */
  static void ronDeltaInto(
      int winner,
      int loser,
      int oya,
      int honba,
      HandScoreBuffer agari,
      HandView winnerHand,
      PointDeltaBuffer out) {
    int basePayment = ScorePayments.ronPoints(agari.basePoints(), winner == oya);
    int points = basePayment + honba * 300;
    int pao = paoPlayer(winner, winnerHand, agari);
    int d0 = 0;
    int d1 = 0;
    int d2 = 0;
    int d3 = 0;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int delta = 0;
      if (seat == winner) {
        delta = points;
      } else if (pao < 0 || pao == loser) {
        delta = seat == loser ? -points : 0;
      } else {
        int half = basePayment / 2;
        if (seat == pao) delta = -half - honba * 300;
        else if (seat == loser) delta = -half;
      }
      if (seat == 0) d0 = delta;
      else if (seat == 1) d1 = delta;
      else if (seat == 2) d2 = delta;
      else d3 = delta;
    }
    out.bind(d0, d1, d2, d3);
  }

  static WinClaim.Tsumo tsumo(
      int winner,
      int oya,
      int honba,
      HandScoreBuffer agari,
      HandView winnerHand,
      boolean riichiDeclared) {
    WinPayment.Tsumo payment =
        tsumoPayment(
            agari.basePoints(), winner == oya ? WinnerRole.DEALER : WinnerRole.CHILD, honba);
    int player0 = 0;
    int player1 = 0;
    int player2 = 0;
    int player3 = 0;
    int pao = paoPlayer(winner, winnerHand, agari);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      int delta;
      if (player == winner) {
        delta = payment.total();
      } else if (pao >= 0) {
        delta = player == pao ? -payment.total() : 0;
      } else {
        delta =
            switch (payment) {
              case WinPayment.DealerTsumo tsumo -> -tsumo.each();
              case WinPayment.ChildTsumo tsumo ->
                  -(player == oya ? tsumo.fromDealer() : tsumo.fromChild());
            };
      }
      switch (player) {
        case 0 -> player0 = delta;
        case 1 -> player1 = delta;
        case 2 -> player2 = delta;
        default -> player3 = delta;
      }
    }
    return new WinClaim.Tsumo(
        winner,
        agari.snapshot(),
        payment,
        honba,
        new PointDelta(player0, player1, player2, player3),
        riichiDeclared);
  }

  static WinPayment.Tsumo tsumoPayment(int basePoints, WinnerRole role, int honba) {
    WinPayment.Tsumo payment = ScorePayments.tsumo(basePoints, role);
    if (honba == 0) {
      return payment;
    }
    int bonus = honba * 100;
    return switch (payment) {
      case WinPayment.DealerTsumo tsumo -> new WinPayment.DealerTsumo(tsumo.each() + bonus);
      case WinPayment.ChildTsumo tsumo ->
          new WinPayment.ChildTsumo(tsumo.fromDealer() + bonus, tsumo.fromChild() + bonus);
    };
  }

  static void applyTsumo(int[] delta, int winner, int oya, WinPayment.Tsumo payment) {
    switch (payment) {
      case WinPayment.DealerTsumo tsumo -> {
        for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
          if (player != winner) {
            delta[player] -= tsumo.each();
          }
        }
      }
      case WinPayment.ChildTsumo tsumo -> {
        for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
          if (player != winner) {
            delta[player] -= player == oya ? tsumo.fromDealer() : tsumo.fromChild();
          }
        }
      }
    }
  }

  static WinClaim.Ron ron(
      int winner,
      int loser,
      int oya,
      int honba,
      HandScoreBuffer agari,
      HandView winnerHand,
      boolean riichiDeclared) {
    WinPayment.Ron payment =
        ronPayment(agari.basePoints(), winner == oya ? WinnerRole.DEALER : WinnerRole.CHILD, honba);
    int player0 = 0;
    int player1 = 0;
    int player2 = 0;
    int player3 = 0;
    int pao = paoPlayer(winner, winnerHand, agari);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      int delta = 0;
      if (player == winner) {
        delta = payment.total();
      } else if (pao < 0 || pao == loser) {
        delta = player == loser ? -payment.points() : 0;
      } else {
        int honbaPayment = honba * 300;
        int half = (payment.points() - honbaPayment) / 2;
        if (player == pao) {
          delta = -half - honbaPayment;
        } else if (player == loser) {
          delta = -half;
        }
      }
      switch (player) {
        case 0 -> player0 = delta;
        case 1 -> player1 = delta;
        case 2 -> player2 = delta;
        default -> player3 = delta;
      }
    }
    return new WinClaim.Ron(
        winner,
        loser,
        agari.snapshot(),
        payment,
        honba,
        new PointDelta(player0, player1, player2, player3),
        riichiDeclared);
  }

  private static WinPayment.Ron ronPayment(int basePoints, WinnerRole role, int honba) {
    WinPayment.Ron payment = ScorePayments.ron(basePoints, role);
    return honba == 0 ? payment : new WinPayment.Ron(payment.points() + honba * 300);
  }

  private static int paoPlayer(int winner, HandView winnerHand, HandScoreBuffer yaku) {
    if (yaku.hasYakuman(ScoringYaku.DAISANGEN)) {
      int player = paoPlayer(winner, winnerHand, ScoringYaku.DAISANGEN);
      if (player >= 0) {
        return player;
      }
    }
    if (yaku.hasYakuman(ScoringYaku.DAISUUSHII)) {
      return paoPlayer(winner, winnerHand, ScoringYaku.DAISUUSHII);
    }
    return -1;
  }

  private static int paoPlayer(int winner, HandView winnerHand, ScoringYaku yakuman) {
    int requiredMelds = yakuman == ScoringYaku.DAISANGEN ? 3 : 4;
    int found = 0;
    for (int meldIndex = 0; meldIndex < winnerHand.meldCount(); meldIndex++) {
      Meld meld = winnerHand.meld(meldIndex);
      int tile = meld.baseTileType();
      boolean relevant = yakuman == ScoringYaku.DAISANGEN ? Tile.isDragon(tile) : Tile.isWind(tile);
      if (!relevant || ++found < requiredMelds) {
        continue;
      }
      if (meld.preservesMenzen()) {
        return -1;
      }
      return (winner + meld.relativeSource().playerOffset()) % GameState.NUM_PLAYERS;
    }
    return -1;
  }
}
