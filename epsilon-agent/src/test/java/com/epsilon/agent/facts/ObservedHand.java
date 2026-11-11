package com.epsilon.agent.facts;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;

/** 記録 O から、自分（相対席 0）の今の手と、その手を採点する条件を書くテスト用のヘルパー。 */
final class ObservedHand {

  private ObservedHand() {}

  /**
   * 自分の純手牌と面子の数を、手の形に書く。
   *
   * @param tiles 種類ごとの枚数。ここから純手牌を読む
   * @param seats 席ごとの公開情報。ここから面子の数を読む（同じ O から作ったもの）
   * @param out 書く先
   */
  static void shape(TileCounts tiles, SeatFacts seats, HandShape out) {
    out.clear();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < tiles.hand(kind); copy++) {
        out.add(kind);
      }
    }
    for (int meld = 0; meld < seats.meldCount(0); meld++) {
      out.addMeld();
    }
  }

  /**
   * 採点の条件に、自分の今の手を書く。書くのは、純手牌とその赤、面子、抜いた北、自風と場風、表示牌、立直（一発は数えない）。
   *
   * @param o 記録 O。ここから風・表示牌・立直・抜いた北を読む
   * @param tiles 種類ごとの枚数。ここから純手牌と赤を読む（同じ O から作ったもの）
   * @param seats 席ごとの公開情報。ここから面子を読む（同じ O から作ったもの）
   * @param context 書く先
   */
  static void context(Observation o, TileCounts tiles, SeatFacts seats, WinContext context) {
    context.clear();
    shape(tiles, seats, context.hand());
    for (int index = 0; index < seats.meldCount(0); index++) {
      context.addMeld(seats.meld(0, index));
    }
    context.setWinds(o.variant().seatWind(o.kyoku(), o.observer()), o.roundWindTile());
    for (int index = 0; index < o.indicators(); index++) {
      context.addIndicator(o.indicatorTile(index));
    }
    context.setRiichi(o.riichi(0), false);
    context.setRedFives(tiles.handRed());
    context.setNuki(o.nuki(0));
  }
}
