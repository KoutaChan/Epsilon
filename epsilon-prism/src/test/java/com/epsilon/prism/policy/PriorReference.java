package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.ShapeFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.Tiles;

/**
 * テストで使う p0 の参照実装。記録 O から agent の事実（手牌の枚数・面子の数・形の解析）を使って、打牌した後の最小の向聴数と、受け入れの
 * うち自分から見えない枚数を求め、{@link Prior} に通す。入力のライターが作る p0 の特徴とは別に実装してあり、両者がビットまで同じことを
 * 確かめるのに使う。
 *
 * <p>値はライターと同じく、Nano の正規化（向聴数は (s + 1) / 9、受け入れは 枚数 / 136）を一度かけてから戻す。スレッドセーフではない。
 */
final class PriorReference {
  private static final float SHANTEN_SCALE = 9f;
  private static final float UKEIRE_SCALE = Tiles.KINDS * (float) Tiles.COPIES;

  private final TileCounts tiles = new TileCounts(TENHOU_4P);
  private final SeatFacts seats = new SeatFacts(TENHOU_4P);
  private final ShapeFacts shape = new ShapeFacts(TENHOU_4P);
  private final HandShape hand = new HandShape();
  private final HandShape after = new HandShape();

  /** O の合法手ごとの p0 を返す。打牌でない手は、向聴数と受け入れを 0 として通す。 */
  float[] prior(Observation o, PriorSettings settings) {
    tiles.load(o);
    seats.load(o);
    loadHand();
    int n = o.legalActions();
    int[] legal = new int[n];
    float[] shanten = new float[n];
    float[] ukeire = new float[n];
    for (int a = 0; a < n; a++) {
      Action action = o.action(a);
      legal[a] = action.code();
      if (ActionGroups.of(action.type()) != ActionGroups.DISCARD) continue;
      after.copyFrom(hand);
      after.remove(action.tile());
      shape.analyze(after, tiles);
      shanten[a] = SHANTEN_SCALE * ((shape.shanten() + 1f) / SHANTEN_SCALE) - 1f;
      ukeire[a] = UKEIRE_SCALE * (shape.acceptance() / UKEIRE_SCALE);
    }
    return Prior.compute(legal, shanten, ukeire, settings);
  }

  /** 自分（相対席 0）の純手牌と面子の数を、手の形に書く。 */
  private void loadHand() {
    hand.clear();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < tiles.hand(kind); copy++) {
        hand.add(kind);
      }
    }
    for (int meld = 0; meld < seats.meldCount(0); meld++) {
      hand.addMeld();
    }
  }
}
