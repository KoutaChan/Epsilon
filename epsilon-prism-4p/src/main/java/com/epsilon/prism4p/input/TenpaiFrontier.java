package com.epsilon.prism4p.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism4p.input.FrontierField.Reach;

/**
 * 遷移の後の手が 1 向聴のとき、その手のフロンティア（{@link Reach} ごとの牌のマスク）を探す。形は {@link ShapeAnalyzer} で見て、役の有無は
 * {@link HandScorer} で採点して見る。探す間にメモリを確保しない。
 *
 * <p>探すのは、13 枚相当で最小の向聴が 1 の手だけ。受け入れの牌を 1 枚ずつ引き、手牌（引いた牌と同じ種類を除く）を 1 枚ずつ捨てる。受け入れからは、
 * 自分の面子と手牌で 5 枚目になる牌と、見えない枚数が 0 の牌を除く。聴牌になったら、待ち（5 枚目になる牌を除く）のどれかに役があるかを、立直なしの
 * 公開の条件で採点して調べる。基本点が 0 でなければ役がある。赤はドラで役の有無を変えないので、採点の条件に書かない。引いた牌について、振聴でない
 * ロンとツモの両方が見つかったら、残りの打牌は調べない。
 */
final class TenpaiFrontier {

  private static final int THIRTEEN = HandShape.MAX_SIZE - 1;
  private static final int FRONTIER_SHANTEN = 1;
  private static final Reach[] REACHES = Reach.values();
  private static final int RON = 1;
  private static final int TSUMO = 2;
  private static final int BOTH = RON | TSUMO;

  private final ShapeAnalyzer shapes;
  private final HandScorer scorer = new HandScorer();
  private final WinContext context;
  private final WinScore score = new WinScore();
  private final long[] masks = new long[REACHES.length];

  TenpaiFrontier(Variant variant) {
    shapes = new ShapeAnalyzer(variant);
    context = new WinContext(variant);
  }

  /**
   * フロンティアを探す。13 枚相当で 1 向聴でなければ、全てのマスクを空にする。
   *
   * @param hand 遷移の後の手
   * @param river 自分の河の牌と、この遷移で捨てる牌のマスク
   * @param tiles 種類ごとの枚数（見えない枚数を読む）
   * @param o 判断の記録（場風と表示牌を読む）
   * @param seatWind 自風の種類
   */
  void analyze(AfterHand hand, long river, TileCounts tiles, Observation o, int seatWind) {
    long freeRon = 0L;
    long furitenRon = 0L;
    long tsumo = 0L;
    HandShape shape = hand.shape();
    if (hand.effectiveTileCount() == THIRTEEN && shapes.shanten(shape) == FRONTIER_SHANTEN) {
      long draws = shapes.improving(shape) & ShapeAnalysis.possibleDraws(shape, hand);
      HandShape cursor = load(hand, o, seatWind);
      long concealed = shape.kinds();
      for (; draws != 0L; draws &= draws - 1) {
        int draw = Long.numberOfTrailingZeros(draws);
        if (tiles.unseen(draw) == 0) {
          continue;
        }
        boolean free = false;
        boolean furiten = false;
        boolean self = false;
        cursor.add(draw);
        for (long discards = concealed & ~Tiles.bit(draw);
            discards != 0L && !(free && self);
            discards &= discards - 1) {
          int discard = Long.numberOfTrailingZeros(discards);
          cursor.remove(discard);
          if (shapes.shanten(cursor) == 0) {
            long waits = shapes.waits(cursor) & ShapeAnalysis.possibleDraws(cursor, hand);
            int found = yaku(waits);
            if ((found & RON) != 0) {
              if ((waits & (river | Tiles.bit(discard))) != 0L) {
                furiten = true;
              } else {
                free = true;
              }
            }
            self |= (found & TSUMO) != 0;
          }
          cursor.add(discard);
        }
        cursor.remove(draw);
        long bit = Tiles.bit(draw);
        if (free) {
          freeRon |= bit;
        } else if (furiten) {
          furitenRon |= bit;
        }
        if (self) {
          tsumo |= bit;
        }
      }
    }
    masks[Reach.FREE_RON.ordinal()] = freeRon;
    masks[Reach.FURITEN_RON.ordinal()] = furitenRon;
    masks[Reach.TSUMO.ordinal()] = tsumo;
  }

  /**
   * 探したフロンティアの牌のマスクを返す。
   *
   * @param reach 分け方
   * @return 牌のマスク（物理の番号）
   */
  long mask(Reach reach) {
    return masks[reach.ordinal()];
  }

  /** 採点の条件に、遷移の後の手と、立直なしの公開の条件を書く。返すのは条件の中の純手牌で、牌を引いて捨てるのにそのまま使う。 */
  private HandShape load(AfterHand hand, Observation o, int seatWind) {
    context.clear();
    hand.writeTo(context);
    context.setRedFives(0L);
    context.setWinds(seatWind, o.roundWindTile());
    for (int index = 0; index < o.indicators(); index++) {
      context.addIndicator(o.indicatorTile(index));
    }
    context.setRiichi(RiichiState.NONE, false);
    return context.hand();
  }

  /** 聴牌の手（条件の純手牌）で、待ちのどれかに役が付く和了の仕方を返す。値は {@link #RON} と {@link #TSUMO} のビットの和。 */
  private int yaku(long waits) {
    int found = 0;
    for (long rest = waits; rest != 0L && found != BOTH; rest &= rest - 1) {
      int kind = Long.numberOfTrailingZeros(rest);
      if ((found & RON) == 0 && wins(kind, WinContext.Source.RON)) {
        found |= RON;
      }
      if ((found & TSUMO) == 0 && wins(kind, WinContext.Source.TSUMO)) {
        found |= TSUMO;
      }
    }
    return found;
  }

  private boolean wins(int kind, WinContext.Source source) {
    context.setWin(kind, source);
    return scorer.score(context, score) && score.base() > 0;
  }
}
