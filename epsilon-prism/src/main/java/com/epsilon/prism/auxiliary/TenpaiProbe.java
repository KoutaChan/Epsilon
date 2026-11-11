package com.epsilon.prism.auxiliary;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 席の手が聴牌か、ロンの役がある聴牌かを確かめる。{@link TenpaiReach} と {@link HandEndAnalyzer} が使う。判定用のオブジェクトはスレッド
 * ごと・ルールごとに持ち、メモリを割り当てない。
 *
 * <p>聴牌の判定はエンジンの荒牌の聴牌と同じで、13 枚相当の手に形だけで見た待ちがあれば聴牌とする。純手牌で 4 枚持つ牌だけを待つ形は不聴。
 * 役は公開されている局の状況（自風・場風）だけで確かめ、ドラ・赤・振聴・偶然の役（一発・海底・河底・槍槓・嶺上）は見ない。手と面子で
 * 4 枚とも持つ牌の待ちは、5 枚目がなく和了れないので見ない。この局で立直を宣言した席は立直が役になるので、聴牌ならそのまま役のある聴牌と
 * する。宣言牌でロンされて立直が成立しなかった席も同じに扱う。14 枚相当の手は、1 枚捨てて聴牌になれるなら聴牌、役のある聴牌になれるなら
 * 役のある聴牌とする。
 */
final class TenpaiProbe {

  private static final ThreadLocal<TenpaiProbe[]> PROBES =
      ThreadLocal.withInitial(() -> new TenpaiProbe[Variant.values().length]);

  private final Variant variant;
  private final ShapeAnalyzer shapes;
  private final HandScorer scorer = new HandScorer();
  private final WinContext context;
  private final WinScore score = new WinScore();
  private final HandShape hand = new HandShape();
  private final HandShape trial = new HandShape();

  private TenpaiProbe(Variant variant) {
    this.variant = variant;
    shapes = new ShapeAnalyzer(variant);
    context = new WinContext(variant);
  }

  /** このスレッドが持つ、そのルールの判定用のオブジェクトを返す。 */
  static TenpaiProbe of(Variant variant) {
    TenpaiProbe[] all = PROBES.get();
    TenpaiProbe probe = all[variant.ordinal()];
    if (probe == null) {
      probe = new TenpaiProbe(variant);
      all[variant.ordinal()] = probe;
    }
    return probe;
  }

  /**
   * 席の手が聴牌なら true を返す。
   *
   * @param full 局面
   * @param seat 席
   * @return 聴牌なら true
   * @throws IllegalStateException 手が 13 枚相当でも 14 枚相当でもないとき
   */
  boolean tenpai(FullView full, int seat) {
    full.copyHand(seat, hand);
    return tenpai(hand);
  }

  /**
   * 席の手がロンの役がある聴牌なら true を返す。待ちごとに役を確かめ、見つかった時点で止める。
   *
   * @param full 局面
   * @param seat 席
   * @return 役のある聴牌なら true
   * @throws IllegalStateException 手が 13 枚相当でも 14 枚相当でもないとき
   */
  boolean yakuTenpai(FullView full, int seat) {
    full.copyHand(seat, hand);
    SeatView view = full.seat(seat);
    if (declaredRiichi(view.events(), seat)) {
      return tenpai(hand);
    }
    return switch (requireSize(hand)) {
      case HandShape.MAX_SIZE - 1 -> hasRonYaku(view, seat, hand);
      default -> {
        for (long kinds = hand.kinds(); kinds != 0L; kinds &= kinds - 1) {
          trial.copyFrom(hand);
          trial.remove(Long.numberOfTrailingZeros(kinds));
          if (hasRonYaku(view, seat, trial)) {
            yield true;
          }
        }
        yield false;
      }
    };
  }

  /**
   * 面子か抜いた北に含まれる牌のうち、席が純手牌と合わせて 4 枚とも持つ牌の種類を返す。純手牌だけで 4 枚持つ牌は含めない。
   *
   * @param view 局面
   * @param seat 席
   * @param shape 席の純手牌
   * @return 牌の種類のマスク
   */
  static long fullyHeld(SeatView view, int seat, HandShape shape) {
    long full = 0L;
    for (int index = 0, melds = view.meldCount(seat); index < melds; index++) {
      long kinds = Meld.of(view.meld(seat, index)).kinds() & ~full;
      for (; kinds != 0L; kinds &= kinds - 1) {
        int kind = Long.numberOfTrailingZeros(kinds);
        if (shape.count(kind) + inMelds(view, seat, kind) >= Tiles.COPIES) {
          full |= Tiles.bit(kind);
        }
      }
    }
    int nuki = view.nuki(seat);
    if (nuki > 0 && shape.count(Tiles.PEI) + nuki >= Tiles.COPIES) {
      full |= Tiles.bit(Tiles.PEI);
    }
    return full;
  }

  private boolean tenpai(HandShape shape) {
    if (requireSize(shape) == HandShape.MAX_SIZE - 1) {
      return shapes.waits(shape) != 0L;
    }
    for (long kinds = shape.kinds(); kinds != 0L; kinds &= kinds - 1) {
      trial.copyFrom(shape);
      trial.remove(Long.numberOfTrailingZeros(kinds));
      if (shapes.waits(trial) != 0L) {
        return true;
      }
    }
    return false;
  }

  /** 13 枚相当の手の、形だけで見た待ちのどれかでロンしたときに役があるなら true を返す。 */
  private boolean hasRonYaku(SeatView view, int seat, HandShape shape) {
    long waits = shapes.waits(shape) & ~fullyHeld(view, seat, shape);
    if (waits == 0L) {
      return false;
    }
    context.clear();
    context.hand().copyFrom(shape);
    for (int index = 0, melds = view.meldCount(seat); index < melds; index++) {
      context.addMeld(Meld.of(view.meld(seat, index)));
    }
    int round = view.round();
    context.setWinds(variant.seatWind(round, seat), variant.roundWind(round));
    for (; waits != 0L; waits &= waits - 1) {
      context.setWin(Long.numberOfTrailingZeros(waits), WinContext.Source.RON);
      if (scorer.score(context, score)) {
        return true;
      }
    }
    return false;
  }

  /** 席の面子に含まれる牌 {@code kind} の枚数を返す。 */
  private static int inMelds(SeatView view, int seat, int kind) {
    int held = 0;
    for (int index = 0, melds = view.meldCount(seat); index < melds; index++) {
      Meld meld = Meld.of(view.meld(seat, index));
      for (int at = 0, size = meld.size(); at < size; at++) {
        if (meld.tile(at) == kind) {
          held++;
        }
      }
    }
    return held;
  }

  /** この局で席が立直を宣言していれば true を返す。宣言牌でロンされた席も true。 */
  private static boolean declaredRiichi(EventLog events, int seat) {
    for (int event = 0, size = events.size(); event < size; event++) {
      if (events.riichi(event) && events.seat(event) == seat) {
        return true;
      }
    }
    return false;
  }

  private static int requireSize(HandShape shape) {
    int size = shape.size();
    if (size != HandShape.MAX_SIZE - 1 && size != HandShape.MAX_SIZE) {
      throw new IllegalStateException("A tenpai check needs 13 or 14 tiles: " + shape);
    }
    return size;
  }
}
