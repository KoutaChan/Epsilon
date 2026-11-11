package com.epsilon.agent.facts;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/**
 * 聴牌の手で、待ちごとに和了したときの点を求める（HandScorer で採点し、裏ドラは数えない）。待ちの種類ごとに、和了の仕方
 * （{@link Method}）と和了牌（{@link WinningTile}）の組み合わせを全て採点する。採点の条件（純手牌・面子・風・表示牌・立直・抜き・
 * 純手牌の赤）は、呼び出し側が {@link #context} に書く。和了牌は {@link #load} が書く。メモリは確保しない。同時に使えるのは 1 つの
 * スレッドだけ。
 *
 * <p>待ちは形の上での和了牌なので、その種類の牌を純手牌・面子・抜いた北で全て持っている待ち（5 枚目の牌を待つ形）も入りうる。
 * そうした待ちは採点せず、和了できない待ちとして扱う。
 */
public final class WaitScores {

  /** 待ちの種類の上限（国士の 13 面）。 */
  public static final int MAX_WAITS = 13;

  public enum Method {
    RON(WinContext.Source.RON),
    TSUMO(WinContext.Source.TSUMO);

    private final WinContext.Source source;

    Method(WinContext.Source source) {
      this.source = source;
    }
  }

  public enum WinningTile {
    /** 赤でない牌。 */
    PLAIN,
    /** 赤の 5。 */
    RED
  }

  private static final Method[] METHODS = Method.values();

  private final TileSet set;
  private final WinContext context;
  private final HandScorer scorer = new HandScorer();

  /** 待ち・和了の仕方・和了牌の組み合わせごとの点。添字は [待ち][和了の仕方][和了牌]。 */
  private final WinScore[][][] scores = new WinScore[MAX_WAITS][METHODS.length][WinningTile.values().length];

  private final int[] kinds = new int[MAX_WAITS];
  private int waits;

  public WaitScores(Variant variant) {
    set = variant.tiles();
    context = new WinContext(variant);
    for (WinScore[][] wait : scores) {
      for (WinScore[] method : wait) {
        for (int tile = 0; tile < method.length; tile++) {
          method[tile] = new WinScore();
        }
      }
    }
  }

  /** 採点の条件を返す。呼び出し側は空にしてから、13 枚相当の純手牌と、面子・風・表示牌・立直・抜き・純手牌の赤を書く。 */
  public WinContext context() {
    return context;
  }

  /**
   * 採点の条件に書いた手で、和了牌を種類の小さい順に採点する。終わった後、採点の条件の和了牌は最後に採点したもののまま残り、
   * 純手牌の赤は元に戻る。待ちの種類が {@link #MAX_WAITS} を超えれば IllegalArgumentException を投げる。
   *
   * @param waits 和了牌の種類
   * @param red 赤の 5 での和了も採点する 5 の種類（ふつうは、赤の 5 がまだ自分から見えていない種類。{@link TileCounts#unseenRed}）。
   *     純手牌に赤がある種類は、ここに入っていても採点しない
   */
  public void load(long waits, long red) {
    if (Long.bitCount(waits) > MAX_WAITS) {
      throw new IllegalArgumentException("More than " + MAX_WAITS + " waits: " + Long.toBinaryString(waits));
    }
    this.waits = 0;
    long handRed = context.redFives();
    try {
      int wait = 0;
      for (long rest = waits; rest != 0; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        boolean exhausted = held(kind) >= set.copies(kind);
        boolean redWin = !exhausted && Tiles.contains(red & ~handRed, kind);
        for (Method method : METHODS) {
          WinScore[] results = scores[wait][method.ordinal()];
          if (exhausted) {
            results[WinningTile.PLAIN.ordinal()].clear();
          } else {
            context.setWin(kind, method.source);
            context.setRedFives(handRed);
            scorer.score(context, results[WinningTile.PLAIN.ordinal()]);
          }
          if (redWin) {
            context.setRedFives(handRed | Tiles.bit(kind));
            scorer.score(context, results[WinningTile.RED.ordinal()]);
          } else {
            results[WinningTile.RED.ordinal()].clear();
          }
        }
        kinds[wait++] = kind;
      }
      this.waits = wait;
    } finally {
      context.setRedFives(handRed);
    }
  }

  /** 採点した待ちの種類の数。 */
  public int waits() {
    return waits;
  }

  /** 待ちの番号から、その待ちの牌の種類を返す（番号は種類の小さい順に振る）。 */
  public int wait(int index) {
    return kinds[Objects.checkIndex(index, waits)];
  }

  /**
   * 待ちを、和了の仕方（ロンかツモか）と和了牌（赤の 5 か、それ以外か）を決めて和了したときの点を返す。返す点は、次に
   * {@link #load} を呼ぶまで有効。役が無いとき、和了できない待ちのとき、赤の 5 での和了を採点しなかったとき（5 でない、
   * {@link #load} で赤の 5 を指定されていない、純手牌に赤がある）は、基本点が 0 の点を返す。
   *
   * @param index 待ちの番号
   * @param method 和了の仕方
   * @param tile 和了牌が赤の 5 か、それ以外か
   */
  public WinScore score(int index, Method method, WinningTile tile) {
    return scores[Objects.checkIndex(index, waits)][method.ordinal()][tile.ordinal()];
  }

  /** 指定した種類の牌を、純手牌・面子・抜いた北で合わせて何枚持っているか。 */
  private int held(int kind) {
    int held = context.hand().count(kind) + (kind == Tiles.PEI ? context.nuki() : 0);
    for (int index = 0; index < context.meldCount(); index++) {
      Meld meld = context.meld(index);
      if (Tiles.contains(meld.kinds(), kind)) {
        held += meld.type() == Meld.Type.CHI ? 1 : meld.size();
      }
    }
    return held;
  }
}
