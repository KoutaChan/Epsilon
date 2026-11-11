package com.epsilon.agent.facts;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;

/**
 * 手の形から求める値を持つ。持つのは、向聴（3 つの形の最小、通常形、七対子、国士）、受け入れの種類、待ち、受け入れの枚数。
 * 判断の時点の手にも、合法手を選んだ後の手にも使う。{@link #analyze} で作り直し、そのときメモリは確保しない。同時に使えるのは
 * 1 つのスレッドだけ。
 *
 * <p>受け入れの種類と待ちは形だけで決まる（{@code ShapeAnalyzer.improving}・{@code waits}。役・振聴・残りの枚数は見ない）。
 * どちらも 13 枚相当の手のときだけ求める。その種類の牌を手と自分の面子で全て持っていても、受け入れや待ちに入ることがある。
 * 受け入れの枚数は、受け入れの各種類で自分から見えない枚数を足したもの（{@link TileCounts#live}。全て持っている種類は 0 枚）。
 */
public final class ShapeFacts {

  private final Variant variant;
  private final ShapeAnalyzer analyzer;
  private int shanten;
  private int standardShanten;
  private int chiitoitsuShanten;
  private int kokushiShanten;
  private boolean special;
  private long improving;
  private long waits;
  private int acceptance;

  public ShapeFacts(Variant variant) {
    this.variant = variant;
    analyzer = new ShapeAnalyzer(variant);
  }

  /**
   * 手を解析して、向聴・受け入れ・待ちを求め直す。種類ごとの枚数のルール（四人打ちか三人打ちか）が作ったときと
   * 違えば、IllegalArgumentException を投げる。
   *
   * @param hand 解析する手（14 枚相当以下）
   * @param tiles 受け入れの枚数を数えるのに使う、種類ごとの枚数
   */
  public void analyze(HandShape hand, TileCounts tiles) {
    if (tiles.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + tiles.variant());
    }
    shanten = analyzer.shanten(hand);
    standardShanten = analyzer.standardShanten(hand);
    special = hand.melds() == 0;
    if (special) {
      chiitoitsuShanten = ShapeAnalyzer.chiitoitsuShanten(hand);
      kokushiShanten = ShapeAnalyzer.kokushiShanten(hand);
    }
    boolean drawing = hand.size() == Variant.HAND;
    improving = drawing ? analyzer.improving(hand) : 0;
    waits = drawing && shanten == 0 ? improving : 0;
    acceptance = tiles.live(improving);
  }

  /** 3 つの形のうち最も小さい向聴（和了の形なら −1）。 */
  public int shanten() {
    return shanten;
  }

  public int standardShanten() {
    return standardShanten;
  }

  /** 面子が無く、七対子と国士の向聴も数える手なら true。 */
  public boolean special() {
    return special;
  }

  /** 七対子の向聴。面子のある手（{@link #special} が false）なら IllegalStateException を投げる。 */
  public int chiitoitsuShanten() {
    requireSpecial();
    return chiitoitsuShanten;
  }

  /** 国士の向聴。面子のある手（{@link #special} が false）なら IllegalStateException を投げる。 */
  public int kokushiShanten() {
    requireSpecial();
    return kokushiShanten;
  }

  /** 引くと向聴が下がる種類（受け入れ）。13 枚相当の手でなければ 0。 */
  public long improving() {
    return improving;
  }

  /** 形の上での和了牌。13 枚相当の聴牌の手でなければ 0。聴牌の手では受け入れと同じになる。 */
  public long waits() {
    return waits;
  }

  /** 受け入れの枚数（受け入れの各種類で、自分から見えない枚数を足したもの）。 */
  public int acceptance() {
    return acceptance;
  }

  private void requireSpecial() {
    if (!special) {
      throw new IllegalStateException("A hand with melds has no seven pairs or thirteen orphans shanten");
    }
  }
}
