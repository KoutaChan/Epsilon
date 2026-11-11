package com.epsilon.prism4p.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 遷移ごとのフロンティアの列。値を求めるのは {@link FrontierFeatures}。
 *
 * <p>フロンティアとは、遷移の後の手（13 枚相当で 1 向聴）から受け入れの牌を 1 枚引き、手牌を 1 枚捨てると、役のある聴牌に届く牌のこと
 * （エンジンの {@code TenpaiFrontierAnalyzer} と同じ決まり）。役は立直なしで、公開の条件（自風・場風）で見る。届いた聴牌の待ちのどれかに役があれば届くとする。
 * 届く牌は {@link Reach} の 3 通りに分け、それぞれについて牌の種類の数と生きた枚数（{@link Measure}）を列にする。1 向聴でない遷移と、
 * 和了・九種九牌の遷移は全て 0。
 */
public enum FrontierField {
  /** 振聴でないロンに役がある聴牌に届く牌の種類の数。 */
  FREE_RON_KINDS(Reach.FREE_RON, Measure.KINDS),
  /** 振聴でないロンに役がある聴牌に届く牌の生きた枚数。 */
  FREE_RON_LIVE(Reach.FREE_RON, Measure.LIVE),
  /** ロンに役がある聴牌に届くが、その聴牌が全て振聴になる牌の種類の数。 */
  FURITEN_RON_KINDS(Reach.FURITEN_RON, Measure.KINDS),
  /** ロンに役がある聴牌に届くが、その聴牌が全て振聴になる牌の生きた枚数。 */
  FURITEN_RON_LIVE(Reach.FURITEN_RON, Measure.LIVE),
  /** ツモに役がある聴牌に届く牌の種類の数。 */
  TSUMO_KINDS(Reach.TSUMO, Measure.KINDS),
  /** ツモに役がある聴牌に届く牌の生きた枚数。 */
  TSUMO_LIVE(Reach.TSUMO, Measure.LIVE);

  /** 1 遷移の列の数。 */
  public static final int WIDTH = values().length;

  private final Reach reach;
  private final Measure measure;

  FrontierField(Reach reach, Measure measure) {
    this.reach = reach;
    this.measure = measure;
  }

  /**
   * 数える牌の分け方を返す。
   *
   * @return 分け方
   */
  public Reach reach() {
    return reach;
  }

  /**
   * 数え方を返す。
   *
   * @return 数え方
   */
  public Measure measure() {
    return measure;
  }

  /** 届く牌の分け方。振聴でないロンと振聴だけのロンは重ならず、ツモはどちらとも重なりうる。 */
  public enum Reach {
    /** 届く聴牌のどれかで、ロンに役があり振聴でない。 */
    FREE_RON,
    /** ロンに役がある聴牌に届くが、どれも振聴になる（待ちが自分の河か、その聴牌にするときに捨てる牌にある）。 */
    FURITEN_RON,
    /** 届く聴牌のどれかで、ツモに役がある。 */
    TSUMO
  }

  /** 牌のマスクの数え方。 */
  public enum Measure {
    /** 牌の種類の数（34 で割る）。 */
    KINDS(Tiles.KINDS),
    /** 自分から見えない枚数の和（遷移の受け入れの生きた枚数と同じく 34 で割る）。 */
    LIVE(Tiles.KINDS);

    private final float scale;

    Measure(float scale) {
      this.scale = scale;
    }

    /**
     * 正規化の定数を返す。
     *
     * @return 値を割る数
     */
    public float scale() {
      return scale;
    }

    /** 牌のマスクを数える（正規化の前の値）。 */
    int count(long tileMask, TileCounts tiles) {
      return switch (this) {
        case KINDS -> Long.bitCount(tileMask);
        case LIVE -> tiles.live(tileMask);
      };
    }
  }
}
