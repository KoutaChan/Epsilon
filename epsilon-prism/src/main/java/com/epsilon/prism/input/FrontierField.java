package com.epsilon.prism.input;

import com.epsilon.core.Tile;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.Element;

/**
 * 前線の入力の欄（遷移ごと、BF16）。前線を使う網だけが読む。基準の入力（{@link InputSchema}）の欄にも指紋にも入れず、
 * 指紋は {@link #fingerprint()} で別に持つ。
 *
 * <p>前線は、13 枚相当で 1 向聴の後の手について、受け入れの牌を 1 枚引いて手の牌を 1 枚捨てると役のある聴牌に届く牌（エンジンの
 * {@code TenpaiFrontierAnalyzer} と同じ決まり）。役は立直なしの公開の条件（自風・場風）で見て、届いた聴牌の待ちのどれかで役があれば届くとする。
 * 届く牌は {@link Reach} の 3 つに分け、欄はそれぞれの牌の種類の数と生きた枚数（{@link Measure}）。1 向聴でない遷移と、和了・九種九牌の遷移は
 * 全て 0。
 *
 * <p>テンソルの形（網との約束）:
 *
 * <ul>
 *   <li>[遷移, {@link #WIDTH}] の {@link Element#BF16}（リトルエンディアン）。列は宣言の順。値は {@link Measure#scale} で割ってから最近接偶数で
 *       丸める。
 *   <li>遷移の並びは、同じ塊の {@link HostBatch} の遷移の区画（{@link InputSchema.Tensor#TRANSITION_SCALAR} など）と同じ。行 r の遷移は
 *       {@link BatchPlan#firstTransition}(r) から {@link BatchPlan#transitions}(r) 個。代表でない双子は遷移を持たない。
 *   <li>前線を使う網だけが、この [遷移, WIDTH] を候補層の遷移の数値の後ろに足して読む。塊は {@link FrontierBatch}、書き手は
 *       {@link FrontierFeatures}。
 * </ul>
 */
public enum FrontierField {
  /** 振聴でないロンに役がある聴牌に届く牌の種類の数。 */
  FREE_RON_KINDS(Reach.FREE_RON, Measure.KINDS),
  /** 振聴でないロンに役がある聴牌に届く牌の生きた枚数。 */
  FREE_RON_LIVE(Reach.FREE_RON, Measure.LIVE),
  /** ロンに役がある聴牌に届くが、届く聴牌が全て振聴の牌の種類の数。 */
  FURITEN_RON_KINDS(Reach.FURITEN_RON, Measure.KINDS),
  /** ロンに役がある聴牌に届くが、届く聴牌が全て振聴の牌の生きた枚数。 */
  FURITEN_RON_LIVE(Reach.FURITEN_RON, Measure.LIVE),
  /** ツモに役がある聴牌に届く牌の種類の数。 */
  TSUMO_KINDS(Reach.TSUMO, Measure.KINDS),
  /** ツモに役がある聴牌に届く牌の生きた枚数。 */
  TSUMO_LIVE(Reach.TSUMO, Measure.LIVE);

  /** 1 遷移の列の数。 */
  public static final int WIDTH = values().length;

  public static final Element ELEMENT = Element.BF16;

  /** 行の軸。 */
  public static final Axis AXIS = Axis.TRANSITION;

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

  /**
   * 前線の入力の指紋を返す。前線を使う網の model.json に、入力の指紋と並べて書く。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** 届く牌の分け方。振聴でないロンと振聴だけのロンは重ならず、ツモはどちらとも重なりうる。 */
  public enum Reach {
    /** 届く聴牌のどれかで、ロンに役があり振聴でない。 */
    FREE_RON,
    /** ロンに役がある聴牌に届くが、そのどれもが振聴（待ちが自分の河か、その聴牌で捨てる牌にある）。 */
    FURITEN_RON,
    /** 届く聴牌のどれかで、ツモに役がある。 */
    TSUMO
  }

  /** 牌のマスクの数え方。 */
  public enum Measure {
    /** 牌の種類の数（34 で割る）。 */
    KINDS(Tile.NUM_TILE_TYPES),
    /** 自分から見えない枚数の和（遷移の受け入れの生きた枚数と同じく 34 で割る）。 */
    LIVE(Tile.NUM_TILE_TYPES);

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
        case LIVE -> tiles.liveCopies(tileMask);
      };
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE = describe().hex();

    /** 欄の名前・順・分け方・数え方・正規化と、要素の型と軸。 */
    private static SchemaDigest describe() {
      StringBuilder fields = new StringBuilder();
      for (FrontierField field : values()) {
        fields.append(field).append(':').append(field.reach).append('/').append(field.measure);
        fields.append('/').append(field.measure.scale).append(',');
      }
      return SchemaDigest.named("epsilon-prism-v2-frontier")
          .values(FrontierField.class)
          .values(Reach.class)
          .values(Measure.class)
          .with(SchemaDigest.name(FrontierField.class), fields)
          .with("element", ELEMENT)
          .with("axis", AXIS);
    }
  }
}
