package com.epsilon.prism.auxiliary;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.agent.observe.codec.BitField;
import com.epsilon.agent.observe.codec.BitLayout;
import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import java.util.Objects;

/**
 * Oracle の記録を {@code long[]} に詰め、またそこから読み出す。Oracle の記録は、判断の時点の全員の手から {@link OracleCapture} が作る
 * 読みのヘッドの教師。詰めるのも読むのもこのクラスだけが行い、配置はフィールドの enum からルールごとに {@link BitLayout} が作る。
 *
 * <p>先頭に相手ごとの記録（{@link OpponentBits}: 形だけで見た待ち 34 bit、最小の向聴数、放銃牌 34 bit）を、判断した席から見て下家から
 * 順に、席の数 − 1 席分置く。その後に放銃牌ごとの点（{@link PointBits}）を、放銃牌の数だけ並べる。並びは相手の順で、同じ相手の中では
 * 牌の順。長さは放銃牌の数で決まり、{@link #longs} で求める。
 *
 * <p>放銃牌は、判断した席が今その牌（赤でない牌）を捨てたら、その相手がロンできる牌。判定はエンジンが行い、役・振聴・5 枚目の有無まで
 * 見る。点はそのロンで和了者が受け取る点で、裏ドラ・本場・供託は含まず、包による払いの分け方も考えない。点は
 * {@link ObservationSchema#POINT_UNIT} 点単位で書く。形式はルールごとに 1 つで、どのスレッドからも使える。
 */
public final class OracleCodec {

  private static final OracleCodec[] CODECS = new OracleCodec[Variant.values().length];

  static {
    for (Variant variant : Variant.values()) {
      CODECS[variant.ordinal()] = new OracleCodec(variant);
    }
  }

  private final Variant variant;

  /** 相手の数（席の数 − 1）。 */
  private final int opponents;

  /** 相手の記録の配置（相手 1 席に 1 記録）。 */
  private final BitLayout<OpponentBits> opponent;

  /** 放銃牌の点の配置（放銃牌 1 つに 1 記録）。 */
  private final BitLayout<PointBits> point;

  /** 放銃牌の点が始まる位置（{@code long} の添字）。 */
  private final int pointsStart;

  private final int maxLongs;
  private final String fingerprint;

  private OracleCodec(Variant variant) {
    this.variant = variant;
    opponents = variant.seats() - 1;
    opponent = BitLayout.of(OpponentBits.class, Long.SIZE, variant);
    point = BitLayout.of(PointBits.class, Short.SIZE, variant);
    pointsStart = opponent.longs(opponents);
    maxLongs = pointsStart + point.longs(opponents * Limits.MAX_WAIT_KINDS);
    fingerprint =
        SchemaDigest.named("epsilon-prism-v2-oracle")
            .layout(opponent)
            .layout(point)
            .with("opponents", opponents)
            .with("tiles", Tiles.KINDS)
            .with("pointUnit", ObservationSchema.POINT_UNIT)
            .hex();
  }

  /**
   * ルールごとの Oracle の記録の形式を返す。
   *
   * @param variant ルール
   * @return そのルールの形式
   */
  public static OracleCodec of(Variant variant) {
    return CODECS[variant.ordinal()];
  }

  public Variant variant() {
    return variant;
  }

  /** 相手の数（席の数 − 1）を返す。 */
  public int opponents() {
    return opponents;
  }

  /** Oracle の記録の {@code long} の数の最大を返す。全ての相手が国士の 13 面待ちで、全ての待ちでロンできるときの長さ。 */
  public int maxLongs() {
    return maxLongs;
  }

  /**
   * Oracle の記録の {@code long} の数を、放銃牌の数から求めて返す。
   *
   * @param words Oracle の記録を置いた配列
   * @param base 記録の先頭の位置
   * @return {@code long} の数
   * @throws IndexOutOfBoundsException 相手の記録が配列に収まらないとき
   */
  public int longs(long[] words, int base) {
    Objects.checkFromIndexSize(base, pointsStart, words.length);
    int dangers = 0;
    for (int relative = 1; relative <= opponents; relative++) {
      dangers += Long.bitCount(danger(words, base, relative));
    }
    return pointsStart + point.longs(dangers);
  }

  /**
   * 相手の最小の向聴数を返す。
   *
   * @param words Oracle の記録を置いた配列
   * @param base 記録の先頭の位置
   * @param relative 相手の相対席（下家が 1）
   * @return 最小の向聴数（0 は聴牌）
   */
  public int shanten(long[] words, int base, int relative) {
    return opponent.get(words, base, record(relative), OpponentBits.SHANTEN);
  }

  /**
   * 相手の、形だけで見た待ちを返す。
   *
   * @param words Oracle の記録を置いた配列
   * @param base 記録の先頭の位置
   * @param relative 相手の相対席（下家が 1）
   * @return 待ちの牌のマスク。ビットは牌で、聴牌でなければ 0
   */
  public long waits(long[] words, int base, int relative) {
    return tiles(words, base, record(relative), OpponentBits.WAIT);
  }

  /**
   * 判断した席が今捨てたら、相手がロンできる牌を返す。
   *
   * @param words Oracle の記録を置いた配列
   * @param base 記録の先頭の位置
   * @param relative 相手の相対席（下家が 1）
   * @return 放銃牌のマスク（ビットは牌）
   */
  public long danger(long[] words, int base, int relative) {
    return tiles(words, base, record(relative), OpponentBits.DANGER);
  }

  /**
   * 放銃牌でロンされたときの点を返す。
   *
   * @param words Oracle の記録を置いた配列
   * @param base 記録の先頭の位置
   * @param index 放銃牌の番号。全ての相手の放銃牌を、相手の順に、同じ相手の中では牌の順に数えた 0 始まりの番号
   * @return そのロンで和了者が受け取る点
   */
  public int points(long[] words, int base, int index) {
    return point.get(words, base + pointsStart, index, PointBits.POINTS) * ObservationSchema.POINT_UNIT;
  }

  /**
   * Oracle の記録の形式のフィンガープリントを返す。行のマニフェストに書き、読むときに照合する。
   *
   * @return SHA-256 の先頭 8 B を 16 進で書いた文字列
   */
  public String fingerprint() {
    return fingerprint;
  }

  /** 相手の記録を書く。Oracle の記録は先に 0 で埋めておくこと。 */
  void writeOpponent(long[] words, int base, int relative, int shanten, long waits, long danger) {
    int record = record(relative);
    opponent.put(words, base, record, OpponentBits.SHANTEN, shanten);
    writeTiles(words, base, record, OpponentBits.WAIT, waits);
    writeTiles(words, base, record, OpponentBits.DANGER, danger);
  }

  /** {@code index} 番目の放銃牌の点を書く。 */
  void writePoints(long[] words, int base, int index, int points) {
    if (points % ObservationSchema.POINT_UNIT != 0) {
      throw new IllegalStateException("Points are not a multiple of " + ObservationSchema.POINT_UNIT + ": " + points);
    }
    point.put(words, base + pointsStart, index, PointBits.POINTS, points / ObservationSchema.POINT_UNIT);
  }

  private int record(int relative) {
    return Objects.checkIndex(relative - 1, opponents);
  }

  private long tiles(long[] words, int base, int record, OpponentBits field) {
    long mask = 0L;
    for (int tile = 0; tile < Tiles.KINDS; tile++) {
      if (opponent.get(words, base, record, field, tile) != 0) {
        mask |= Tiles.bit(tile);
      }
    }
    return mask;
  }

  private void writeTiles(long[] words, int base, int record, OpponentBits field, long mask) {
    if ((mask & ~Tiles.ALL_KINDS) != 0L) {
      throw new IllegalArgumentException(field + " mask has bits beyond the tiles: " + mask);
    }
    for (long rest = mask; rest != 0L; rest &= rest - 1) {
      opponent.put(words, base, record, field, Long.numberOfTrailingZeros(rest), 1);
    }
  }

  /** 相手 1 席の記録のフィールド。 */
  enum OpponentBits implements BitField {
    /** 形だけで見た待ち。繰り返しの番号は牌で、聴牌でなければ全て 0。 */
    WAIT(1, Tiles.KINDS),
    /** 最小の向聴数（0 は聴牌）。 */
    SHANTEN(Limits.MAX_SHANTEN, 1),
    /** 判断した席が今その牌を捨てたら、この相手がロンできるなら 1。繰り返しの番号は牌。 */
    DANGER(1, Tiles.KINDS);

    private final int max;
    private final int count;

    OpponentBits(int max, int count) {
      this.max = max;
      this.count = count;
    }

    @Override
    public int max(Variant variant) {
      return max;
    }

    @Override
    public int count(Variant variant) {
      return count;
    }
  }

  /** 放銃牌 1 つの点のフィールド。 */
  enum PointBits implements BitField {
    /** そのロンで和了者が受け取る点（{@link ObservationSchema#POINT_UNIT} 点単位）。 */
    POINTS;

    @Override
    public int max(Variant variant) {
      return Limits.MAX_WIN_POINTS / ObservationSchema.POINT_UNIT;
    }
  }
}
