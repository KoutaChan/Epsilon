package com.epsilon.prism.auxiliary;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.observe.ObservationSchema;
import java.util.Objects;

/**
 * 神託（判断の時点の全員の手から作る読みの頭の教師。{@link OracleCapture}）を {@code long[]} に詰めて解く。詰めと解きはこのクラスだけが行い、
 * 配置は欄の enum から {@link BitLayout} が作る。
 *
 * <p>先頭に相手 3 席（判断の席から見た下家・対面・上家の順）の記録（{@link OpponentBits}: 形の待ち 34 bit・最小の向聴・放銃 34 bit）を置き、
 * その後に放銃の牌ごとの点（{@link PointBits}）を、相手の順・相手の中は牌の順に、放銃の牌の数だけ並べる。長さは放銃の牌の数で決まり、
 * {@link #longs} で出す。
 *
 * <p>放銃は「判断の席が今その牌（赤でない牌）を捨てたら、その相手がロンできる」（役・振聴・5 枚目まで見るエンジンの判定）。点はそのロンの
 * 和了の点（和了者が受け取る点。裏ドラ・本場・供託を含まず、包の分け方も含まない）で、{@link ObservationSchema#POINT_UNIT} 点単位で書く。
 */
public final class OracleCodec {

  /** 相手の数（下家・対面・上家）。 */
  private static final int OPPONENTS = GameState.NUM_PLAYERS - 1;

  /** 相手の記録の配置（相手 1 席に 1 記録）。 */
  static final BitLayout<OpponentBits> OPPONENT = BitLayout.of(OpponentBits.class, Long.SIZE);

  /** 放銃の点の配置（放銃の牌 1 つに 1 記録）。 */
  static final BitLayout<PointBits> POINT = BitLayout.of(PointBits.class, Short.SIZE);

  /** 放銃の点の先頭（{@code long} の添字）。 */
  private static final int POINTS_START = OPPONENT.longs(OPPONENTS);

  /** 神託の {@code long} の最大（相手 3 席が全て国士の 13 面待ちで、全ての待ちでロンできる）。 */
  public static final int MAX_LONGS = POINTS_START + POINT.longs(OPPONENTS * Limits.MAX_WAIT_KINDS);

  private OracleCodec() {}

  /**
   * 神託の {@code long} の数を返す（放銃の牌の数から出す）。
   *
   * @param words 神託を置いた配列
   * @param base 神託の先頭
   * @return {@code long} の数
   * @throws IndexOutOfBoundsException 相手の記録が配列に収まらないとき
   */
  public static int longs(long[] words, int base) {
    Objects.checkFromIndexSize(base, POINTS_START, words.length);
    int dangers = 0;
    for (int opponent = 1; opponent <= OPPONENTS; opponent++) {
      dangers += Long.bitCount(danger(words, base, opponent));
    }
    return POINTS_START + POINT.longs(dangers);
  }

  /**
   * 相手の最小の向聴を返す。
   *
   * @param words 神託を置いた配列
   * @param base 神託の先頭
   * @param opponent 相手の相対席（下家 1・対面 2・上家 3）
   * @return 最小の向聴（0 は聴牌）
   */
  public static int shanten(long[] words, int base, int opponent) {
    return OPPONENT.get(words, base, record(opponent), OpponentBits.SHANTEN);
  }

  /**
   * 相手の形の待ちを返す。
   *
   * @param words 神託を置いた配列
   * @param base 神託の先頭
   * @param opponent 相手の相対席（下家 1・対面 2・上家 3）
   * @return 待ちの牌のマスク（ビットは牌。聴牌でなければ 0）
   */
  public static long waits(long[] words, int base, int opponent) {
    return tiles(words, base, record(opponent), OpponentBits.WAIT);
  }

  /**
   * 判断の席が今捨てたら相手がロンできる牌を返す。
   *
   * @param words 神託を置いた配列
   * @param base 神託の先頭
   * @param opponent 相手の相対席（下家 1・対面 2・上家 3）
   * @return 放銃の牌のマスク（ビットは牌）
   */
  public static long danger(long[] words, int base, int opponent) {
    return tiles(words, base, record(opponent), OpponentBits.DANGER);
  }

  /**
   * 放銃の点を返す。
   *
   * @param words 神託を置いた配列
   * @param base 神託の先頭
   * @param index 放銃の牌の番号（相手の順・相手の中は牌の順に、全ての相手の放銃の牌を数えた 0 始まりの番号）
   * @return そのロンの和了の点
   */
  public static int points(long[] words, int base, int index) {
    return POINT.get(words, base + POINTS_START, index, PointBits.POINTS)
        * ObservationSchema.POINT_UNIT;
  }

  /**
   * 神託の記録の指紋を返す。行の目録に書き、読むときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** 相手の記録を書く（神託は 0 で始まる前提）。 */
  static void writeOpponent(
      long[] words, int base, int opponent, int shanten, long waits, long danger) {
    int record = record(opponent);
    OPPONENT.put(words, base, record, OpponentBits.SHANTEN, shanten);
    writeTiles(words, base, record, OpponentBits.WAIT, waits);
    writeTiles(words, base, record, OpponentBits.DANGER, danger);
  }

  /** {@code index} 番目の放銃の点を書く。 */
  static void writePoints(long[] words, int base, int index, int points) {
    if (points % ObservationSchema.POINT_UNIT != 0) {
      throw new IllegalStateException(
          "points are not a multiple of " + ObservationSchema.POINT_UNIT + ": " + points);
    }
    POINT.put(
        words, base + POINTS_START, index, PointBits.POINTS, points / ObservationSchema.POINT_UNIT);
  }

  private static int record(int opponent) {
    return Objects.checkIndex(opponent - 1, OPPONENTS);
  }

  private static long tiles(long[] words, int base, int record, OpponentBits field) {
    long mask = 0L;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      if (OPPONENT.get(words, base, record, field, tile) != 0) {
        mask |= 1L << tile;
      }
    }
    return mask;
  }

  private static void writeTiles(
      long[] words, int base, int record, OpponentBits field, long mask) {
    if ((mask >>> Tile.NUM_TILE_TYPES) != 0L) {
      throw new IllegalArgumentException(field + " mask has bits beyond the tiles: " + mask);
    }
    for (long rest = mask; rest != 0L; rest &= rest - 1) {
      OPPONENT.put(words, base, record, field, Long.numberOfTrailingZeros(rest), 1);
    }
  }

  /** 相手 1 席の記録の欄。 */
  enum OpponentBits implements BitField {
    /** 形の待ち（繰り返しの番号は牌）。聴牌でなければ全て 0。 */
    WAIT(1, Tile.NUM_TILE_TYPES),
    /** 最小の向聴（0 は聴牌）。 */
    SHANTEN(Limits.MAX_SHANTEN, 1),
    /** 判断の席が今その牌を捨てたら、この相手がロンできる（繰り返しの番号は牌）。 */
    DANGER(1, Tile.NUM_TILE_TYPES);

    private final int max;
    private final int count;

    OpponentBits(int max, int count) {
      this.max = max;
      this.count = count;
    }

    @Override
    public int max() {
      return max;
    }

    @Override
    public int count() {
      return count;
    }
  }

  /** 放銃の牌 1 つの点の欄。 */
  enum PointBits implements BitField {
    /** そのロンの和了の点（{@link ObservationSchema#POINT_UNIT} 点単位）。 */
    POINTS;

    @Override
    public int max() {
      return Limits.MAX_WIN_POINTS / ObservationSchema.POINT_UNIT;
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE =
        SchemaDigest.named("epsilon-prism-v2-oracle")
            .layout(OPPONENT)
            .layout(POINT)
            .with("opponents", OPPONENTS)
            .with("tiles", Tile.NUM_TILE_TYPES)
            .with("pointUnit", ObservationSchema.POINT_UNIT)
            .hex();
  }
}
