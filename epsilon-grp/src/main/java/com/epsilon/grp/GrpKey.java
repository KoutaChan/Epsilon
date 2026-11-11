package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * 卓のキーを作り、読む。キーは学習の行とメモに使う。次局・本場・供託と、絶対席ごとの点を 100 で割った値を 1 つの long に
 * 詰める（四人打ち 60 ビット、三人打ち 49 ビット）。収まらない卓（負の点、100 点の倍数でない点、フィールドの幅を超える値）は
 * 丸めずに NONE を返す。ルールはキーに入らないので、読む側がルールを知っている必要がある。
 */
public final class GrpKey {

  /** キーに収まらない卓を表す値。キーは 64 ビットより狭いので、どのキーとも重ならない。 */
  public static final long NONE = -1L;

  /** 点を詰める単位。 */
  private static final int POINT_UNIT = 100;

  /**
   * キーのフィールド。定数ごとにビットの幅を持ち、位置は宣言の順に幅を足して決める。点のフィールドは席の数だけ並ぶので
   * 最後に置く。
   */
  private enum Part {
    ROUND(4),
    HONBA(6),
    STICKS(6),
    SCORE(11);

    private final int bits;

    Part(int bits) {
      this.bits = bits;
    }

    private long max() {
      return (1L << bits) - 1;
    }
  }

  private static final Part[] PARTS = Part.values();

  /** フィールドごとのビットの位置（宣言の順）。点のフィールドは、席 0 の点の位置を指す。 */
  private static final int[] SHIFTS = shifts();

  /** 卓の型ごとに点を読む関数。何もキャプチャしないメソッド参照で渡すので、呼び出しごとにメモリを確保しない。 */
  @FunctionalInterface
  private interface Scores<T> {
    int score(T table, int seat);
  }

  private GrpKey() {}

  /** 半荘が続く卓のキーを返す。収まらなければ NONE を返す。 */
  public static long of(GrpState state) {
    return pack(state, state.variant().seats(), state.round(), state.honba(), state.sticks(), GrpState::score);
  }

  /** 半荘が続く卓のキーを返す。収まらなければ NONE を返す。終局した卓では IllegalArgumentException を投げる。 */
  public static long of(TableState table) {
    if (table.gameOver()) {
      throw new IllegalArgumentException("A finished game has no next hand: " + table);
    }
    return pack(table, table.variant().seats(), table.round(), table.honba(), table.kyotaku(), TableState::score);
  }

  /** キーのビット数を返す（四人打ち 60、三人打ち 49）。 */
  public static int bits(Variant variant) {
    return SHIFTS[Part.SCORE.ordinal()] + variant.seats() * Part.SCORE.bits;
  }

  /**
   * そのルールの卓のキーとして正しければ true を返す。NONE でなく、そのルールのビット数より上のビットが 0 で、局が西場の
   * 最後の局以前なら正しい。
   */
  public static boolean valid(Variant variant, long key) {
    return key != NONE && key >>> bits(variant) == 0 && round(key) <= variant.finalRound();
  }

  public static int round(long key) {
    return (int) read(key, Part.ROUND, SHIFTS[Part.ROUND.ordinal()]);
  }

  public static int honba(long key) {
    return (int) read(key, Part.HONBA, SHIFTS[Part.HONBA.ordinal()]);
  }

  public static int sticks(long key) {
    return (int) read(key, Part.STICKS, SHIFTS[Part.STICKS.ordinal()]);
  }

  /** キーから、指定した絶対席の点を読む。 */
  public static int score(long key, int seat) {
    return (int) read(key, Part.SCORE, scoreShift(seat)) * POINT_UNIT;
  }

  private static <T> long pack(T table, int seats, int round, int honba, int sticks, Scores<T> scores) {
    if (!fits(Part.ROUND, round) || !fits(Part.HONBA, honba) || !fits(Part.STICKS, sticks)) {
      return NONE;
    }
    long key = (long) round << SHIFTS[Part.ROUND.ordinal()]
        | (long) honba << SHIFTS[Part.HONBA.ordinal()]
        | (long) sticks << SHIFTS[Part.STICKS.ordinal()];
    for (int seat = 0; key != NONE && seat < seats; seat++) {
      key = withScore(key, seat, scores.score(table, seat));
    }
    return key;
  }

  private static long withScore(long key, int seat, int score) {
    if (score % POINT_UNIT != 0 || !fits(Part.SCORE, score / POINT_UNIT)) {
      return NONE;
    }
    return key | (long) (score / POINT_UNIT) << scoreShift(seat);
  }

  private static boolean fits(Part part, int value) {
    return value >= 0 && value <= part.max();
  }

  private static long read(long key, Part part, int shift) {
    return key >>> shift & part.max();
  }

  private static int scoreShift(int seat) {
    return SHIFTS[Part.SCORE.ordinal()] + seat * Part.SCORE.bits;
  }

  private static int[] shifts() {
    int[] shifts = new int[PARTS.length];
    for (int index = 1; index < PARTS.length; index++) {
      shifts[index] = shifts[index - 1] + PARTS[index - 1].bits;
    }
    return shifts;
  }
}
