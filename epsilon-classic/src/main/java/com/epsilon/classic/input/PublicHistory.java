package com.epsilon.classic.input;

import java.util.Arrays;

/**
 * 公開の履歴。Major の入力で、状態を表す整数の行の末尾に置く。クエリの席（相対席の順に 4 つ）ごとに、河と面子のトークンの時点が
 * クエリの席の立直・副露・槓の時点とどれだけ前後するかを書く。トークンは 112 個で、並びは状態の行の河・面子と同じ。この値は
 * 埋め込みに入れない。
 *
 * <p>時点は、判断の記録の中のイベントの番号で表す。トークンの元の時点は、打牌したときか面子が成立したとき（加槓したポンはポンが
 * 成立したとき）。後の時点は加槓したときで、ほかのトークンには無い。クエリの席の時点は、立直なら宣言牌、副露なら暗槓でない面子が
 * 最後に成立したとき、槓なら最後に槓子ができたとき。1 トークンの値は {@link Nibble} を宣言の順に 4 bit ずつ詰めたもので、下の 16 bit と
 * 上の 16 bit を 2 つの要素に分けて書く。トークンの無い位置は 0。
 */
final class PublicHistory {

  /** 時点が無いことを表す値。後の時点が無いトークンと、クエリの席にその時点が無いときに使う。 */
  static final int ABSENT = -2;

  /** 立直しているのに宣言牌が河に無く、時点が分からないことを表す値。 */
  static final int UNKNOWN = -1;

  private static final int NIBBLE_BITS = 4;

  private PublicHistory() {}

  /** 1 トークンの値の中の、4 bit ずつのフィールド。 */
  enum Nibble {
    /** 元の時点と、クエリの席の立直の関係。 */
    ORIGIN_RIICHI,
    /** 元の時点と、クエリの席の副露の関係。 */
    ORIGIN_OPEN_MELD,
    /** 元の時点と、クエリの席の槓の関係。 */
    ORIGIN_KAN,
    /** 後の時点と、クエリの席の立直の関係。 */
    UPDATE_RIICHI,
    UPDATE_OPEN_MELD,
    UPDATE_KAN,
    /** クエリの席から見た、トークンの席の相対席。 */
    QUERY_RELATIVE_AUTHOR;

    int shift() {
      return ordinal() * NIBBLE_BITS;
    }
  }

  /**
   * 2 つの時点の関係。距離は、トークンの時点からクエリの席の時点を引いた値。距離の区間は、定数を宣言の順に見て、距離が上端以下になる
   * 最初の定数で表す。
   */
  enum Relation {
    LE_MINUS_8(-8),
    MINUS_7_TO_4(-4),
    MINUS_3_TO_2(-2),
    MINUS_1(-1),
    ZERO(0),
    PLUS_1(1),
    PLUS_2_TO_3(3),
    PLUS_4_TO_7(7),
    GE_8(Integer.MAX_VALUE),
    /** クエリの席の時点が無い。 */
    NO_ANCHOR(Integer.MAX_VALUE),
    /** トークンの時点が無い。 */
    NO_UPDATE(Integer.MAX_VALUE),
    /** どちらかの時点が分からない。 */
    UNKNOWN(Integer.MAX_VALUE);

    private static final Relation[] DISTANCES = {
      LE_MINUS_8, MINUS_7_TO_4, MINUS_3_TO_2, MINUS_1, ZERO, PLUS_1, PLUS_2_TO_3, PLUS_4_TO_7, GE_8
    };

    private final int upper;

    Relation(int upper) {
      this.upper = upper;
    }

    /**
     * トークンの時点とクエリの席の時点の関係。距離を見る前に、トークンの時点が無いか、クエリの席の時点が無いか、どちらかが分からないかを、
     * この順に調べる。
     *
     * @param event トークンの時点
     * @param anchor クエリの席の時点
     */
    static Relation of(int event, int anchor) {
      if (event == ABSENT) {
        return NO_UPDATE;
      }
      if (anchor == ABSENT) {
        return NO_ANCHOR;
      }
      if (event == PublicHistory.UNKNOWN || anchor == PublicHistory.UNKNOWN) {
        return UNKNOWN;
      }
      int distance = event - anchor;
      for (Relation relation : DISTANCES) {
        if (distance <= relation.upper) {
          return relation;
        }
      }
      throw new IllegalStateException("No relation for distance " + distance);
    }
  }

  /**
   * 1 トークンの値。
   *
   * @param origin 元の時点
   * @param update 後の時点（無ければ {@link #ABSENT}）
   * @param riichi クエリの席の立直の時点
   * @param openMeld クエリの席の副露の時点
   * @param kan クエリの席の槓の時点
   * @param author クエリの席から見た、トークンの席の相対席
   * @return {@link Nibble} を 4 bit ずつ詰めた値
   */
  static int pack(int origin, int update, int riichi, int openMeld, int kan, int author) {
    return Relation.of(origin, riichi).ordinal() << Nibble.ORIGIN_RIICHI.shift()
        | Relation.of(origin, openMeld).ordinal() << Nibble.ORIGIN_OPEN_MELD.shift()
        | Relation.of(origin, kan).ordinal() << Nibble.ORIGIN_KAN.shift()
        | Relation.of(update, riichi).ordinal() << Nibble.UPDATE_RIICHI.shift()
        | Relation.of(update, openMeld).ordinal() << Nibble.UPDATE_OPEN_MELD.shift()
        | Relation.of(update, kan).ordinal() << Nibble.UPDATE_KAN.shift()
        | author << Nibble.QUERY_RELATIVE_AUTHOR.shift();
  }

  /** フィンガープリントに使う記述。フィールドと関係の定数の並びと、要素の数を書く。 */
  static String describe() {
    return "nibbles="
        + Arrays.toString(Nibble.values())
        + ";relations="
        + Arrays.toString(Relation.values())
        + ";tokens="
        + ClassicSchema.HISTORY_TOKENS
        + "x"
        + ClassicSchema.HISTORY_WORDS;
  }
}
