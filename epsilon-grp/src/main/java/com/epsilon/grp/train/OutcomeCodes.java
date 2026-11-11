package com.epsilon.grp.train;

import com.epsilon.grp.HandOutcomes;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;

/**
 * 記録した局の結果（HandResult の生の形のコード）を、局の結果のネットワーク q が持つ結果 1 つに直し、その対数確率の項の番号を
 * 書く。q は和了者の立直も確率に入れるので、生の形のコードから結果がちょうど 1 つに決まる。包のある和了と流し満貫は q で表せない。
 * 荒牌で、立直した席が聴牌していない結果も表せない（q の立直の項は聴牌の席にしか無い）。同時に使えるのは 1 スレッドだけ。
 */
public final class OutcomeCodes {

  private static final int NONE = HandOutcomes.NONE;

  private final HandOutcomes hands;
  private final int seats;

  /** コードを、親を 0 とする相対席で展開する作業用の結果。 */
  private final RoundOutcome scratch;

  /**
   * 区分ごとに、終わり方の番号を引く表。席は相対席。添字は、ツモなら和了者、ロンなら 和了者 × S ＋ 放銃者、ダブロンなら
   * (放銃者 × S ＋ 小さい方の和了者) × S ＋ 大きい方の和了者、荒牌なら聴牌の席のビット。
   */
  private final int[] tsumo;

  private final int[] ron;
  private final int[] doubleRon;
  private final int[] draw;
  private final int abortive;

  /** 最後に直した結果。{@link #decode} が true を返した後だけ読む。 */
  private int main;

  private int base;
  private int other;
  private int riichi;

  public OutcomeCodes(Variant variant) {
    hands = HandOutcomes.of(Objects.requireNonNull(variant, "variant"));
    seats = variant.seats();
    scratch = new RoundOutcome(variant);
    tsumo = absent(seats);
    ron = absent(seats * seats);
    doubleRon = absent(seats * seats * seats);
    draw = absent(1 << seats);
    int found = NONE;
    for (int main = 0; main < hands.mainCount(); main++) {
      switch (hands.section(main)) {
        case TSUMO -> tsumo[hands.winner(main)] = main;
        case RON -> ron[hands.winner(main) * seats + hands.loser(main)] = main;
        case DOUBLE_RON ->
            doubleRon[(hands.loser(main) * seats + hands.winner(main)) * seats + hands.secondWinner(main)] = main;
        case DRAW -> draw[hands.tenpai(main)] = main;
        case ABORTIVE -> found = main;
      }
    }
    abortive = found;
  }

  public Variant variant() {
    return hands.variant();
  }

  /**
   * コードが表す結果の項の番号（HandOutcomes の項の番号）を書く。書く順は、終わり方、和了ごとの打点、立直の項（立直の有無を
   * 確率に入れる席ごと）。
   *
   * @param code 局の結果の生の形のコード（{@link HandResult#valid} を満たすこと）
   * @param out 書き込む先
   * @param at 書き始める位置
   * @return 書いた項の数。q で表せない結果なら 0 で、何も書かない
   */
  public int terms(long code, short[] out, int at) {
    return decode(code) ? OutcomeRows.outcomeTerms(hands, main, base, other, riichi, out, at) : 0;
  }

  /**
   * 結果のコードを、q が持つ結果 1 つに直す。直した結果は {@link #main}・{@link #base}・{@link #other}・{@link #riichi} で読む
   * （次に直すまで有効）。
   *
   * @param code 局の結果の生の形のコード（{@link HandResult#valid} を満たすこと）
   * @return q で表せる結果なら true。表せなければ false で、そのときは直した結果を読まないこと
   */
  public boolean decode(long code) {
    HandResult.place(code, 0, scratch);
    riichi = scratch.riichiMask();
    base = NONE;
    other = NONE;
    switch (scratch.kind()) {
      case TSUMO -> {
        if (scratch.pao(0) != RoundOutcome.NO_SEAT) {
          return false;
        }
        main = tsumo[scratch.winner(0)];
        base = grade(0);
      }
      case RON -> {
        for (int index = 0; index < scratch.winnerCount(); index++) {
          if (scratch.pao(index) != RoundOutcome.NO_SEAT) {
            return false;
          }
        }
        int loser = scratch.loser();
        if (scratch.winnerCount() == 1) {
          main = ron[scratch.winner(0) * seats + loser];
          base = grade(0);
        } else {
          int lower = scratch.winner(0) < scratch.winner(1) ? 0 : 1;
          main = doubleRon[(loser * seats + scratch.winner(lower)) * seats + scratch.winner(1 - lower)];
          base = grade(lower);
          other = grade(1 - lower);
        }
      }
      case EXHAUSTIVE -> {
        int tenpai = scratch.tenpaiMask();
        if (scratch.nagashiMask() != 0 || (riichi & ~tenpai) != 0) {
          return false;
        }
        main = draw[tenpai];
      }
      case ABORTIVE -> main = abortive;
      default -> throw new IllegalStateException("Unknown ending " + scratch.kind());
    }
    return true;
  }

  /** 直した結果の、終わり方の番号（HandOutcomes の番号）。 */
  public int main() {
    return main;
  }

  /**
   * 直した結果の、和了者（ダブロンなら {@code HandOutcomes.winner} の席）の打点の番号。和了でなければ
   * {@code HandOutcomes.NONE}。
   */
  public int base() {
    return base;
  }

  /** 直した結果の、ダブロンの 2 人目の打点の番号。ダブロンでなければ {@code HandOutcomes.NONE}。 */
  public int other() {
    return other;
  }

  /** 直した結果の、立直が成立した席のビット（親から見た相対席）。 */
  public int riichi() {
    return riichi;
  }

  /** 展開した結果の index 番目の和了について、打点の番号（BaseClass の宣言の順）。 */
  private int grade(int index) {
    return BaseClass.of(scratch.base(index)).ordinal();
  }

  private static int[] absent(int length) {
    int[] table = new int[length];
    Arrays.fill(table, NONE);
    return table;
  }
}
