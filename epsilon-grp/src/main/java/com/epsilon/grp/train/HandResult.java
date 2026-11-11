package com.epsilon.grp.train;

import com.epsilon.grp.Seats;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;

/**
 * 局の結果のコード。RoundOutcome を、その局の親から見た相対席で 1 つの long に詰め、また RoundOutcome に書き戻す。包・流し満貫・
 * ダブロン（2 人とも包の席を持てる）を全て表せる。フィールドは上の桁から、終わり方・和了者・放銃者・和了者の打点・その包の席・
 * 2 人目の和了者・その打点・その包の席・聴牌・流し満貫・立直の順で、合わせて 39 ビット。上の桁から並べるので、コードの大小は
 * 終わり方 → 和了者 → 放銃者 → 打点 → … の順で決まり、似た結果はコードの順で隣に並ぶ。
 *
 * <p>席のフィールドは相対席に 1 を足した値で、席が無ければ 0。打点は BaseClass の番号。聴牌・流し満貫・立直は相対席のビット。その
 * 終わり方で使わないフィールドは 0。
 *
 * <p>どの卓から精算しても点・局・本場・供託・終局が同じになる違いのうち、次の 2 つはコードに持たない。
 * <ol>
 *   <li>途中流局の種類。書き戻すときは、そのルールの最初の種類にする（精算も局の進みも種類に依らない）。
 *   <li>流し満貫のある荒牌での子の聴牌。聴牌は親の分だけ持つ。不聴罰符は無く、聴牌は親が続くかにだけ効く。
 * </ol>
 * 最初の和了者の立直はコードに残す（この形を生の形と呼ぶ）。払った 1,000 点は場の供託と一緒に戻るので点の動きは変わらないが、
 * 局の結果のネットワークは和了者の立直も確率に入れる。
 */
public final class HandResult {

  /** コードのフィールド。宣言の順に上の桁から並べる。ビットの位置は一度だけ求める。 */
  private enum Part {
    END(2),
    WINNER(3),
    LOSER(3),
    BASE(5),
    PAO(3),
    SECOND(3),
    SECOND_BASE(5),
    SECOND_PAO(3),
    TENPAI(4),
    NAGASHI(4),
    RIICHI(4);

    private final int bits;

    Part(int bits) {
      this.bits = bits;
    }

    private long max() {
      return (1L << bits) - 1;
    }
  }

  private static final Part[] PARTS = Part.values();

  /** フィールドごとのビットの位置（宣言の順）。 */
  private static final int[] SHIFTS = shifts();

  /** コードのビット数。 */
  public static final int BITS = SHIFTS[Part.END.ordinal()] + Part.END.bits;

  /** 親の相対席のビット。 */
  private static final int DEALER_BIT = 1;

  private static final EndKind[] ENDS = EndKind.values();
  private static final BaseClass[] BASES = BaseClass.values();

  /** コードから書き戻すときの途中流局の種類。ルールごとに、Variant の番号の順に並べる。 */
  private static final AbortiveDraw[] ABORTIVES = abortives();

  private HandResult() {}

  /**
   * 局の結果を、生の形のコードにする（最初の和了者の立直を残す）。
   *
   * @param outcome 局の結果
   * @param dealer その局の親の絶対席
   */
  public static long raw(RoundOutcome outcome, int dealer) {
    return encode(outcome, dealer);
  }

  /**
   * コードを局の結果に書き戻す。コードは {@link #valid} を満たすものに限る（確かめない）。メモリは確保しない。
   *
   * @param code コード
   * @param dealer その局の親の絶対席
   * @param into 書き戻す先
   */
  public static void place(long code, int dealer, RoundOutcome into) {
    Variant variant = into.variant();
    int seats = variant.seats();
    int riichi = absolute((int) read(code, Part.RIICHI), dealer, seats);
    switch (ENDS[(int) read(code, Part.END)]) {
      case TSUMO -> into.setTsumo(
          base(code, Part.BASE), seat(code, Part.WINNER, dealer, seats), seat(code, Part.PAO, dealer, seats), riichi);
      case RON -> {
        into.setRon(
            base(code, Part.BASE),
            seat(code, Part.WINNER, dealer, seats),
            seat(code, Part.LOSER, dealer, seats),
            seat(code, Part.PAO, dealer, seats),
            riichi);
        if (read(code, Part.SECOND) != 0) {
          into.addRonWinner(
              base(code, Part.SECOND_BASE),
              seat(code, Part.SECOND, dealer, seats),
              seat(code, Part.SECOND_PAO, dealer, seats));
        }
      }
      case EXHAUSTIVE -> into.setExhaustive(
          absolute((int) read(code, Part.TENPAI), dealer, seats),
          absolute((int) read(code, Part.NAGASHI), dealer, seats),
          riichi);
      case ABORTIVE -> into.setAbortive(ABORTIVES[variant.ordinal()], riichi);
    }
  }

  /** コードが表す局の終わり方。 */
  public static EndKind end(long code) {
    return ENDS[(int) read(code, Part.END)];
  }

  /** この局で立直が成立した席（親から見た相対席のビット）。 */
  public static int riichi(long code) {
    return (int) read(code, Part.RIICHI);
  }

  /**
   * そのルールで書き戻せるコードか。39 ビットに収まり、フィールドが終わり方に合い（使わないフィールドは 0）、途中流局の種類と
   * 流し満貫のときの子の聴牌を持たず、書き戻してから詰め直すと同じコードになるときだけ true。
   *
   * @param scratch 試しに書き戻す先。その局の結果のルールで確かめる。中身は書き換わる
   */
  public static boolean valid(long code, RoundOutcome scratch) {
    if (code < 0 || code >>> BITS != 0
        || read(code, Part.BASE) >= BASES.length || read(code, Part.SECOND_BASE) >= BASES.length) {
      return false;
    }
    try {
      place(code, 0, scratch);
    } catch (IllegalArgumentException | IllegalStateException | IndexOutOfBoundsException malformed) {
      return false;
    }
    return encode(scratch, 0) == code;
  }

  /** 生の形のコードに詰める。 */
  private static long encode(RoundOutcome outcome, int dealer) {
    int seats = outcome.variant().seats();
    long code = put(Part.END, outcome.kind().ordinal());
    int riichi = relative(outcome.riichiMask(), dealer, seats);
    switch (outcome.kind()) {
      case TSUMO, RON -> {
        code |= seatField(Part.WINNER, outcome.winner(0), dealer, seats)
            | put(Part.BASE, BaseClass.of(outcome.base(0)).ordinal())
            | seatField(Part.PAO, outcome.pao(0), dealer, seats);
        if (outcome.kind() == EndKind.RON) {
          code |= seatField(Part.LOSER, outcome.loser(), dealer, seats);
        }
        if (outcome.winnerCount() > 1) {
          code |= seatField(Part.SECOND, outcome.winner(1), dealer, seats)
              | put(Part.SECOND_BASE, BaseClass.of(outcome.base(1)).ordinal())
              | seatField(Part.SECOND_PAO, outcome.pao(1), dealer, seats);
        }
      }
      case EXHAUSTIVE -> {
        int tenpai = relative(outcome.tenpaiMask(), dealer, seats);
        int nagashi = relative(outcome.nagashiMask(), dealer, seats);
        if (nagashi != 0) {
          tenpai &= DEALER_BIT;
        }
        code |= put(Part.TENPAI, tenpai) | put(Part.NAGASHI, nagashi);
      }
      case ABORTIVE -> {
        // 途中流局の種類は持たない。
      }
    }
    return code | put(Part.RIICHI, riichi);
  }

  private static long put(Part part, long value) {
    return value << SHIFTS[part.ordinal()];
  }

  private static long read(long code, Part part) {
    return code >>> SHIFTS[part.ordinal()] & part.max();
  }

  /** 席のフィールドの値。席が無ければ 0、あれば親から見た相対席 ＋ 1。 */
  private static long seatField(Part part, int seat, int dealer, int seats) {
    return seat == RoundOutcome.NO_SEAT ? 0 : put(part, Seats.relative(dealer, seat, seats) + 1);
  }

  /** 席のフィールドが表す絶対席。フィールドが 0 なら NO_SEAT。 */
  private static int seat(long code, Part part, int dealer, int seats) {
    int field = (int) read(code, part);
    return field == 0 ? RoundOutcome.NO_SEAT : Seats.absolute(dealer, field - 1, seats);
  }

  private static int base(long code, Part part) {
    return BASES[(int) read(code, part)].base();
  }

  /** 絶対席のビットを、親から見た相対席のビットに直す（親の分だけ右へ回す）。 */
  private static int relative(int mask, int dealer, int seats) {
    return (mask >>> dealer | mask << seats - dealer) & (1 << seats) - 1;
  }

  /** 親から見た相対席のビットを、絶対席のビットに直す（親の分だけ左へ回す）。 */
  private static int absolute(int mask, int dealer, int seats) {
    return (mask << dealer | mask >>> seats - dealer) & (1 << seats) - 1;
  }

  private static int[] shifts() {
    int[] shifts = new int[PARTS.length];
    for (int index = PARTS.length - 2; index >= 0; index--) {
      shifts[index] = shifts[index + 1] + PARTS[index + 1].bits;
    }
    return shifts;
  }

  private static AbortiveDraw[] abortives() {
    Variant[] variants = Variant.values();
    AbortiveDraw[] draws = new AbortiveDraw[variants.length];
    for (Variant variant : variants) {
      for (AbortiveDraw draw : AbortiveDraw.values()) {
        if (variant.aborts(draw)) {
          draws[variant.ordinal()] = draw;
          break;
        }
      }
      if (draws[variant.ordinal()] == null) {
        throw new IllegalStateException(variant + " has no abortive draw");
      }
    }
    return draws;
  }
}
