package com.epsilon.grp;

import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * 次局の結果の一覧表。ルールごとに 1 回だけ作り、どのメソッドもメモリを確保しない。席は次局の親を 0 とする相対席。局の終わり方の
 * 番号は、区分（OutcomeSection）を宣言の順に並べ、区分の中では各定数の説明に書いた順に振る。打点の番号は BaseClass の宣言の順。
 * q(o|s) の対数確率の項の位置は、OutcomeTerm を宣言の順に並べて決める。
 *
 * <p>立直の有無を確率に入れる席とは、終わり方が決まった上で、立直したかどうかが確率の変数になる席のこと。和了と途中流局では
 * 全ての席、荒牌では聴牌した席だけがそうなる。
 */
public final class HandOutcomes {

  /** 打点の種類の数（BaseClass の値の数）。 */
  public static final int BASES = BaseClass.values().length;

  /** 席が無いことを表す値。和了でない結果の和了者や、ロンでない結果の放銃者に使う。 */
  public static final int NONE = -1;

  private static final BaseClass[] BASE_CLASSES = BaseClass.values();
  private static final OutcomeTerm[] TERMS = OutcomeTerm.values();
  private static final int METHODS = WinMethod.values().length;
  private static final int ROLES = RiichiRole.values().length;
  private static final Map<Variant, HandOutcomes> TABLES = tables();

  private final Variant variant;
  private final int seats;
  private final int mainCount;

  /** 途中流局を RoundOutcome に書き込むときの種類（そのルールにある最初の種類）。どの種類でも点の動きと局の進み方は同じ。 */
  private final AbortiveDraw abortive;

  /** 終わり方ごとの区分、和了者、2 人目の和了者、放銃者、聴牌した席のビット。 */
  private final OutcomeSection[] sections;
  private final int[] winners;
  private final int[] secondWinners;
  private final int[] losers;
  private final int[] tenpaiMasks;

  /** 終わり方と席ごとの立場。並びは [終わり方 × S ＋ 席] で、立直の有無を確率に入れない席は null。 */
  private final RiichiRole[] roles;

  /**
   * 終わり方ごとの、立直の有無を確率に入れる席の数と、その席の一覧。i 番目の席（相対席の小さい順）を [終わり方 × S ＋ i] に
   * 置く。
   */
  private final int[] factorCounts;
  private final int[] factorSeats;

  /** 項のまとまりごとの開始位置（OutcomeTerm の宣言の順）。最後の要素は項の数。 */
  private final int[] termStarts;

  /**
   * 1 つの結果の対数確率に入る項の最大数。内訳は、終わり方の 1 項、和了者ごとの打点の項、立直の有無を確率に入れる席ごとの
   * 立直の項。
   */
  private final int maxTerms;

  private HandOutcomes(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    mainCount = OutcomeSection.total(variant);
    abortive = firstAbortive(variant);
    sections = new OutcomeSection[mainCount];
    winners = new int[mainCount];
    secondWinners = new int[mainCount];
    losers = new int[mainCount];
    tenpaiMasks = new int[mainCount];
    roles = new RiichiRole[mainCount * seats];
    factorCounts = new int[mainCount];
    factorSeats = new int[mainCount * seats];
    int main = 0;
    for (int winner = 0; winner < seats; winner++) {
      main = put(main, OutcomeSection.TSUMO, winner, NONE, NONE, 0);
    }
    for (int winner = 0; winner < seats; winner++) {
      for (int loser = 0; loser < seats; loser++) {
        if (loser != winner) {
          main = put(main, OutcomeSection.RON, winner, NONE, loser, 0);
        }
      }
    }
    for (int loser = 0; loser < seats; loser++) {
      for (int first = 0; first < seats; first++) {
        for (int second = first + 1; second < seats; second++) {
          if (first != loser && second != loser) {
            main = put(main, OutcomeSection.DOUBLE_RON, first, second, loser, 0);
          }
        }
      }
    }
    for (int mask = 0; mask < 1 << seats; mask++) {
      main = put(main, OutcomeSection.DRAW, NONE, NONE, NONE, mask);
    }
    main = put(main, OutcomeSection.ABORTIVE, NONE, NONE, NONE, 0);
    int expected = 0;
    for (OutcomeSection section : OutcomeSection.values()) {
      for (int index = 0; index < section.count(variant); index++) {
        if (sections[expected++] != section) {
          throw new IllegalStateException("Main outcome " + (expected - 1) + " is out of the declared order");
        }
      }
    }
    if (main != mainCount) {
      throw new IllegalStateException("Built " + main + " main outcomes, declared " + mainCount);
    }
    termStarts = new int[TERMS.length + 1];
    for (OutcomeTerm term : TERMS) {
      termStarts[term.ordinal() + 1] = termStarts[term.ordinal()] + term.width(variant);
    }
    int most = 0;
    for (int index = 0; index < mainCount; index++) {
      int winnerCount = (winners[index] == NONE ? 0 : 1) + (secondWinners[index] == NONE ? 0 : 1);
      most = Math.max(most, 1 + winnerCount + factorCounts[index]);
    }
    maxTerms = most;
  }

  public static HandOutcomes of(Variant variant) {
    return TABLES.get(Objects.requireNonNull(variant, "variant"));
  }

  public Variant variant() {
    return variant;
  }

  /** 終わり方の数を返す（四人打ちで 45）。 */
  public int mainCount() {
    return mainCount;
  }

  public OutcomeSection section(int main) {
    return sections[main];
  }

  /** 和了者を返す。ダブロンなら 2 人のうち相対席の小さい方。和了でなければ NONE。 */
  public int winner(int main) {
    return winners[main];
  }

  /** ダブロンで、相対席の大きい方の和了者を返す。ほかの結果では NONE。 */
  public int secondWinner(int main) {
    return secondWinners[main];
  }

  /** ロンとダブロンの放銃者を返す。ほかの結果では NONE。 */
  public int loser(int main) {
    return losers[main];
  }

  /** 荒牌で聴牌した席のビット（相対席）を返す。ほかの結果では 0。 */
  public int tenpai(int main) {
    return tenpaiMasks[main];
  }

  /**
   * その終わり方での席の立場を返す。立直の有無を確率に入れない席は null。
   *
   * @param main 終わり方の番号
   * @param seat 相対席
   */
  public RiichiRole role(int main, int seat) {
    Objects.checkIndex(seat, seats);
    return roles[main * seats + seat];
  }

  /** その終わり方で、立直の有無を確率に入れる席の数を返す。 */
  public int factorCount(int main) {
    return factorCounts[main];
  }

  /**
   * 立直の有無を確率に入れる席を 1 つ返す。
   *
   * @param main 終わり方の番号
   * @param index 何番目の席か（0 始まり、相対席の小さい順）
   * @return 相対席
   */
  public int factorSeat(int main, int index) {
    Objects.checkIndex(index, factorCounts[main]);
    return factorSeats[main * seats + index];
  }

  /** 1 つの結果の対数確率に入る項の最大数を返す（四人打ちで 7）。 */
  public int maxTerms() {
    return maxTerms;
  }

  /** 打点の番号（BaseClass の宣言の順）に対応する基本点を返す。 */
  public static int base(int index) {
    return BASE_CLASSES[index].base();
  }

  /**
   * 終わり方を局の結果に書き込む。席は、次局の親から数えた相対席を絶対席に直して書く。
   *
   * @param main 終わり方の番号
   * @param base 和了者の打点の番号。ダブロンなら相対席の小さい方の和了者の打点
   * @param other ダブロンで、相対席の大きい方の和了者の打点の番号。ほかの結果では使わない
   * @param riichi この局で立直が成立した絶対席のビット
   * @param dealer 次局の親の絶対席
   * @param outcome 書き込む先
   */
  public void place(int main, int base, int other, int riichi, int dealer, RoundOutcome outcome) {
    switch (sections[main]) {
      case TSUMO -> outcome.setTsumo(base(base), absolute(dealer, winners[main]), RoundOutcome.NO_SEAT, riichi);
      case RON -> outcome.setRon(
          base(base), absolute(dealer, winners[main]), absolute(dealer, losers[main]), RoundOutcome.NO_SEAT, riichi);
      case DOUBLE_RON -> {
        outcome.setRon(
            base(base), absolute(dealer, winners[main]), absolute(dealer, losers[main]), RoundOutcome.NO_SEAT, riichi);
        outcome.addRonWinner(base(other), absolute(dealer, secondWinners[main]), RoundOutcome.NO_SEAT);
      }
      case DRAW -> {
        int tenpai = 0;
        for (int seat = 0; seat < seats; seat++) {
          if ((tenpaiMasks[main] & 1 << seat) != 0) {
            tenpai |= 1 << absolute(dealer, seat);
          }
        }
        outcome.setExhaustive(tenpai, 0, riichi);
      }
      case ABORTIVE -> outcome.setAbortive(abortive, riichi);
    }
  }

  private int absolute(int dealer, int relative) {
    return Seats.absolute(dealer, relative, seats);
  }

  /** 項のまとまりの開始位置を返す。 */
  public int termStart(OutcomeTerm term) {
    return termStarts[term.ordinal()];
  }

  /** 項の数を返す（四人打ちで 301）。 */
  public int terms() {
    return termStarts[TERMS.length];
  }

  /** ネットワークが出すロジットの数を返す。終わり方・打点・「立直した」の 3 つの項のまとまりの分で、四人打ちで 277。 */
  public int logits() {
    return termStarts[OutcomeTerm.NO_RIICHI.ordinal()];
  }

  public int mainTerm(int main) {
    return termStarts[OutcomeTerm.MAIN.ordinal()] + Objects.checkIndex(main, mainCount);
  }

  /** 和了者の相対席、和了の形、打点の番号に対応する打点の項の位置を返す。 */
  public int gradeTerm(int seat, WinMethod method, int base) {
    Objects.checkIndex(seat, seats);
    Objects.checkIndex(base, BASES);
    return termStarts[OutcomeTerm.GRADE.ordinal()] + (seat * METHODS + method.ordinal()) * BASES + base;
  }

  /**
   * 相対席と立場に対応する立直の項の位置を返す。
   *
   * @param declared true なら「立直した」の項、false なら「立直していない」の項
   */
  public int riichiTerm(int seat, RiichiRole role, boolean declared) {
    Objects.checkIndex(seat, seats);
    OutcomeTerm term = declared ? OutcomeTerm.RIICHI : OutcomeTerm.NO_RIICHI;
    return termStarts[term.ordinal()] + seat * ROLES + role.ordinal();
  }

  private int put(int main, OutcomeSection section, int winner, int second, int loser, int tenpai) {
    sections[main] = section;
    winners[main] = winner;
    secondWinners[main] = second;
    losers[main] = loser;
    tenpaiMasks[main] = tenpai;
    int factors = 0;
    for (int seat = 0; seat < seats; seat++) {
      RiichiRole role = switch (section) {
        case TSUMO -> seat == winner ? RiichiRole.WINNER : RiichiRole.PAYER;
        case RON, DOUBLE_RON -> seat == winner || seat == second
            ? RiichiRole.WINNER
            : seat == loser ? RiichiRole.LOSER : RiichiRole.BYSTANDER;
        case DRAW -> (tenpai & 1 << seat) != 0 ? RiichiRole.TENPAI : null;
        case ABORTIVE -> RiichiRole.ABORTIVE;
      };
      roles[main * seats + seat] = role;
      if (role != null) {
        factorSeats[main * seats + factors++] = seat;
      }
    }
    factorCounts[main] = factors;
    return main + 1;
  }

  private static AbortiveDraw firstAbortive(Variant variant) {
    for (AbortiveDraw draw : AbortiveDraw.values()) {
      if (variant.aborts(draw)) {
        return draw;
      }
    }
    throw new IllegalStateException(variant + " has no abortive draw");
  }

  private static Map<Variant, HandOutcomes> tables() {
    Map<Variant, HandOutcomes> tables = new EnumMap<>(Variant.class);
    for (Variant variant : Variant.values()) {
      tables.put(variant, new HandOutcomes(variant));
    }
    return tables;
  }
}
