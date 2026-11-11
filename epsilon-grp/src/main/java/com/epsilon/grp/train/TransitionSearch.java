package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.Seats;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.DealerOutcome;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * 記録した 1 局の遷移（局の始まりの卓 → 次の局の卓、または最終の点）を再現できる局の結果を探す。1 つのスレッドだけで使う。
 * 候補の結果は RoundOutcome に置き、精算（Settlement）と局の進行（Progression）にかけて、できた卓を目標と全ての値で比べる。
 * 和了者の点は打点が上がるほど増える（ScoreMath）ので、終わり方と立直の組み合わせごとに、和了者の点を手がかりに打点を二分探索する。
 *
 * <p>{@link #next} と {@link #last} は、局の結果のネットワークが表せる結果（HandOutcomes の終わり方 × 和了ごとの打点 × 立直の
 * 項を持つ席の立直）を探し、再現できた結果を全て残す。{@link #extras} は、そこに無い結果（包のある役満の和了と流し満貫。立直は
 * どの席でもよい）で再現できるかだけを見る。{@link #ending} は、半荘が終わるかの判定だけを記録を作ったエンジンのものに替えて、
 * 表せる結果を探す。こちらも再現できた結果を全て残す。
 */
final class TransitionSearch {

  private static final int NONE = HandOutcomes.NONE;
  private static final int FIRST_YAKUMAN = BaseClass.YAKUMAN.ordinal();
  private static final int LAST_YAKUMAN = HandOutcomes.BASES - 1;

  private final Variant variant;
  private final int seats;
  private final HandOutcomes hands;
  private final TableState start;
  private final TableState after;
  private final RoundOutcome outcome;
  private final int[] delta;
  private final int[] scores;

  /** 記録を作ったエンジンの判定で使う、精算の後の点。 */
  private final int[] judged;

  /** 探す目標。半荘が終わるかどうかと、続くなら次の局の番号・本場・供託・点、終わるなら最終の点。 */
  private boolean targetOver;
  private int targetRound;
  private int targetHonba;
  private int targetSticks;
  private final int[] targetScores;

  /** 見つけた結果。終わり方、打点の番号 2 つ（無ければ {@code NONE}）、立直した席のビット（相対席）。 */
  private int found;
  private int[] foundMain = new int[8];
  private int[] foundBase = new int[8];
  private int[] foundOther = new int[8];
  private int[] foundRiichi = new int[8];

  TransitionSearch(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    hands = HandOutcomes.of(variant);
    start = new TableState(variant);
    after = new TableState(variant);
    outcome = new RoundOutcome(variant);
    delta = new int[seats];
    scores = new int[seats];
    judged = new int[seats];
    targetScores = new int[seats];
  }

  /** 探す元を、半荘の始め（東 1 局 0 本場、供託なし、全員が配給原点）にする。 */
  void fromStart() {
    Arrays.fill(scores, variant.startPoints());
    start.set(0, 0, 0, scores);
  }

  /** 探す元を、半荘が続く局の始まりの卓にする。 */
  void from(GrpState state) {
    state.copyInto(start);
  }

  /** 次の局の卓を目標にして、局の結果のネットワークが表せる結果を探す。見つけた数を返す。 */
  int next(GrpState next) {
    if (next.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: search " + variant + ", state " + next.variant());
    }
    targetOver = false;
    targetRound = next.round();
    targetHonba = next.honba();
    targetSticks = next.sticks();
    for (int seat = 0; seat < seats; seat++) {
      targetScores[seat] = next.score(seat);
    }
    return search(false);
  }

  /** 元の局で半荘が終わり、最終の点（残りの供託をトップに渡した後）が指定した点になる結果を探す。見つけた数を返す。 */
  int last(int[] finalScores) {
    if (finalScores.length != seats) {
      throw new IllegalArgumentException(variant + " needs " + seats + " final scores, got " + finalScores.length);
    }
    targetOver = true;
    System.arraycopy(finalScores, 0, targetScores, 0, seats);
    return search(false);
  }

  /** 今の元から目標への遷移を、局の結果のネットワークが表せない結果（包のある役満の和了、流し満貫）で再現できれば true を返す。 */
  boolean extras() {
    return liable() || nagashi();
  }

  /** 今の元から目標への遷移を、包の席がある役満の和了（ツモかロン。立直はどの席でもよい）で再現できれば true を返す。 */
  boolean liable() {
    int all = 1 << seats;
    for (int riichi = 0; riichi < all; riichi++) {
      for (int winner = 0; winner < seats; winner++) {
        for (int pao = 0; pao < seats; pao++) {
          if (pao == winner) {
            continue;
          }
          if (liable(winner, RoundOutcome.NO_SEAT, pao, riichi)) {
            return true;
          }
          for (int loser = 0; loser < seats; loser++) {
            if (loser != winner && loser != pao && liable(winner, loser, pao, riichi)) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  /** 今の元から目標への遷移を、流し満貫（親は聴牌でも不聴でもよい。立直はどの席でもよい）で再現できれば true を返す。 */
  boolean nagashi() {
    int dealer = start.dealer();
    int all = 1 << seats;
    for (int riichi = 0; riichi < all; riichi++) {
      for (int nagashi = 1; nagashi < all; nagashi++) {
        for (int tenpai = 0; tenpai <= 1 << dealer; tenpai += 1 << dealer) {
          outcome.setExhaustive(tenpai, nagashi, riichi);
          apply();
          if (matches()) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * 半荘が終わるかの判定だけを記録を作ったエンジンのものに替えて、今の元から目標への遷移を、局の結果のネットワークが表せる結果で
   * 探す。再現できた結果は {@link #next} と同じく全て残し、見つけた数を返す。このエンジンの判定は、和了者が場の供託を受け取る前の
   * 点で飛びと返し点を見て、西場の最後の局ではどの終わり方でも半荘を終える。点の移動と、半荘が続くときの次局の決め方は Progression
   * と同じ。
   */
  int ending() {
    return search(true);
  }

  /** 最後に呼んだ {@link #next}・{@link #last}・{@link #ending} で見つけた結果の数。 */
  int found() {
    return found;
  }

  int main(int index) {
    return foundMain[index];
  }

  /** 和了者（ダブロンなら {@code HandOutcomes.winner} の席）の打点の番号。和了でなければ {@code NONE}。 */
  int base(int index) {
    return foundBase[index];
  }

  /** ダブロンのとき、{@code HandOutcomes.secondWinner} の席の打点の番号。ほかは {@code NONE}。 */
  int other(int index) {
    return foundOther[index];
  }

  /** 立直が成立した席のビット（相対席）。 */
  int riichi(int index) {
    return foundRiichi[index];
  }

  /**
   * 局の結果のネットワークが表せる結果を全て試し、再現できた結果を残す。
   *
   * @param recorded 半荘が終わるかを、記録を作ったエンジンの判定で決めるなら true、Progression の判定で決めるなら false
   * @return 見つけた数
   */
  private int search(boolean recorded) {
    found = 0;
    int dealer = start.dealer();
    for (int main = 0; main < hands.mainCount(); main++) {
      int factors = hands.factorCount(main);
      for (int mask = 0; mask < 1 << factors; mask++) {
        int relative = 0;
        int riichi = 0;
        for (int index = 0; index < factors; index++) {
          if ((mask & 1 << index) != 0) {
            int seat = hands.factorSeat(main, index);
            relative |= 1 << seat;
            riichi |= 1 << Seats.absolute(dealer, seat, seats);
          }
        }
        switch (hands.section(main)) {
          case TSUMO, RON -> {
            int base = bisect(main, riichi, NONE, Seats.absolute(dealer, hands.winner(main), seats), false);
            if (base != NONE && reproduces(recorded)) {
              record(main, base, NONE, relative);
            }
          }
          case DOUBLE_RON -> {
            int base = bisect(main, riichi, 0, Seats.absolute(dealer, hands.winner(main), seats), false);
            if (base != NONE) {
              int other = bisect(main, riichi, base, Seats.absolute(dealer, hands.secondWinner(main), seats), true);
              if (other != NONE && reproduces(recorded)) {
                record(main, base, other, relative);
              }
            }
          }
          case DRAW, ABORTIVE -> {
            hands.place(main, NONE, NONE, riichi, dealer, outcome);
            apply();
            if (reproduces(recorded)) {
              record(main, NONE, NONE, relative);
            }
          }
        }
      }
    }
    return found;
  }

  /**
   * 今の結果（{@link #apply} の後）が目標を再現すれば true を返す。
   *
   * @param recorded 半荘が終わるかを、記録を作ったエンジンの判定で決めるなら true、Progression の判定で決めるなら false
   */
  private boolean reproduces(boolean recorded) {
    return recorded ? endsLikeTheRecord() : matches();
  }

  /**
   * 指定した席の点が目標の点になる打点の番号を、二分探索で探す。無ければ {@code NONE} を返す。番号を返したときは、局の後の卓が
   * その番号での結果になっている。
   *
   * @param main 終わり方の番号
   * @param riichi 立直した席のビット（絶対席）
   * @param fixed もう一方の和了者の打点の番号（ダブロンのとき）
   * @param seat 点を比べる席（絶対席）
   * @param second ダブロンの 2 人目の打点を探すなら true
   */
  private int bisect(int main, int riichi, int fixed, int seat, boolean second) {
    int want = targetScores[seat];
    int dealer = start.dealer();
    int low = 0;
    int high = HandOutcomes.BASES - 1;
    while (low <= high) {
      int middle = (low + high) >>> 1;
      if (second) {
        hands.place(main, fixed, middle, riichi, dealer, outcome);
      } else {
        hands.place(main, middle, fixed, riichi, dealer, outcome);
      }
      apply();
      int score = after.score(seat);
      if (score == want) {
        return middle;
      }
      if (score < want) {
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return NONE;
  }

  /** 包の席がある役満の和了で再現できれば true を返す。放銃者が無ければツモ、あればロン。打点は役満の倍率を二分探索する。席は絶対席。 */
  private boolean liable(int winner, int loser, int pao, int riichi) {
    int want = targetScores[winner];
    int low = FIRST_YAKUMAN;
    int high = LAST_YAKUMAN;
    while (low <= high) {
      int middle = (low + high) >>> 1;
      if (loser == RoundOutcome.NO_SEAT) {
        outcome.setTsumo(HandOutcomes.base(middle), winner, pao, riichi);
      } else {
        outcome.setRon(HandOutcomes.base(middle), winner, loser, pao, riichi);
      }
      apply();
      int score = after.score(winner);
      if (score == want) {
        return matches();
      }
      if (score < want) {
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return false;
  }

  private void apply() {
    Settlement.apply(variant, start, outcome, delta);
    Progression.next(start, outcome, delta, after);
  }

  private boolean matches() {
    if (after.gameOver() != targetOver) {
      return false;
    }
    if (!targetOver
        && (after.round() != targetRound || after.honba() != targetHonba || after.kyotaku() != targetSticks)) {
      return false;
    }
    for (int seat = 0; seat < seats; seat++) {
      if (after.score(seat) != targetScores[seat]) {
        return false;
      }
    }
    return true;
  }

  /**
   * 今の結果（{@link #apply} の後）について、半荘が終わるかを記録を作ったエンジンの判定で決め直し、その卓が目標と同じなら true を返す。
   * 終わるなら、残った供託をトップに渡した点を最終の点と比べる。続くなら、次局の番号・本場・供託を Progression と同じ決め方で
   * 求めて比べる。
   */
  private boolean endsLikeTheRecord() {
    boolean win = outcome.kind().win();
    int deposits = start.kyotaku() + Integer.bitCount(outcome.riichiMask());
    for (int seat = 0; seat < seats; seat++) {
      judged[seat] = Progression.judgedScore(start, delta, seat);
    }
    DealerOutcome dealerOutcome = Progression.dealerOutcome(outcome, start.dealer());
    if (recordEnds(win ? outcome.winner(0) : RoundOutcome.NO_SEAT, deposits, dealerOutcome) != targetOver) {
      return false;
    }
    if (targetOver) {
      if (!win) {
        Placement.awardDeposits(judged, deposits);
      }
    } else {
      boolean passes = dealerOutcome == DealerOutcome.PASSES;
      int round = passes ? start.round() + 1 : start.round();
      int honba = passes && win ? 0 : start.honba() + 1;
      if (round != targetRound || honba != targetHonba || (win ? 0 : deposits) != targetSticks) {
        return false;
      }
    }
    return Arrays.equals(judged, targetScores);
  }

  /**
   * 記録を作ったエンジンの判定で半荘が終わるなら true を返す。見る点は、精算の後の点（{@code judged}）から、和了者が受け取った場の供託を
   * 引いたもの。
   *
   * @param receiver 場の供託を受け取った和了者。流局なら {@code NO_SEAT}
   * @param deposits 場の供託の本数
   * @param dealerOutcome 親の結果（親が流れる・連荘・途中流局）
   */
  private boolean recordEnds(int receiver, int deposits, DealerOutcome dealerOutcome) {
    if (receiver != RoundOutcome.NO_SEAT) {
      judged[receiver] -= Variant.DEPOSIT * deposits;
    }
    boolean ends = recordEnds(dealerOutcome);
    if (receiver != RoundOutcome.NO_SEAT) {
      judged[receiver] += Variant.DEPOSIT * deposits;
    }
    return ends;
  }

  private boolean recordEnds(DealerOutcome dealerOutcome) {
    for (int seat = 0; seat < seats; seat++) {
      if (Progression.busts(judged[seat])) {
        return true;
      }
    }
    int round = start.round();
    if (round >= variant.finalRound()) {
      return true;
    }
    if (round < variant.lastRound()) {
      return false;
    }
    int top = Placement.top(judged);
    int dealer = start.dealer();
    return switch (dealerOutcome) {
      case PASSES -> Progression.reachesReturn(variant, judged[top]);
      case CONTINUES -> top == dealer && Progression.reachesReturn(variant, judged[dealer]);
      case ABORTIVE -> false;
    };
  }

  private void record(int main, int base, int other, int riichi) {
    if (found == foundMain.length) {
      int capacity = 2 * found;
      foundMain = Arrays.copyOf(foundMain, capacity);
      foundBase = Arrays.copyOf(foundBase, capacity);
      foundOther = Arrays.copyOf(foundOther, capacity);
      foundRiichi = Arrays.copyOf(foundRiichi, capacity);
    }
    foundMain[found] = main;
    foundBase[found] = base;
    foundOther[found] = other;
    foundRiichi[found] = riichi;
    found++;
  }
}
