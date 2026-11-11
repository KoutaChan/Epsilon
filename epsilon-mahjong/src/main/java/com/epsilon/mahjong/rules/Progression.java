package com.epsilon.mahjong.rules;

import java.util.Arrays;
import java.util.Objects;

/** 局と半荘の進め方（天鳳）。どれもエンジンの状態を使わない純粋な関数で、メモリを確保しない。 */
public final class Progression {

  private Progression() {}

  /**
   * 局が終わって点が動いた後の卓を書く。半荘が続くなら次局の始まりを、終わるなら終局の卓を書く。終局の卓の局と本場は、いま終わった
   * 局のまま。
   *
   * <p>終局かどうかは精算の後の点（{@link #judgedScore}）で決め、終局なら残った供託をトップに渡す。親が流れたら次の局に進む。本場は、
   * 子だけが和了したら 0 に戻し、それ以外（親が続くか流局）では 1 増やす。
   *
   * @param start 終わった局の始まりの卓
   * @param outcome 局の結果
   * @param delta 席ごとの点の動き（{@link Settlement#apply} が書いた値）
   * @param into 書く先の卓。始まりの卓と同じでもよい
   * @throws IllegalArgumentException ルールが合わないとき、終局の卓を渡したとき、点の動きの和が供託の動きと合わないとき
   */
  public static void next(TableState start, RoundOutcome outcome, int[] delta, TableState into) {
    Variant variant = start.variant();
    if (outcome.variant() != variant || into.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: table " + variant + ", outcome " + outcome.variant() + ", into " + into.variant());
    }
    if (start.gameOver()) {
      throw new IllegalArgumentException("No round starts from a finished game: " + start);
    }
    Settlement.requireDelta(variant, delta);
    int round = start.round();
    int honba = start.honba();
    boolean win = outcome.kind().win();
    int remaining = win ? 0 : Settlement.deposits(start, outcome);
    long moved = 0;
    for (int points : delta) {
      moved += points;
    }
    if (moved != (long) Variant.DEPOSIT * (start.kyotaku() - remaining)) {
      throw new IllegalArgumentException(
          "Deltas " + Arrays.toString(delta) + " do not match the deposits of " + outcome + " from " + start);
    }
    DealerOutcome dealerOutcome = dealerOutcome(outcome, start.dealer());
    for (int seat = 0; seat < delta.length; seat++) {
      into.setScore(seat, judgedScore(start, delta, seat));
    }
    into.setRound(round, honba, remaining);
    if (finishes(into, dealerOutcome)) {
      into.finish();
    } else if (dealerOutcome == DealerOutcome.PASSES) {
      into.setRound(round + 1, win ? 0 : honba + 1, remaining);
    } else {
      into.setRound(round, honba + 1, remaining);
    }
  }

  /**
   * 精算の後の点。飛びと返し点の判定はこの点で行う。和了なら、場の供託を最初の和了者が受け取った後の点になる。流局なら、供託を場に
   * 残したままの点になる（点の動きがそう書かれている）。{@link #next} で書く先と始まりが同じ卓でも使えるように、その席の点だけを
   * 読む。
   *
   * @param start 終わった局の始まりの卓
   * @param delta 席ごとの点の動き（{@link Settlement#apply} が書いた値）
   * @param seat 席
   * @return その席の精算の後の点
   */
  public static int judgedScore(TableState start, int[] delta, int seat) {
    return start.score(seat) + delta[seat];
  }

  /**
   * 局の後に親が続くか流れるか。親が和了者の 1 人のときと、荒牌で親が聴牌のときに続く。途中流局でも続くが、和了止め・聴牌止めには
   * ならない。
   *
   * @param outcome 局の結果
   * @param dealer 親の席
   * @return 親が続くか、流れるか、途中流局で続くか
   */
  public static DealerOutcome dealerOutcome(RoundOutcome outcome, int dealer) {
    Objects.checkIndex(dealer, outcome.variant().seats());
    return switch (outcome.kind()) {
      case TSUMO, RON -> wins(outcome, dealer) ? DealerOutcome.CONTINUES : DealerOutcome.PASSES;
      case EXHAUSTIVE -> (outcome.tenpaiMask() & (1 << dealer)) != 0 ? DealerOutcome.CONTINUES : DealerOutcome.PASSES;
      case ABORTIVE -> DealerOutcome.ABORTIVE;
    };
  }

  /**
   * 半荘が終わるかどうか。次のどれかに当てはまれば終わる。
   *
   * <ul>
   *   <li>飛んだ席がある。
   *   <li>南場の最後の局以降で親が流れ、トップが返し点に届いているか、その局が西場の最後の局だった。
   *   <li>南場の最後の局以降で親が続き（途中流局を除く）、親がトップで返し点に届いている（和了止め・聴牌止め）。
   * </ul>
   *
   * <p>西場の最後の局でも、親が続く限りその局を続ける。途中流局では終わらない。トップは、同点なら起家に近い席。
   *
   * @param table 精算の後の卓（点は {@link #judgedScore} の値）。局はいま終わった局
   * @param outcome 局の後に親が続くか流れるか
   * @return 半荘が終わるなら true
   */
  public static boolean finishes(TableState table, DealerOutcome outcome) {
    return finishes(table.variant(), table.round(), table.scoreArray(), outcome);
  }

  /**
   * 卓を作らずに、点の配列から半荘が終わるかを決める。判定は {@link #finishes(TableState, DealerOutcome)} と同じ。メモリを確保せず、
   * 点と供託の和も確かめないので、仮定の点で何度でも呼べる。
   *
   * @param variant ルール
   * @param round いま終わった局の番号
   * @param scores 絶対席の順に並べた、精算の後の点（読むだけ）
   * @param dealer 局の後に親が続くか流れるか
   * @return 半荘が終わるなら true
   * @throws IllegalArgumentException 局の番号がルールの範囲の外のとき、または点の数が席の数と違うとき
   */
  public static boolean finishes(Variant variant, int round, int[] scores, DealerOutcome dealer) {
    if (round < 0 || round > variant.finalRound() || scores.length != variant.seats()) {
      throw new IllegalArgumentException(
          "Round " + round + " and " + scores.length + " scores do not fit " + variant);
    }
    for (int score : scores) {
      if (busts(score)) {
        return true;
      }
    }
    if (round < variant.lastRound()) {
      return false;
    }
    int top = Placement.top(scores);
    boolean reached = reachesReturn(variant, scores[top]);
    return switch (dealer) {
      case PASSES -> reached || round == variant.finalRound();
      case CONTINUES -> reached && top == variant.dealer(round);
      case ABORTIVE -> false;
    };
  }

  /** 0 点未満なら飛び。ちょうど 0 点なら続ける。 */
  public static boolean busts(int points) {
    return points < 0;
  }

  /** 点が返し点以上なら、返し点に届いている。 */
  public static boolean reachesReturn(Variant variant, int points) {
    return points >= variant.returnPoints();
  }

  private static boolean wins(RoundOutcome outcome, int seat) {
    for (int i = 0; i < outcome.winnerCount(); i++) {
      if (outcome.winner(i) == seat) {
        return true;
      }
    }
    return false;
  }
}
