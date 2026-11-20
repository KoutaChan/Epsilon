package com.epsilon.grp;

import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * 和了者が精算の後に就く順位。和了者が 1 人の終わり方（ツモかロン）と打点ごとに、その局で誰も立直しなかったものとして精算
 * （Settlement）した後の点から、和了者の順位（Placement）を求める。場の供託と本場は精算に入る。打点が大きいほど和了者の点は増え、
 * ほかの席の点は増えないので、打点の番号が大きいほど順位の番号は小さいか同じになる。同時に使えるのは 1 スレッドだけ。
 */
public final class WinRanks {

  private final Variant variant;
  private final HandOutcomes hands;
  private final RoundOutcome outcome;
  private final int[] delta;
  private final int[] scores;

  public WinRanks(Variant variant) {
    this.variant = variant;
    hands = HandOutcomes.of(variant);
    outcome = new RoundOutcome(variant);
    delta = new int[variant.seats()];
    scores = new int[variant.seats()];
  }

  /**
   * 全ての打点について、和了者が精算の後に就く順位を書く。両端の打点の順位が同じ区間は、間の打点を精算せずに同じ順位で埋める。
   *
   * @param start 局の始まりの卓（半荘が続くこと）
   * @param main 和了者が 1 人の終わり方（ツモかロン）の番号
   * @param out 書き先。位置は打点の番号で、長さは打点の種類の数
   */
  public void ranks(TableState start, int main, int[] out) {
    int last = HandOutcomes.BASES - 1;
    fill(start, main, 0, rank(start, main, 0), last, rank(start, main, last), out);
  }

  /** 両端の順位が分かっている打点の区間を埋める。 */
  private void fill(
      TableState start, int main, int low, int lowRank, int high, int highRank, int[] out) {
    if (lowRank == highRank) {
      Arrays.fill(out, low, high + 1, lowRank);
      return;
    }
    out[low] = lowRank;
    out[high] = highRank;
    if (high - low > 1) {
      int middle = (low + high) >>> 1;
      int middleRank = rank(start, main, middle);
      fill(start, main, low, lowRank, middle, middleRank, out);
      fill(start, main, middle, middleRank, high, highRank, out);
    }
  }

  /**
   * 和了者が精算の後に就く順位を返す。
   *
   * @param start 局の始まりの卓（半荘が続くこと）
   * @param main 和了者が 1 人の終わり方（ツモかロン）の番号
   * @param base 打点の番号
   * @return 順位（0 始まり）
   */
  public int rank(TableState start, int main, int base) {
    int dealer = start.dealer();
    hands.place(main, base, HandOutcomes.NONE, 0, dealer, outcome);
    Settlement.apply(variant, start, outcome, delta);
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = start.score(seat) + delta[seat];
    }
    return Placement.rank(scores, Seats.absolute(dealer, hands.winner(main), scores.length));
  }
}
