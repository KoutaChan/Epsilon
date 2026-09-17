package com.epsilon.engine;

import java.util.Objects;

/**
 * 四人分の点棒増減。
 *
 * <dl>
 *   <dt>{@code player0}
 *   <dd>席0の点棒増減
 *   <dt>{@code player1}
 *   <dd>席1の点棒増減
 *   <dt>{@code player2}
 *   <dd>席2の点棒増減
 *   <dt>{@code player3}
 *   <dd>席3の点棒増減
 * </dl>
 */
public final class PointDelta {

  private final int player0;
  private final int player1;
  private final int player2;
  private final int player3;

  public PointDelta(int player0, int player1, int player2, int player3) {
    this.player0 = player0;
    this.player1 = player1;
    this.player2 = player2;
    this.player3 = player3;
  }

  public int player0() {
    return player0;
  }

  public int player1() {
    return player1;
  }

  public int player2() {
    return player2;
  }

  public int player3() {
    return player3;
  }

  /**
   * 指定席の点棒増減を返す。
   *
   * @param player 席番号（0-3）
   * @return 指定席の点棒増減
   * @throws IndexOutOfBoundsException 席番号が0-3の外なら発生
   */
  public int get(int player) {
    return switch (player) {
      case 0 -> player0;
      case 1 -> player1;
      case 2 -> player2;
      case 3 -> player3;
      default -> throw new IndexOutOfBoundsException(player);
    };
  }

  /**
   * 四席分を席番号順の配列へ変換する。
   *
   * @return 新しく生成した4要素配列
   */
  public int[] toArray() {
    return new int[] {player0, player1, player2, player3};
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof PointDelta delta
        && player0 == delta.player0
        && player1 == delta.player1
        && player2 == delta.player2
        && player3 == delta.player3;
  }

  @Override
  public int hashCode() {
    return Objects.hash(player0, player1, player2, player3);
  }

  @Override
  public String toString() {
    return "PointDelta[player0="
        + player0
        + ", player1="
        + player1
        + ", player2="
        + player2
        + ", player3="
        + player3
        + ']';
  }
}
