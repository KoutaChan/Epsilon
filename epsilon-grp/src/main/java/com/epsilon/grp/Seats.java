package com.epsilon.grp;

/** GRP の席の変換。相対席は次局の親を 0 とし、絶対席は起家を 0 とする。 */
public final class Seats {

  private Seats() {}

  /** 相対席を絶対席に直す。 */
  public static int absolute(int dealer, int relative, int seats) {
    return (dealer + relative) % seats;
  }

  /** 絶対席を相対席に直す。 */
  public static int relative(int dealer, int seat, int seats) {
    return Math.floorMod(seat - dealer, seats);
  }

  /** n 席から 2 席を選ぶ組の数 n(n − 1)/2。 */
  public static int pairs(int count) {
    return count * (count - 1) / 2;
  }
}
