package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.rules.Variant;

/**
 * 天鳳の局の番号（INIT の seed の先頭）と、ルールの局の番号とを変換する。天鳳は三人打ちでも四人打ちと同じ番号で書き、場風 × 4 ＋ 局
 * とする（東 1 局 = 0）。ルールの局の番号は場風 × 席の数 ＋ 局なので、四人打ちなら同じ値になる。
 */
public final class TenhouRoundIndex {

  /** 天鳳の番号で、場風 1 つが占める局の数（三人打ちでも 4）。 */
  private static final int WIND_STRIDE = 4;

  private TenhouRoundIndex() {}

  /** 天鳳の番号を、ルールの局の番号に直す。そのルールに無い局なら IllegalArgumentException を投げる。 */
  public static int round(int tenhou, Variant variant) {
    int seats = variant.seats();
    int round = tenhou / WIND_STRIDE * seats + tenhou % WIND_STRIDE;
    if (tenhou < 0 || tenhou % WIND_STRIDE >= seats || round > variant.finalRound()) {
      throw new IllegalArgumentException("Tenhou round " + tenhou + " is not a round of " + variant);
    }
    return round;
  }

  /** ルールの局の番号を、天鳳の番号に直す。そのルールに無い局なら IllegalArgumentException を投げる。 */
  public static int tenhou(int round, Variant variant) {
    if (round < 0 || round > variant.finalRound()) {
      throw new IllegalArgumentException("Round " + round + " is not a round of " + variant);
    }
    int seats = variant.seats();
    return round / seats * WIND_STRIDE + round % seats;
  }
}
