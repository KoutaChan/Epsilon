package com.epsilon.engine;

/** 候補行動を適用した場合の4席分の点棒増減を保持する、再利用可能なバッファ。 */
final class PointDeltaBuffer {

  private int player0;
  private int player1;
  private int player2;
  private int player3;

  PointDeltaBuffer bind(int player0, int player1, int player2, int player3) {
    this.player0 = player0;
    this.player1 = player1;
    this.player2 = player2;
    this.player3 = player3;
    return this;
  }

  int player0() {
    return player0;
  }

  int player1() {
    return player1;
  }

  int player2() {
    return player2;
  }

  int player3() {
    return player3;
  }

  int get(int player) {
    return switch (player) {
      case 0 -> player0;
      case 1 -> player1;
      case 2 -> player2;
      case 3 -> player3;
      default -> throw new IndexOutOfBoundsException(player);
    };
  }
}
