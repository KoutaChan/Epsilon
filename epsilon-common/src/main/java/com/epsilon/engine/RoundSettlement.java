package com.epsilon.engine;

/**
 * 一局の精算結果。得点と次局への遷移を、局終了時に作る不変の値として保持する。
 *
 * <p>得点は個別のフィールドに保持し、配列が必要な場合だけ{@link #snapshotFinalScores()}で独立した配列を作る。
 */
public final class RoundSettlement {

  private final RoundResult result;
  private final int player0Score;
  private final int player1Score;
  private final int player2Score;
  private final int player3Score;
  private final RoundTransition transition;

  RoundSettlement(RoundResult result, int[] finalScores, RoundTransition transition) {
    this(result, finalScores[0], finalScores[1], finalScores[2], finalScores[3], transition);
  }

  RoundSettlement(
      RoundResult result,
      int player0,
      int player1,
      int player2,
      int player3,
      RoundTransition transition) {
    this.result = result;
    player0Score = player0;
    player1Score = player1;
    player2Score = player2;
    player3Score = player3;
    this.transition = transition;
  }

  /**
   * 点棒精算の根拠となった局結果を返す。
   *
   * @return 和了または流局を表す局結果
   */
  public RoundResult result() {
    return result;
  }

  /**
   * 精算後の全席の持ち点を返す。
   *
   * @return 席順どおりの持ち点の防御的コピー
   */
  public int[] snapshotFinalScores() {
    return new int[] {player0Score, player1Score, player2Score, player3Score};
  }

  /**
   * 指定席の精算後持ち点を返す。
   *
   * @param player 0始まりの席インデックス
   * @return 指定席の持ち点
   */
  public int finalScore(int player) {
    return switch (player) {
      case 0 -> player0Score;
      case 1 -> player1Score;
      case 2 -> player2Score;
      case 3 -> player3Score;
      default -> throw new IndexOutOfBoundsException("player=" + player);
    };
  }

  /**
   * 次局または終局へ進むための遷移情報を返す。
   *
   * @return 連荘、本場、供託および終局判定を含む遷移
   */
  public RoundTransition transition() {
    return transition;
  }
}
