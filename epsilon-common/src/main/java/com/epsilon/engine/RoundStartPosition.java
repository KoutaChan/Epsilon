package com.epsilon.engine;

/** Public conditions of a fresh round; concealed hands and wall are always generated afresh. */
public record RoundStartPosition(
    int kyokuIndex, int dealer, int honba, int kyotakuCount, int[] scores) {
  public RoundStartPosition {
    if (kyokuIndex < 0
        || dealer != Math.floorMod(kyokuIndex, 4)
        || honba < 0
        || kyotakuCount < 0
        || scores.length != 4) {
      throw new IllegalArgumentException("Invalid public round start");
    }
    scores = scores.clone();
    long total = 1000L * kyotakuCount;
    for (int score : scores) {
      if (score < 0 || score % 100 != 0) {
        throw new IllegalArgumentException("Round start score must be nonnegative and in hundreds");
      }
      total += score;
    }
    if (total != 100_000L) {
      throw new IllegalArgumentException("Scores and riichi deposits must total 100000");
    }
  }

  @Override
  public int[] scores() {
    return scores.clone();
  }
}
