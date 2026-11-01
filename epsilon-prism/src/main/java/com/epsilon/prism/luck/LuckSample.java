package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.prism.luck.LuckSchema.HeaderBits;
import java.util.Arrays;

/**
 * 運の網 Ω の学習の標本 1 件。結果は O° の頭の欄にあり、{@link #outcome} で出す（重ねて持たない）。Ω の c は当てはめのときに作るので持たない。
 *
 * <p>配列は作るときと返すときに写す（標本は値）。数の多い読み書きは
 * {@link LuckSampleWriter#append(long, long[], int, float[], int, float)} と {@link LuckSampleStore.Cursor} を使い、
 * この型は作らない。
 *
 * @param game 半荘 ID（保留は半荘ごとに分ける）
 * @param observation 観測 O°（長さはちょうど {@link LuckSchema#longs} の数）
 * @param y 局末の Φ（O° と同じく行為者から見た相対席の順に 4 席）
 * @param stratum 層の鍵（{@link LuckStratum}）
 * @param weight 重み（1/標本の率）
 */
public record LuckSample(long game, long[] observation, float[] y, int stratum, float weight) {

  private static final LuckEvent[] EVENTS = LuckEvent.values();

  /**
   * 標本を作る。
   *
   * @throws IllegalArgumentException O° の長さが形と合わないか、Y・層・重みが範囲の外のとき
   */
  public LuckSample {
    observation = observation.clone();
    y = y.clone();
    if (observation.length < LuckSchema.SEAT_START + LuckSchema.SEAT.longs(GameState.NUM_PLAYERS)
        || LuckSchema.longs(observation, 0) != observation.length) {
      throw new IllegalArgumentException("observation length does not match its schema");
    }
    requireValues(y, stratum, weight);
  }

  /** O° の写し。 */
  @Override
  public long[] observation() {
    return observation.clone();
  }

  /** Y の写し。 */
  @Override
  public float[] y() {
    return y.clone();
  }

  public LuckEvent event() {
    return EVENTS[LuckSchema.HEADER.get(observation, 0, 0, HeaderBits.EVENT)];
  }

  /** 結果の番号（{@link LuckSchema#outcome}）。 */
  public int outcome() {
    return LuckSchema.outcome(
        LuckSchema.HEADER.get(observation, 0, 0, HeaderBits.OUTCOME_TILE),
        LuckSchema.HEADER.get(observation, 0, 0, HeaderBits.OUTCOME_AKA) != 0);
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof LuckSample sample
        && game == sample.game
        && Arrays.equals(observation, sample.observation)
        && Arrays.equals(y, sample.y)
        && stratum == sample.stratum
        && Float.compare(weight, sample.weight) == 0;
  }

  @Override
  public int hashCode() {
    int hash = 31 * Long.hashCode(game) + Arrays.hashCode(observation);
    return 31 * (31 * (31 * hash + Arrays.hashCode(y)) + stratum) + Float.hashCode(weight);
  }

  @Override
  public String toString() {
    return "LuckSample[game="
        + game
        + ", event="
        + event()
        + ", outcome="
        + outcome()
        + ", y="
        + Arrays.toString(y)
        + ", stratum="
        + stratum
        + ", weight="
        + weight
        + ", longs="
        + observation.length
        + "]";
  }

  /**
   * 標本の値の範囲を確かめる。
   *
   * @throws IllegalArgumentException Y が 4 席で有限でないか、層・重みが範囲の外のとき
   */
  static void requireValues(float[] y, int stratum, float weight) {
    if (y.length != GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("y must have " + GameState.NUM_PLAYERS + " seats");
    }
    for (float value : y) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("y is not finite: " + Arrays.toString(y));
      }
    }
    if (stratum < 0 || stratum >= LuckStratum.COUNT) {
      throw new IllegalArgumentException("stratum is outside 0.." + (LuckStratum.COUNT - 1));
    }
    if (!(weight > 0) || !Float.isFinite(weight)) {
      throw new IllegalArgumentException("weight must be positive and finite: " + weight);
    }
  }
}
