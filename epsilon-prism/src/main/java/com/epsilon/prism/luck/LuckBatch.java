package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.prism.luck.LuckSchema.HeaderBits;
import com.epsilon.prism.observe.ObservationArena;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * 1 本の経路の 1 局で引いた標本（O°・層の鍵・重み）。局末の Y（4 席の Φ）が出るまで持ち、{@link #settle} で Y を付けて書き手に渡す。
 *
 * <p>呼ぶ場所: {@code GameRunner} の endHand が {@link LuckCapture#detach} で受け取り、その局の
 * {@code HandEnd} と一緒にドライバへ渡す。{@code CollectionDriver} はその {@code HandEnd} の局末の Φ を求めたところで
 * {@link #settle} を 1 回呼ぶ。Y は実現した局末の Φ（その経路の MC の目標）なので、局末の値を持たない経路（主経路で枠のない局）の
 * 標本を捨てると結果に依る選びになる。主経路では枠がなくても局末の Φ を求めること。
 *
 * <p>作るのは {@link LuckCapture} だけ。書くのは持ち主のスレッドだけで、{@link #settle} は受け渡しの後にドライバのスレッドが呼ぶ。
 */
public final class LuckBatch {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int INITIAL_LONGS = 512;
  private static final int INITIAL_SAMPLES = 8;

  private final long game;
  private final ObservationArena arena = new ObservationArena(INITIAL_LONGS);
  private final float[] y = new float[SEATS];
  private int[] bases = new int[INITIAL_SAMPLES];
  private int[] strata = new int[INITIAL_SAMPLES];
  private float[] weights = new float[INITIAL_SAMPLES];
  private int size;

  /**
   * 束を作る。
   *
   * @param game 半荘 ID（経路は 1 つの半荘に属する）
   */
  LuckBatch(long game) {
    this.game = game;
  }

  public int size() {
    return size;
  }

  /**
   * 局末の Φ を Y にして、全ての標本を書き手に渡す。Y は O° と同じく行為者から見た相対席の順に並べる。
   *
   * @param phi 局末の Φ（絶対席の順に 4 席）
   * @param writer 書き手
   * @throws IOException 書き手が書けなかったとき
   * @throws IllegalArgumentException Φ が 4 席でないか、有限でないとき
   */
  public void settle(double[] phi, LuckSampleWriter writer) throws IOException {
    if (phi.length != SEATS) {
      throw new IllegalArgumentException("phi must have " + SEATS + " seats: " + phi.length);
    }
    for (double value : phi) {
      if (!Double.isFinite(value)) {
        throw new IllegalArgumentException("phi is not finite: " + Arrays.toString(phi));
      }
    }
    long[] words = arena.words();
    for (int sample = 0; sample < size; sample++) {
      int actor = LuckSchema.HEADER.get(words, bases[sample], 0, HeaderBits.ACTOR);
      for (int seat = 0; seat < SEATS; seat++) {
        y[seat] = (float) phi[(actor + seat) % SEATS];
      }
      writer.append(game, words, bases[sample], y, strata[sample], weights[sample]);
    }
  }

  /** O° を書く置き場。 */
  ObservationArena arena() {
    return arena;
  }

  /** 書いた O° を標本に足す。 */
  void add(int base, int stratum, float weight) {
    if (size == bases.length) {
      int capacity = size * 2;
      bases = Arrays.copyOf(bases, capacity);
      strata = Arrays.copyOf(strata, capacity);
      weights = Arrays.copyOf(weights, capacity);
    }
    bases[size] = base;
    strata[size] = stratum;
    weights[size] = weight;
    size++;
  }

  /** 標本 {@code sample} の O° の先頭（{@link #arena} の中）。 */
  int base(int sample) {
    return bases[Objects.checkIndex(sample, size)];
  }

  /** 標本 {@code sample} の層の鍵。 */
  int stratum(int sample) {
    return strata[Objects.checkIndex(sample, size)];
  }

  /** 標本 {@code sample} の重み。 */
  float weight(int sample) {
    return weights[Objects.checkIndex(sample, size)];
  }
}
