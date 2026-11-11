package com.epsilon.classic.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 自己対局の 1 半荘の軌跡。学習する席の判断ごとに、判断の記録、選んだ手、振る舞いの方策でその手を選ぶ確率、選び替えで結果が
 * 変わりえたかを持つ。ほかに、局ごとに局の始まりの GRP の周辺確率を持ち、終局の点も持つ。判断は半荘の中で起きた順に並ぶ。
 *
 * <p>局末の価値（{@link #endValue}）は、半荘が続く局末なら Φ（次の局の始まりの GRP の周辺確率で順位の効用を平均した値）、半荘が
 * 終わる局末なら終局の順位の効用。GRP の周辺確率を持たない半荘では、どの局末も終局の順位の効用にする。局末の Φ は、次の局で価値の
 * 入力に渡す事前分布と同じ周辺確率から作る。そのため局の中の価値の目標は、次の局の始まりの価値と同じ基準につながる。
 */
public final class GameTrajectory {

  private final Variant variant;
  private final long id;
  private final ObservationBatch observations;
  private final byte[] chosen;
  private final float[] behaviour;
  private final boolean[] causal;
  private final byte[] seat;
  private final short[] hand;

  /** [局] → 局の始まりの周辺確率（絶対席 × 順位）。GRP を使わない半荘は null。 */
  private final float[][] priors;

  private final int hands;
  private final int[] finalScores;

  private GameTrajectory(Builder builder, int[] finalScores) {
    variant = builder.variant;
    id = builder.id;
    observations = builder.observations;
    int size = builder.size;
    chosen = Arrays.copyOf(builder.chosen, size);
    behaviour = Arrays.copyOf(builder.behaviour, size);
    causal = Arrays.copyOf(builder.causal, size);
    seat = Arrays.copyOf(builder.seat, size);
    hand = Arrays.copyOf(builder.hand, size);
    hands = builder.priors.size();
    priors = builder.withPriors ? builder.priors.toArray(float[][]::new) : null;
    this.finalScores = finalScores.clone();
  }

  public Variant variant() {
    return variant;
  }

  /** 半荘 ID。 */
  public long id() {
    return id;
  }

  /** 判断の数。 */
  public int size() {
    return chosen.length;
  }

  /** 局の数。 */
  public int hands() {
    return hands;
  }

  /** リーダーを、この判断の記録を読むように付け替える。 */
  public Observation bind(int decision, Observation reader) {
    return reader.bind(observations, decision);
  }

  /** 選んだ手（合法手の何番目か）。 */
  public int chosen(int decision) {
    return Byte.toUnsignedInt(chosen[decision]);
  }

  /** 振る舞いの方策で、選んだ手を選ぶ確率。 */
  public float behaviour(int decision) {
    return behaviour[decision];
  }

  /** 他家の選びを変えずにこの判断の手だけを選び替えると、局の進みが変わりえたなら true。 */
  public boolean causal(int decision) {
    return causal[decision];
  }

  /** 判断した席（絶対席）。 */
  public int seat(int decision) {
    return seat[decision];
  }

  /** 判断した局（半荘の何局目か）。 */
  public int hand(int decision) {
    return hand[decision];
  }

  /** GRP の周辺確率を持つなら true。 */
  public boolean hasPriors() {
    return priors != null;
  }

  /**
   * 局の始まりの GRP の周辺確率。
   *
   * @return 絶対席 × 順位を席の順に並べた値（GRP の周辺確率を持たない半荘は null）
   */
  public float[] prior(int hand) {
    Objects.checkIndex(hand, hands);
    return priors == null ? null : priors[hand];
  }

  /** 終局の順位（0 始まり。同点は起家に近い席が上）。 */
  public int finalRank(int seat) {
    return Placement.rank(finalScores, Objects.checkIndex(seat, variant.seats()));
  }

  /** 終局の点（絶対席の順）。 */
  public int[] finalScores() {
    return finalScores.clone();
  }

  /**
   * 局末の席の価値（効用の単位）。
   *
   * @param hand 局（半荘の何局目か）
   * @param seat 絶対席
   * @param utility 順位の効用
   */
  public double endValue(int hand, int seat, RankUtility utility) {
    Objects.checkIndex(hand, hands);
    int seats = variant.seats();
    Objects.checkIndex(seat, seats);
    if (priors == null || hand + 1 == hands) {
      return utility.value(variant, finalRank(seat));
    }
    float[] next = priors[hand + 1];
    double phi = 0;
    for (int rank = 0; rank < seats; rank++) {
      phi += next[seat * seats + rank] * utility.value(variant, rank);
    }
    return phi;
  }

  /**
   * 1 半荘の軌跡を書き足していく。卓を進める処理が持ち、局の始まりごとに {@link #hand}、学習する席の判断ごとに {@link #decision}、
   * 半荘の終わりに {@link #finish} を呼ぶ。同時に使えるのは 1 つのスレッドだけ。
   */
  public static final class Builder {

    private static final int INITIAL_DECISIONS = 256;
    private static final int LONGS_PER_DECISION = 32;

    private final Variant variant;
    private final long id;
    private final ObservationBatch observations;
    private final List<float[]> priors = new ArrayList<>();
    private byte[] chosen = new byte[INITIAL_DECISIONS];
    private float[] behaviour = new float[INITIAL_DECISIONS];
    private boolean[] causal = new boolean[INITIAL_DECISIONS];
    private byte[] seat = new byte[INITIAL_DECISIONS];
    private short[] hand = new short[INITIAL_DECISIONS];
    private int size;
    private boolean withPriors;
    private boolean finished;

    /**
     * @param id 半荘 ID
     */
    public Builder(Variant variant, long id) {
      this.variant = Objects.requireNonNull(variant, "variant");
      this.id = id;
      observations = new ObservationBatch(variant, INITIAL_DECISIONS * LONGS_PER_DECISION);
    }

    /**
     * 局を始める。半荘の最初の判断の前と、局末から次の局を配った後に呼ぶ。
     *
     * @param prior 局の始まりの GRP の周辺確率（絶対席 × 順位を席の順に並べる。席ごとの和は 1）。GRP を使わない半荘は全ての局で null
     * @throws IllegalArgumentException 周辺確率の長さか値が正しくないか、局によって周辺確率の有無が違うとき
     * @throws IllegalStateException 半荘を終えた後か、局の数が多すぎるとき
     */
    public void hand(float[] prior) {
      requireOpen();
      if (!priors.isEmpty() && (prior != null) != withPriors) {
        throw new IllegalArgumentException("Every hand of game " + id + " must have a GRP prior or none");
      }
      if (prior != null) {
        int seats = variant.seats();
        if (prior.length != seats * seats) {
          throw new IllegalArgumentException("A GRP prior holds " + seats * seats + " values, not " + prior.length);
        }
        for (float probability : prior) {
          if (!(probability >= 0) || !Float.isFinite(probability)) {
            throw new IllegalArgumentException("GRP prior values must be finite and non-negative: " + probability);
          }
        }
      }
      if (priors.size() > Short.MAX_VALUE) {
        throw new IllegalStateException("Game " + id + " has too many hands");
      }
      withPriors = prior != null;
      priors.add(prior == null ? null : prior.clone());
    }

    /**
     * 学習する席の判断を足す。判断の記録は写すので、呼んだ後にリーダーを付け替えてよい。
     *
     * @param observation 判断の記録を読んでいるリーダー
     * @param choice 選んだ手（合法手の何番目か）
     * @param probability 振る舞いの方策で、選んだ手を選ぶ確率（0 より大きく 1 以下）
     * @param effective 他家の選びを変えずにこの手だけを選び替えると、局の進みが変わりえたなら true
     * @throws IllegalArgumentException ルールが違うか、確率が範囲の外のとき
     * @throws IndexOutOfBoundsException 選んだ手が合法手の範囲の外のとき
     * @throws IllegalStateException 局を始めていないか、半荘を終えた後のとき
     */
    public void decision(Observation observation, int choice, float probability, boolean effective) {
      requireOpen();
      if (priors.isEmpty()) {
        throw new IllegalStateException("Start a hand of game " + id + " before its decisions");
      }
      Objects.checkIndex(choice, observation.legalActions());
      if (!(probability > 0 && probability <= 1)) {
        throw new IllegalArgumentException("The behaviour probability must be in (0, 1]: " + probability);
      }
      observations.add(observation);
      if (size == chosen.length) {
        int capacity = 2 * size;
        chosen = Arrays.copyOf(chosen, capacity);
        behaviour = Arrays.copyOf(behaviour, capacity);
        causal = Arrays.copyOf(causal, capacity);
        seat = Arrays.copyOf(seat, capacity);
        hand = Arrays.copyOf(hand, capacity);
      }
      chosen[size] = (byte) choice;
      behaviour[size] = probability;
      causal[size] = effective;
      seat[size] = (byte) observation.observer();
      hand[size] = (short) (priors.size() - 1);
      size++;
    }

    /**
     * 半荘を終えて軌跡を作る。この後は書き足せない。
     *
     * @param finalScores 終局の点（絶対席の順）
     * @throws IllegalArgumentException 終局の点の数が席の数と違うとき
     * @throws IllegalStateException 局が 1 つも無いか、半荘を終えた後のとき
     */
    public GameTrajectory finish(int[] finalScores) {
      requireOpen();
      if (priors.isEmpty()) {
        throw new IllegalStateException("Game " + id + " has no hand");
      }
      if (finalScores.length != variant.seats()) {
        throw new IllegalArgumentException(
            "Final scores of " + variant.seats() + " seats are needed, not " + finalScores.length);
      }
      finished = true;
      return new GameTrajectory(this, finalScores);
    }

    private void requireOpen() {
      if (finished) {
        throw new IllegalStateException("Game " + id + " is finished");
      }
    }
  }
}
