package com.epsilon.classic.campaign;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.classic.ClassicPolicy;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * 学習器の席。方策の合法手の確率 π に探索を混ぜた振る舞いの方策 μ（{@link Exploration}）を打つ分布として書き、卓はこの分布から手を引く。
 * CPU の方策は進行のスレッドでその場で答え、GPU の方策には受付の区分ごとにまとめたバッチで答えさせる。
 *
 * <p>迷いの百分位を区間の中で散らす一様値は、キー・牌山の番号・判断の記録 O の中身だけで決める（{@link #tie}）。同じ O の π が同じなら、
 * 呼ぶ順や同時の呼び出しに依らず μ も同じになる。
 */
final class LearnerSeat implements TableSeat<Query> {

  /** 送り出し用のバッチを確保するときに見込む、1 行あたりの {@code long} の数。 */
  private static final int LONGS_PER_ROW = 64;

  private static final double UNIT = 0x1.0p-53;

  private final ClassicPolicy policy;
  private final Exploration exploration;
  private final long key;
  private final ThreadLocal<Observation> readers;

  /** その場で答えるときに π を書くバッファ（スレッドごと）。 */
  private final ThreadLocal<float[]> policies = ThreadLocal.withInitial(() -> new float[ClassicSchema.MAX_LEGAL]);

  /**
   * @param policy 学習器の方策（所有権を受け取る）
   * @param exploration μ を作る探索。迷いはこの席が答えた判断の分だけ数える
   * @param key 迷いの百分位を散らす一様値のキー
   */
  LearnerSeat(ClassicPolicy policy, Exploration exploration, long key) {
    this.policy = Objects.requireNonNull(policy, "policy");
    this.exploration = Objects.requireNonNull(exploration, "exploration");
    this.key = key;
    readers = ThreadLocal.withInitial(() -> new Observation(policy.variant()));
  }

  @Override
  public boolean inline() {
    return policy instanceof ClassicPolicy.Inline;
  }

  @Override
  public void evaluateInline(Query query) {
    Observation observation = query.bind(readers.get());
    float[] probabilities = policies.get();
    policy.probabilities(observation, probabilities);
    double tie = tie(key, query.wall(), query.recordWords(), query.recordBase(), observation.longs());
    query.sigma = behaviour(observation, probabilities, tie);
  }

  @Override
  public int admission(Observation observation) {
    return policy instanceof ClassicPolicy.Batched batched ? batched.admission(observation) : 0;
  }

  @Override
  public int maxBatch(int admission) {
    return batched().maxBatch(admission);
  }

  @Override
  public boolean needsWork() {
    return !(policy instanceof ClassicPolicy.Batched batched) || batched.needsWork();
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireIngress(int admission) {
    return batched().tryAcquire(admission);
  }

  /** 問い合わせの O を送り出し用のバッチに写して π を推論させ、π が出たスレッドで μ を書く。 */
  @Override
  public CompletableFuture<Void> submit(List<Query> batch, DecisionInferenceIngress ingress) {
    ClassicPolicy.Batched batched = batched();
    int rows = batch.size();
    float[][] probabilities = new float[rows][];
    double[] ties = new double[rows];
    ObservationBatch out;
    try {
      Observation reader = readers.get();
      out = new ObservationBatch(policy.variant(), Math.max(1, rows * LONGS_PER_ROW));
      for (int row = 0; row < rows; row++) {
        Query query = batch.get(row);
        Observation observation = query.bind(reader);
        ties[row] = tie(key, query.wall(), query.recordWords(), query.recordBase(), observation.longs());
        probabilities[row] = new float[observation.legalActions()];
        out.add(observation);
      }
    } catch (RuntimeException | Error failure) {
      ingress.abort();
      throw failure;
    }
    return batched
        .probabilities(out, probabilities, ingress)
        .thenRun(
            () -> {
              Observation reader = readers.get();
              for (int row = 0; row < rows; row++) {
                batch.get(row).sigma = behaviour(reader.bind(out, row), probabilities[row], ties[row]);
              }
            });
  }

  @Override
  public void close() {
    policy.close();
  }

  /** π から μ を作る（合法手の数の長さの新しい配列）。 */
  private float[] behaviour(Observation observation, float[] probabilities, double tie) {
    float[] behaviour = new float[observation.legalActions()];
    exploration.behaviour(observation, probabilities, tie, behaviour);
    return behaviour;
  }

  private ClassicPolicy.Batched batched() {
    if (!(policy instanceof ClassicPolicy.Batched batched)) {
      throw new UnsupportedOperationException(policy.label() + " answers inline");
    }
    return batched;
  }

  /**
   * 迷いの百分位を区間の中で散らす一様値（0 以上 1 未満）。
   *
   * <pre>
   * h ← mix64(key ^ mix64(wall))
   * h ← mix64(h ^ O[i])        （O の {@code long} を先頭から順に）
   * 一様値 = (h >>> 11)·2^−53
   * </pre>
   *
   * @param key キー
   * @param wall 半荘の牌山の番号
   * @param words O を置いた配列
   * @param base 配列の中の O の先頭
   * @param longs O の {@code long} の数
   */
  static double tie(long key, long wall, long[] words, int base, int longs) {
    long hash = SeedMixer.mix64(key ^ SeedMixer.mix64(wall));
    for (int index = base; index < base + longs; index++) {
      hash = SeedMixer.mix64(hash ^ words[index]);
    }
    return (hash >>> 11) * UNIT;
  }
}
