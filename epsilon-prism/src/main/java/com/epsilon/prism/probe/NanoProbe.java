package com.epsilon.prism.probe;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.nano.ai.decision.runtime.NanoReviewPolicyProvider;
import com.epsilon.prism.collect.CandidateProbe;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.spi.ReviewPolicy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.LongAdder;

/**
 * Epsilon-Nano の貪欲の手を返す測定器（{@link CandidateProbe}。probe-nano）。Nano の方策の確率（{@link NanoReviewPolicyProvider} の
 * {@link ReviewPolicy}）の最大の手で、同点は番号の小さい手。Nano は測るだけで、打つのは常に PRISM。
 *
 * <p>スレッド: {@link ReviewPolicy} はスレッド安全と仮定しない。空いている ReviewPolicy を 1 つ借りて評価し、返す（空きがなければ新しく開く）。
 * 同時に開くのは同時に問い合わせた進行ワーカーの数まで（advanceWorkers 以下）で、借りた ReviewPolicy は 1 スレッドだけが使う。スレッドに縛らない
 * ので、ワーカーのスレッドが入れ替わっても数は増えない。全て {@link #close} で閉じる。
 */
public final class NanoProbe implements CandidateProbe, AutoCloseable {
  private final NanoReviewPolicyProvider provider = new NanoReviewPolicyProvider();
  private final Path checkpoint;
  private final Map<String, String> options;
  private final ConcurrentLinkedQueue<ReviewPolicy> idle = new ConcurrentLinkedQueue<>();
  private final List<ReviewPolicy> opened = new ArrayList<>();
  private final LongAdder[][] counts = new LongAdder[Outcome.values().length][DecisionClass.COUNT];
  private boolean closed;

  /**
   * Nano のチェックポイントを開く（1 つ目はここで開いて、読めないチェックポイントを収集の前に落とす）。
   *
   * @param checkpoint Nano の Decision チェックポイントのディレクトリ
   * @param device {@link NanoReviewPolicyProvider} のデバイス（cpu か gpu:N）
   */
  public NanoProbe(Path checkpoint, String device) {
    this.checkpoint = checkpoint;
    options = Map.of("device", device);
    for (LongAdder[] row : counts)
      for (int c = 0; c < row.length; c++) row[c] = new LongAdder();
    idle.add(open());
  }

  @Override
  public int choose(PublicObservation observation, List<Action> legal, int played) {
    ReviewPolicy policy = idle.poll();
    if (policy == null) policy = open();
    try {
      float[] probabilities = policy.evaluate(observation, legal).probabilities();
      if (probabilities.length != legal.size())
        throw new IllegalStateException(
            "Nano returned " + probabilities.length + " probabilities for " + legal.size());
      return argmax(probabilities);
    } finally {
      idle.add(policy);
    }
  }

  @Override
  public void counted(DecisionClass decisionClass, Outcome outcome) {
    counts[outcome.ordinal()][decisionClass.ordinal()].increment();
  }

  /** 類ごとの outcome の数（{@link DecisionClass} の番号の順）。 */
  public long[] counts(Outcome outcome) {
    long[] result = new long[DecisionClass.COUNT];
    for (int c = 0; c < result.length; c++) result[c] = counts[outcome.ordinal()][c].sum();
    return result;
  }

  /** 開いた ReviewPolicy の数（同時に問い合わせたスレッドの最大）。 */
  public synchronized int opened() {
    return opened.size();
  }

  /** 最大の確率の手（同点は番号の小さい手）。 */
  static int argmax(float[] probabilities) {
    int best = 0;
    for (int a = 1; a < probabilities.length; a++)
      if (probabilities[a] > probabilities[best]) best = a;
    return best;
  }

  /** 新しい ReviewPolicy を開く（読み込みは 1 つずつ）。 */
  private synchronized ReviewPolicy open() {
    if (closed) throw new IllegalStateException("Nano probe is closed");
    ReviewPolicy policy = provider.openReview(checkpoint, options);
    opened.add(policy);
    return policy;
  }

  @Override
  public synchronized void close() {
    closed = true;
    RuntimeException failure = null;
    for (ReviewPolicy policy : opened) {
      try {
        policy.close();
      } catch (RuntimeException e) {
        if (failure == null) failure = e;
        else failure.addSuppressed(e);
      }
    }
    opened.clear();
    idle.clear();
    if (failure != null) throw failure;
  }
}
