package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.classic.model.SampleDecisions;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.testng.annotations.Test;

/**
 * 1 つのルールベースの打ち手を、2 つのスレッドから同時に呼ぶ。どちらのスレッドも、1 つのスレッドで選んだときと同じ手を返す。判断の記録は
 * 同じ {@code long} の列で、スレッドごとに自分の {@code Observation} で読む。ウォームアップの後は、手を選んでもメモリを確保しない。
 * 2 つのスレッドは、サンプルを互いに逆の順に回る。
 */
public class RuleBotThreadsTest {

  private static final Variant VARIANT = SampleDecisions.VARIANT;

  /** サンプルに集める判断の、種類ごとの最大数。種類はロン・ツモ・九種九牌・鳴き・槓・立直・打牌だけの 7 つ。 */
  private static final int PER_KIND = 64;

  /** サンプルを集めるときに進めるステップ数の上限。 */
  private static final int STEPS = 400_000;

  /** サンプルを回る回数。1 回目でウォームアップし、2 回目からの確保を数える。 */
  private static final int PASSES = 4;

  /**
   * メモリ確保の計測そのものの誤差を見込んだ余裕（バイト）。確保を数える間に手を選ぶ回数は 1000 回を超えるので、1 回選ぶごとに
   * 16 バイト以上を 1 回確保するだけで 16 KB になり、余裕を大きく超える。
   */
  private static final long SLACK_BYTES = 4_096;

  private static final long WAIT_SECONDS = 300;

  @Test(timeOut = 600_000)
  public void twoThreadsChooseWhatOneThreadChooses() throws Exception {
    SampleDecisions samples = new SampleDecisions(29, PER_KIND, STEPS);
    assertTrue(samples.size() > 250, samples.counts().toString());
    int[] expected = new int[samples.size()];
    RuleBot reference = new RuleBot(VARIANT);
    Observation o = new Observation(VARIANT);
    for (int row = 0; row < samples.size(); row++) {
      expected[row] = reference.act(samples.bind(o, row));
      assertTrue(expected[row] >= 0 && expected[row] < o.legalActions(), "row " + row);
    }
    RuleBot shared = new RuleBot(VARIANT);
    CyclicBarrier start = new CyclicBarrier(2);
    ExecutorService threads = Executors.newFixedThreadPool(2);
    try {
      List<Future<Pass>> passes =
          List.of(
              threads.submit(() -> play(shared, samples, start, false)),
              threads.submit(() -> play(shared, samples, start, true)));
      for (Future<Pass> future : passes) {
        Pass pass = future.get(WAIT_SECONDS, TimeUnit.SECONDS);
        assertEquals(pass.chosen(), expected);
        assertEquals(pass.changes(), 0, "moves that changed between passes");
        if (pass.allocated() >= 0) {
          assertTrue(pass.allocated() < SLACK_BYTES, pass.allocated() + " bytes over " + pass.measured() + " acts");
        }
      }
    } finally {
      threads.shutdownNow();
    }
  }

  /**
   * 1 つのスレッドの結果。
   *
   * @param chosen 1 回目に選んだ手（サンプルの番号の順）
   * @param changes 2 回目から、1 回目と違う手を選んだ数
   * @param allocated 2 回目から確保したバイト数（スレッドごとの数を読めなければ −1）
   * @param measured 2 回目から手を選んだ回数
   */
  private record Pass(int[] chosen, int changes, long allocated, int measured) {}

  /** 1 つのスレッドで、サンプルの全ての判断を番号の順か逆の順に決まった回数だけ回り、手を選ぶ。 */
  private static Pass play(RuleBot bot, SampleDecisions samples, CyclicBarrier start, boolean reverse)
      throws Exception {
    ThreadMXBean counter = ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    boolean counted =
        counter != null && counter.isThreadAllocatedMemorySupported() && counter.isThreadAllocatedMemoryEnabled();
    Observation o = new Observation(VARIANT);
    int size = samples.size();
    int[] chosen = new int[size];
    int changes = 0;
    long before = 0;
    start.await(WAIT_SECONDS, TimeUnit.SECONDS);
    for (int pass = 0; pass < PASSES; pass++) {
      if (pass == 1 && counted) {
        before = counter.getCurrentThreadAllocatedBytes();
      }
      for (int step = 0; step < size; step++) {
        int row = reverse ? size - 1 - step : step;
        int move = bot.act(samples.bind(o, row));
        if (pass == 0) {
          chosen[row] = move;
        } else if (move != chosen[row]) {
          changes++;
        }
      }
    }
    long allocated = counted ? counter.getCurrentThreadAllocatedBytes() - before : -1;
    return new Pass(chosen, changes, allocated, (PASSES - 1) * size);
  }
}
