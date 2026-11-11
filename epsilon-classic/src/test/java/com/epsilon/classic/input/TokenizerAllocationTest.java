package com.epsilon.classic.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomPlay;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * ウォームアップの後は、判断する全ての席の入力を Nano と Major の両方で書いても、メモリを確保しない。判断の記録を作る処理と、
 * 自己対局を進めるエンジンもメモリを確保しない。
 */
public class TokenizerAllocationTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 50_000;

  /**
   * 確保したバイト数の数え方そのものの誤差を見込んだ余裕（バイト）。測るのは 5 × 10^4 ステップなので、1 ステップで 16 バイト以上を
   * 1 回確保するだけで 800 KB になる。
   */
  private static final long SLACK_BYTES = 4_096;

  private final RandomPlay play = new RandomPlay(VARIANT, 29);
  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation o = new Observation(VARIANT);
  private final ClassicExtras extras = ClassicExtras.empty();
  private final float[] prior = new float[ClassicSchema.SEATS * ClassicSchema.SEATS];

  /** どの判断も収まる区分（1 席の合法手は 32 まで、鳴いた後の打牌は 11 まで）。 */
  private final Bucket largest = Buckets.required(32, 11);
  private final ClassicTokenizer nano = new ClassicTokenizer(Architecture.NANO, VARIANT);
  private final ClassicTokenizer major = new ClassicTokenizer(Architecture.MAJOR, VARIANT);
  private final ClassicInputs nanoInputs = new ClassicInputs(ClassicSchema.of(Architecture.NANO), largest, 1);
  private final ClassicInputs majorInputs = new ClassicInputs(ClassicSchema.of(Architecture.MAJOR), largest, 1);

  @Test(timeOut = 900_000)
  public void tokenizingAllocatesNothingOnceWarm() {
    ThreadMXBean threads = ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null || !threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    long checksum = run(WARM_UP);
    long before = threads.getCurrentThreadAllocatedBytes();
    checksum += run(MEASURED);
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    Assert.assertTrue(
        allocated < SLACK_BYTES, allocated + " bytes over " + MEASURED + " steps (checksum " + checksum + ")");
  }

  /** 指定したステップ数だけ進め、判断の時点ごとに判断する全ての席の入力を書く。最適化で計算が消されないように、書いた値の一部の和を返す。 */
  private long run(int steps) {
    long checksum = 0;
    for (int step = 0; step < steps; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        batch.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          o.bind(batch, writer.capture(play.engine.view(decisions.seat(slot)), decisions, slot, batch));
          nano.write(o, extras, nanoInputs, 0);
          major.write(o, extras, prior, majorInputs, 0);
          checksum += nanoInputs.categorical()[nanoInputs.start(Tensor.ACTION_CATEGORIES, 0)];
          checksum += majorInputs.categorical()[majorInputs.start(Tensor.TRANSITION_TILES, 0)];
        }
      }
      play.step();
    }
    return checksum;
  }
}
