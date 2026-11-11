package com.epsilon.classic;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.classic.model.SampleDecisions;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 乱数で初期化した小さなチェックポイントを系列から開き、ランダムな自己対局から取った判断（ロン・ツモ・九種九牌・鳴き・槓・立直を含む）で
 * 動かす。
 *
 * <p>CPU では次のことを確かめる。その場で答える方策は、評価で確率がいちばん高い手を選ぶ。その場で答える方策を複数のスレッドから同時に
 * 呼んでも、答えは変わらない。CPU で作ったまとめて推論する方策が、受付の区分ごとのバッチで選ぶ手は、その場で答える方策の手と同じ。
 * まとめて推論する方策は、自分の予約だけを受け取る。
 *
 * <p>GPU があれば（rocm のグループ）、BF16 でまとめて推論する方策がどの判断でも合法手を選ぶことを確かめ、CPU FP32 の手と一致した
 * 割合をログに書く。容量を呼び出し元の実行環境の受付で予約することも確かめる。
 */
public class ClassicPolicyTest {

  private static final Logger LOG = LoggerFactory.getLogger(ClassicPolicyTest.class);

  private static final Variant VARIANT = SampleDecisions.VARIANT;
  private static final PolicyOptions CPU = PolicyOptions.defaults();
  private static final PolicyOptions GPU = new PolicyOptions("gpu0", Precision.BFLOAT16, 8, Kernels.EAGER);
  private static final int PER_KIND = 4;
  private static final int STEPS = 400_000;
  private static final long GPU_WAIT_SECONDS = 60;

  /** 同じ方策を同時に呼ぶスレッドの数と、各スレッドがサンプルを巡る回数。 */
  private static final int CALLERS = 4, CALLER_PASSES = 4;

  private final Map<Architecture, Path> checkpoints = new EnumMap<>(Architecture.class);
  private Path root;
  private SampleDecisions samples;

  /**
   * テスト用のチェックポイントを書く。native と rocm のどちらか一方のグループだけを回しても動くよう、グループを付けずにいつも走らせる
   * （グループを付けると、外したグループの準備として飛ばされる）。
   */
  @BeforeClass(alwaysRun = true)
  public void writeCheckpoints() throws IOException {
    root = Files.createTempDirectory("classic-policy");
    samples = new SampleDecisions(43, PER_KIND, STEPS);
    LOG.info("Sampled {} decisions: {}", samples.size(), samples.counts());
    for (Architecture architecture : Architecture.values()) {
      Path directory = root.resolve(architecture.name().toLowerCase(Locale.ROOT) + "-random");
      RandomCheckpoints.write(directory, architecture, VARIANT, 19 + architecture.ordinal());
      checkpoints.put(architecture, directory);
    }
  }

  @AfterClass(alwaysRun = true)
  public void deleteCheckpoints() throws IOException {
    if (root == null) {
      return;
    }
    try (Stream<Path> files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  @Test(groups = "native")
  public void theInlinePolicyActsAsTheGreedyMoveOfItsEvaluation() throws IOException {
    Observation o = new Observation(VARIANT);
    for (Architecture architecture : Architecture.values()) {
      Path checkpoint = checkpoints.get(architecture);
      try (Policy policy = ModelFamily.of(architecture.series()).open(checkpoint, VARIANT, CPU)) {
        assertTrue(policy instanceof ClassicPolicy.Inline, policy.getClass().getName());
        assertEquals(policy.label(), architecture.series().prefix() + ":" + checkpoint.getFileName());
        assertEquals(policy.variant(), VARIANT);
        InlinePolicy inline = (InlinePolicy) policy;
        for (int row = 0; row < samples.size(); row++) {
          samples.bind(o, row);
          Evaluation evaluation = policy.evaluate(o);
          assertEquals(evaluation.probabilities().length, o.legalActions());
          assertNull(evaluation.values());
          assertTrue(evaluation.probabilities()[evaluation.greedy()] > 0, architecture + " row " + row);
          assertEquals(inline.act(o), evaluation.greedy(), architecture + " row " + row);
        }
      }
    }
    try (NetworkRuntime runtime = new NetworkRuntime(1, 1, 1);
        Policy policy = ModelFamily.of(Series.NANO).open(checkpoints.get(Architecture.NANO), VARIANT, CPU, runtime)) {
      assertTrue(policy instanceof ClassicPolicy.Inline, policy.getClass().getName());
    }
  }

  /** その場で答える方策を複数のスレッドから同時に呼んでも、1 つのスレッドで呼んだときと同じ確率と手を返す。 */
  @Test(groups = "native", timeOut = 600_000)
  public void theInlinePolicyAnswersConcurrentCallersAsOneThreadDoes() throws Exception {
    ExecutorService callers = Executors.newFixedThreadPool(CALLERS);
    try {
      for (Architecture architecture : Architecture.values()) {
        try (Policy opened = ModelFamily.of(architecture.series()).open(checkpoints.get(architecture), VARIANT, CPU)) {
          ClassicPolicy policy = (ClassicPolicy) opened;
          Observation o = new Observation(VARIANT);
          float[] row = new float[ClassicSchema.MAX_LEGAL];
          int[] greedy = new int[samples.size()];
          float[][] expected = new float[samples.size()][];
          for (int index = 0; index < samples.size(); index++) {
            samples.bind(o, index);
            greedy[index] = policy.probabilities(o, row);
            expected[index] = Arrays.copyOf(row, o.legalActions());
          }
          List<Future<List<String>>> answers = new ArrayList<>();
          for (int caller = 0; caller < CALLERS; caller++) {
            int first = caller;
            answers.add(callers.submit(() -> mismatches(policy, first, greedy, expected)));
          }
          List<String> failures = new ArrayList<>();
          for (Future<List<String>> answer : answers) {
            try {
              failures.addAll(answer.get());
            } catch (ExecutionException failure) {
              failures.add(String.valueOf(failure.getCause()));
            }
          }
          assertEquals(failures, List.of(), architecture.toString());
        }
      }
    } finally {
      callers.shutdownNow();
    }
  }

  @Test(groups = "native")
  public void aBatchedPolicyOnTheCpuChoosesTheInlineMoves() throws IOException {
    for (Architecture architecture : Architecture.values()) {
      Path checkpoint = checkpoints.get(architecture);
      try (Policy reference = ModelFamily.of(architecture.series()).open(checkpoint, VARIANT, CPU);
          ClassicPolicy.Batched batched = cpuBatched(checkpoint, reference.label())) {
        Agreement agreement = compare(batched, (InlinePolicy) reference);
        assertEquals(agreement.mismatches(), List.of(), architecture.toString());
        assertEquals(agreement.total(), samples.size());
      }
    }
  }

  /**
   * パラメーターを CPU に置いた、まとめて推論する方策は、容量を数えない予約を返し、いつも次のバッチを求める。{@code choose} は自分の
   * 予約だけを 1 回だけ受け取り、受け付けないバッチでは予約を返してから例外を投げる。
   */
  @Test(groups = "native")
  public void aBatchedPolicyOnTheCpuTakesItsOwnReservation() throws IOException {
    Path checkpoint = checkpoints.get(Architecture.NANO);
    try (ClassicPolicy.Batched policy = cpuBatched(checkpoint, "nano:first");
        ClassicPolicy.Batched other = cpuBatched(checkpoint, "nano:second")) {
      ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
      Observation o = samples.bind(new Observation(VARIANT), 0);
      batch.add(o);
      assertTrue(policy.needsWork());
      DecisionInferenceIngress.Attempt attempt = policy.tryAcquire(policy.admission(o));
      assertTrue(attempt.acquired());
      int[] chosen = {-1};
      policy.choose(batch, chosen, attempt.ingress()).join();
      assertTrue(chosen[0] >= 0 && chosen[0] < o.legalActions(), "choice " + chosen[0]);
      expectThrows(IllegalStateException.class, () -> policy.choose(batch, chosen, attempt.ingress()));
      DecisionInferenceIngress foreign = other.tryAcquire(0).ingress();
      expectThrows(IllegalArgumentException.class, () -> policy.choose(batch, chosen, foreign));
      AtomicInteger released = new AtomicInteger();
      DecisionInferenceIngress counted = new DecisionInferenceIngress(policy, null, released::incrementAndGet);
      ObservationBatch empty = new ObservationBatch(VARIANT, 16);
      expectThrows(IllegalArgumentException.class, () -> policy.choose(empty, chosen, counted));
      assertEquals(released.get(), 1);
    }
  }

  /**
   * 合法手の確率の書き出し: その場で推論して書く確率を 0〜1 に抑えて和でそろえると評価の確率になり、返す手は評価の貪欲の手。CPU で作った
   * まとめて推論する方策が受付の区分ごとのバッチで書く確率は、その場で書く確率と同じ。行の長さが合法手の数と違うバッチは、予約を返してから
   * 拒む。
   */
  @Test(groups = "native")
  public void theProbabilitiesAgreeAcrossTheInlineAndBatchedPaths() throws IOException {
    Path checkpoint = checkpoints.get(Architecture.NANO);
    try (Policy opened = ModelFamily.of(Series.NANO).open(checkpoint, VARIANT, CPU);
        ClassicPolicy.Batched batched = cpuBatched(checkpoint, opened.label())) {
      ClassicPolicy inline = (ClassicPolicy) opened;
      Observation o = new Observation(VARIANT);
      float[] row = new float[ClassicSchema.MAX_LEGAL];
      Map<Integer, ObservationBatch> batches = new TreeMap<>();
      Map<Integer, List<float[]>> expected = new TreeMap<>();
      for (int index = 0; index < samples.size(); index++) {
        samples.bind(o, index);
        int legal = o.legalActions();
        Evaluation evaluation = inline.evaluate(o);
        assertEquals(inline.probabilities(o, row), evaluation.greedy(), "row " + index);
        float[] probabilities = Arrays.copyOf(row, legal);
        double sum = 0;
        for (float probability : probabilities) {
          sum += Math.min(1f, Math.max(0f, probability));
        }
        for (int action = 0; action < legal; action++) {
          float clamped = Math.min(1f, Math.max(0f, probabilities[action]));
          assertEquals(evaluation.probabilities()[action], (float) (clamped / sum), 1e-6f, "row " + index);
        }
        int admission = batched.admission(o);
        batches.computeIfAbsent(admission, ignored -> new ObservationBatch(VARIANT, 1_024)).add(o);
        expected.computeIfAbsent(admission, ignored -> new ArrayList<>()).add(probabilities);
      }
      for (Map.Entry<Integer, ObservationBatch> entry : batches.entrySet()) {
        ObservationBatch batch = entry.getValue();
        List<float[]> rows = expected.get(entry.getKey());
        float[][] written = new float[batch.size()][];
        for (int index = 0; index < written.length; index++) {
          written[index] = new float[rows.get(index).length];
        }
        batched.probabilities(batch, written, reserve(batched, entry.getKey())).join();
        for (int index = 0; index < written.length; index++) {
          for (int action = 0; action < written[index].length; action++) {
            assertEquals(written[index][action], rows.get(index)[action], 1e-5f, "admission " + entry.getKey());
          }
        }
      }
      ObservationBatch one = new ObservationBatch(VARIANT, 1_024);
      one.add(samples.bind(o, 0));
      AtomicInteger released = new AtomicInteger();
      DecisionInferenceIngress counted = new DecisionInferenceIngress(batched, null, released::incrementAndGet);
      float[][] wrong = {new float[o.legalActions() + 1]};
      expectThrows(IllegalArgumentException.class, () -> batched.probabilities(one, wrong, counted));
      assertEquals(released.get(), 1);
    }
  }

  @Test(groups = "rocm")
  public void theGpuPolicyChoosesLegalMovesInBfloat16() throws IOException {
    requireGpu();
    for (Architecture architecture : Architecture.values()) {
      Path checkpoint = checkpoints.get(architecture);
      try (Policy reference = ModelFamily.of(architecture.series()).open(checkpoint, VARIANT, CPU);
          Policy policy = ModelFamily.of(architecture.series()).open(checkpoint, VARIANT, GPU)) {
        assertTrue(policy instanceof BatchedPolicy, policy.getClass().getName());
        Agreement agreement = compare((BatchedPolicy) policy, (InlinePolicy) reference);
        assertEquals(agreement.total(), samples.size());
        LOG.info(
            "{} GPU BF16 greedy moves agree with CPU FP32 on {} of {} decisions",
            architecture,
            agreement.total() - agreement.mismatches().size(),
            agreement.total());
      }
    }
  }

  /**
   * GPU でまとめて推論する方策は、呼び出し元の実行環境の受付でホストの容量を予約する。容量（実行を待てるバッチの数）までは予約でき、
   * それを超えると、空いたときに知らせる通知を返す。予約したバッチで GPU のスロットが埋まっている間は、次のバッチを求めない。
   */
  @Test(groups = "rocm")
  public void theGpuPolicyReservesTheCallersCapacity() throws Exception {
    requireGpu();
    Path checkpoint = checkpoints.get(Architecture.NANO);
    try (NetworkRuntime runtime = new NetworkRuntime(2, 1, 2)) {
      try (Policy opened = ModelFamily.of(Series.NANO).open(checkpoint, VARIANT, GPU, runtime)) {
        BatchedPolicy policy = (BatchedPolicy) opened;
        assertTrue(policy.needsWork());
        DecisionInferenceIngress.Attempt first = policy.tryAcquire(0);
        DecisionInferenceIngress.Attempt second = policy.tryAcquire(0);
        assertTrue(first.acquired() && second.acquired());
        assertFalse(policy.needsWork());
        DecisionInferenceIngress.Attempt blocked = policy.tryAcquire(0);
        assertFalse(blocked.acquired());
        ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
        Observation o = samples.bind(new Observation(VARIANT), 0);
        batch.add(o);
        int[] chosen = {-1};
        try {
          policy.choose(batch, chosen, first.ingress()).get(GPU_WAIT_SECONDS, TimeUnit.SECONDS);
        } finally {
          // バッチが失敗しても 2 つ目の予約を返す（返さないと、方策を閉じる処理が予約の終わりを待ち続ける）
          second.ingress().abort();
        }
        assertTrue(chosen[0] >= 0 && chosen[0] < o.legalActions(), "choice " + chosen[0]);
        blocked.available().get(GPU_WAIT_SECONDS, TimeUnit.SECONDS);
        DecisionInferenceIngress.Attempt again = policy.tryAcquire(0);
        assertTrue(again.acquired());
        again.ingress().abort();
      }
    }
  }

  /**
   * まとめて推論する方策の手と、その場で答える方策の手を比べた結果。
   *
   * @param total 比べた判断の数
   * @param mismatches 手が違った判断（サンプルの番号と両方の手）
   */
  private record Agreement(int total, List<String> mismatches) {}

  /**
   * サンプルの全ての判断を、受付の区分ごとに 1 つのバッチにまとめる。区分ごとに容量を予約して手を選び、どの手も合法手であることを
   * 確かめてから、基準の方策の手と比べる。
   *
   * @param batched 比べる、まとめて推論する方策
   * @param reference 基準の方策（CPU FP32）
   * @return 比べた判断の数と、手が違った判断
   */
  private Agreement compare(BatchedPolicy batched, InlinePolicy reference) {
    Observation o = new Observation(VARIANT);
    Map<Integer, ObservationBatch> batches = new TreeMap<>();
    Map<Integer, List<Integer>> rows = new TreeMap<>();
    for (int row = 0; row < samples.size(); row++) {
      samples.bind(o, row);
      int admission = batched.admission(o);
      batches.computeIfAbsent(admission, ignored -> new ObservationBatch(VARIANT, 1_024)).add(o);
      rows.computeIfAbsent(admission, ignored -> new ArrayList<>()).add(row);
    }
    int total = 0;
    List<String> mismatches = new ArrayList<>();
    for (Map.Entry<Integer, ObservationBatch> entry : batches.entrySet()) {
      ObservationBatch batch = entry.getValue();
      assertTrue(batch.size() <= batched.maxBatch(entry.getKey()), "admission " + entry.getKey());
      int[] chosen = new int[batch.size()];
      batched.choose(batch, chosen, reserve(batched, entry.getKey())).join();
      for (int index = 0; index < batch.size(); index++) {
        o.bind(batch, index);
        assertTrue(chosen[index] >= 0 && chosen[index] < o.legalActions(), "choice " + chosen[index]);
        int expected = reference.act(o);
        if (chosen[index] != expected) {
          int row = rows.get(entry.getKey()).get(index);
          mismatches.add(row + ": " + o.action(chosen[index]) + " vs " + o.action(expected));
        }
        total++;
      }
    }
    return new Agreement(total, mismatches);
  }

  /**
   * サンプルの判断を {@code first} 番目から巡って何周か推論し、1 つのスレッドで求めた答えと違った判断を返す。
   *
   * @param greedy 判断ごとの、1 つのスレッドで求めた手
   * @param expected 判断ごとの、1 つのスレッドで求めた合法手の確率
   */
  private List<String> mismatches(ClassicPolicy policy, int first, int[] greedy, float[][] expected) {
    Observation o = new Observation(VARIANT);
    float[] row = new float[ClassicSchema.MAX_LEGAL];
    List<String> mismatches = new ArrayList<>();
    for (int step = 0; step < CALLER_PASSES * samples.size(); step++) {
      int index = (first + step) % samples.size();
      samples.bind(o, index);
      boolean same = policy.probabilities(o, row) == greedy[index];
      for (int action = 0; action < expected[index].length; action++) {
        same &= Math.abs(row[action] - expected[index][action]) <= 1e-6f;
      }
      if (!same) {
        mismatches.add("row " + index);
      }
    }
    return mismatches;
  }

  /** 区分のバッチの容量を予約する（空きが無ければ、空くのを待って予約し直す）。 */
  private static DecisionInferenceIngress reserve(BatchedPolicy policy, int admission) {
    while (true) {
      DecisionInferenceIngress.Attempt attempt = policy.tryAcquire(admission);
      if (attempt.acquired()) {
        return attempt.ingress();
      }
      attempt.available().join();
    }
  }

  private static void requireGpu() {
    if (Engine.getEngine("PyTorch").getGpuCount() == 0) {
      throw new SkipException("No GPU is visible");
    }
  }

  /** パラメーターを CPU に置いて、まとめて推論する方策を作る。パラメーターのマネージャーは方策が閉じる。 */
  private static ClassicPolicy.Batched cpuBatched(Path checkpoint, String label) throws IOException {
    NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
    try {
      ClassicNetwork network = ClassicCheckpoint.read(checkpoint, VARIANT, manager);
      return new ClassicPolicy.Batched(network, manager, Precision.FLOAT32, label, null);
    } catch (IOException | RuntimeException failure) {
      manager.close();
      throw failure;
    }
  }
}
