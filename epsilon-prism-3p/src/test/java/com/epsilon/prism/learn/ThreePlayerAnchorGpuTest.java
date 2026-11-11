package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.backend.ThreePlayerRows;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism3p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 三人打ちのアンカーを HIP のバックエンドで動かせることを確かめる（rocm のグループ。GPU 1 枚）。三人打ちの入力形式で乱数で初期化した v2 の
 * 深さ 2 と V3 のネットワークを保存し、同じバッチ（エンジンで打った三人打ちの判断と、公開のイベントが多い行）で比べる。HIP のバックエンドを
 * 指定したアンカーは {@link Anchor.Hip} で開き、その s は行ごとの先頭の位置を使うと eager のアンカーと同じ手の順に並ぶ。
 *
 * <p>s の比べ方: GPU の BF16 の 2 つのバックエンド（HIP と eager）の差は丸めのずれで、乱数で初期化した 1 つのネットワークでは 2e-2 を
 * 超えることもある。そこで形ごとに種を変えた {@value #SEEDS} 個のネットワークで、CPU の FP32 の eager のアンカーを真の値とし、HIP の s の
 * ずれの平均が eager の BF16 の s のずれの平均の {@value #ROUNDING} 倍以内であることを確かめる。HIP と eager の BF16 の差の平均は、
 * 三人打ちの HIP の推論のテストと同じしきい値 {@value #LIMIT} 以内であること。
 */
@Test(groups = "rocm")
public class ThreePlayerAnchorGpuTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerAnchorGpuTest.class);

  /** HIP と eager の BF16 の s の、相対 RMS の平均の上限。 */
  private static final double LIMIT = 4e-2;

  /** HIP の s の FP32 からのずれの平均を、eager の BF16 の s のずれの平均で割った比の上限。 */
  private static final double ROUNDING = 1.25;

  /** 形ごとに作るネットワークの数（種を変えて作る）。 */
  private static final int SEEDS = 4;

  public void hipAnchorMatchesEagerAnchor() throws Exception {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(48, 900));
    rows.addAll(ThreePlayerRows.longRows(4));
    HostBatch batch = ThreePlayerRows.encode(rows);
    HostInputs inputs = new HostInputs(batch);
    BatchPlan plan = batch.plan();
    BatchPlan.Capacity capacity = StepFeed.capacity(InputSchema.FORMAT, FitPlan.TOKEN_BUDGET);
    Device gpu = Device.gpu(0);
    PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
    List<String> failures = new ArrayList<>();
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      double between = 0;
      double hipRounding = 0;
      double eagerRounding = 0;
      for (int seed = 0; seed < SEEDS; seed++) {
        Path root = Files.createTempDirectory("prism-3p-anchor");
        Path directory = root.resolve("model");
        try {
          try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
            ThreePlayerRows.network(manager, arm.shape(), 91 + 2L * seed + arm.ordinal()).save(directory);
          }
          float[] hip;
          float[] eager;
          float[] truth;
          try (PtStream stream = engine.newStream(gpu);
              Anchor fast = Anchor.open(directory, InferencePath.HIP, gpu, stream, null, capacity);
              Anchor slow = Anchor.open(directory, InferencePath.EAGER, gpu, stream, null, capacity);
              Anchor exact = Anchor.open(directory, InferencePath.EAGER, Device.cpu(), null, null, capacity)) {
            Assert.assertTrue(fast instanceof Anchor.Hip, arm.label() + ": HIP path on the GPU");
            Assert.assertTrue(slow instanceof Anchor.Eager, arm.label() + ": eager path on request");
            hip = ordered(fast.scores(inputs), plan);
            eager = ordered(slow.scores(inputs), plan);
            truth = ordered(exact.scores(inputs), plan);
          }
          Assert.assertTrue(squared(truth) > 0, arm.label() + ": anchor scores must be non-trivial");
          double pair = relative(eager, hip);
          double hipError = relative(truth, hip);
          double eagerError = relative(truth, eager);
          between += pair / SEEDS;
          hipRounding += hipError / SEEDS;
          eagerRounding += eagerError / SEEDS;
          log.info(
              "PRISM-3P-ANCHOR {} seed={} {} scores in {} rows: HIP vs eager BF16 relative RMS {}, from CPU FP32 HIP {}"
                  + " eager BF16 {}, argmax agreement HIP vs eager BF16 {} HIP vs FP32 {} eager BF16 vs FP32 {}",
              arm.label(),
              seed,
              hip.length,
              plan.rows(),
              String.format(Locale.ROOT, "%.3e", pair),
              String.format(Locale.ROOT, "%.3e", hipError),
              String.format(Locale.ROOT, "%.3e", eagerError),
              String.format(Locale.ROOT, "%.4f", agreement(hip, eager, plan)),
              String.format(Locale.ROOT, "%.4f", agreement(hip, truth, plan)),
              String.format(Locale.ROOT, "%.4f", agreement(eager, truth, plan)));
        } finally {
          Snapshot.deleteTree(root);
        }
      }
      log.info(
          "PRISM-3P-ANCHOR-SUMMARY {} seeds={} mean HIP vs eager BF16 {} (limit {}), mean from CPU FP32 HIP {} eager BF16 {}",
          arm.label(),
          SEEDS,
          String.format(Locale.ROOT, "%.3e", between),
          LIMIT,
          String.format(Locale.ROOT, "%.3e", hipRounding),
          String.format(Locale.ROOT, "%.3e", eagerRounding));
      if (!(between <= LIMIT)) {
        failures.add(arm.label() + " mean HIP vs eager BF16 relative RMS " + between);
      }
      if (!(hipRounding <= ROUNDING * eagerRounding)) {
        failures.add(
            arm.label() + " HIP is further from FP32 (" + hipRounding + ") than eager BF16 (" + eagerRounding + ")");
      }
    }
    Assert.assertEquals(failures, List.of());
  }

  /** アンカーの s を、バッチの手の順に並べ直す（行ごとの先頭の位置を使う）。 */
  private static float[] ordered(Anchor.Scores scores, BatchPlan plan) {
    Assert.assertEquals(scores.starts().length, plan.rows());
    float[] out = new float[plan.firstAction(plan.rows() - 1) + plan.actions(plan.rows() - 1)];
    for (int row = 0; row < plan.rows(); row++) {
      System.arraycopy(scores.values(), scores.starts()[row], out, plan.firstAction(row), plan.actions(row));
    }
    return out;
  }

  /** 行ごとの最善手が一致する行の割合。 */
  private static double agreement(float[] a, float[] b, BatchPlan plan) {
    int agree = 0;
    for (int row = 0; row < plan.rows(); row++) {
      int first = plan.firstAction(row);
      agree += best(a, first, plan.actions(row)) == best(b, first, plan.actions(row)) ? 1 : 0;
    }
    return agree / (double) plan.rows();
  }

  private static int best(float[] values, int first, int count) {
    int best = first;
    for (int k = first + 1; k < first + count; k++) {
      if (values[k] > values[best]) {
        best = k;
      }
    }
    return best;
  }

  private static double relative(float[] expected, float[] actual) {
    double error = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) expected[i] - actual[i];
      error += d * d;
    }
    return Math.sqrt(error / Math.max(squared(expected), 1e-30));
  }

  private static double squared(float[] values) {
    double sum = 0;
    for (float value : values) {
      sum += (double) value * value;
    }
    return sum;
  }
}
