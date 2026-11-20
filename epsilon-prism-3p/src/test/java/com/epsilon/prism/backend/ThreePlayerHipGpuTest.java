package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.model.EagerStages;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 三人打ちの PRISM で、製品の入口（{@link NativeLane}）の推論が eager と一致するかを確かめる（rocm のグループ。GPU 1 枚）。ネットワークは
 * 三人打ちの入力形式で乱数で初期化した v2 の形と V3 で、行はエンジンで打った三人打ちの半荘の判断（全ての手の種類と、公開のイベントが
 * 多い行を含む）。
 *
 * <ul>
 *   <li>通しの s: eager（GPU の BF16 と CPU の FP32）と相対 RMS {@value #SCORE_LIMIT} 以内、CPU の参照と {@value #SCORE_REFERENCE_LIMIT} 以内。
 *       argmax は、eager 自身の BF16 の丸めで順位が変わらない行で {@value HipParity#ARGMAX_LIMIT} 以上一致する。全ての行と自信のある行の
 *       一致はログに出すだけ
 *   <li>同じバッチを 2 回実行しても s がビット単位で同じ。行を並べ替え、前に別の行を足してバッチを組み直しても、同じ行の s はビット単位で同じ
 * </ul>
 */
@Test(groups = "rocm")
public class ThreePlayerHipGpuTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerHipGpuTest.class);

  private static final double SCORE_LIMIT = 4e-2;
  private static final double SCORE_REFERENCE_LIMIT = 2e-2;

  /** 通しの s を比べる、ふつうのバッチの数。 */
  private static final int BATCHES = 4;

  /** 長い行のバッチに入れる、公開のイベントが多い行の数。 */
  private static final int LONG_ROWS = 12;

  @DataProvider
  public Object[][] tokenArms() {
    return new Object[][] {
      {ShapeArm.V2, 50}, {ShapeArm.V3, 51},
    };
  }

  /**
   * 通しの s を比べる。{@value #BATCHES} 個のふつうのバッチと長い行のバッチで、HIP の s を eager（GPU の BF16、CPU の FP32）と CPU の
   * 参照と比べる。argmax は、全ての行と自信のある行の一致をログに出し、eager 自身の BF16 の丸めで順位が変わらない行の一致を確かめる。
   */
  @Test(dataProvider = "tokenArms")
  public void scoresMatchEager(ShapeArm arm, int seed) throws IOException {
    List<HostBatch> batches = new ArrayList<>();
    for (int b = 0; b < BATCHES; b++) {
      batches.add(ThreePlayerRows.encode(ThreePlayerRows.mixedRows(GpuCase.ROWS, 600 + 13L * b)));
    }
    List<long[]> longRows = new ArrayList<>(ThreePlayerRows.mixedRows(2, 900));
    longRows.addAll(ThreePlayerRows.longRows(LONG_ROWS));
    HostBatch longBatch = ThreePlayerRows.encode(longRows);
    List<String> failures = new ArrayList<>();
    Path directory = Files.createTempDirectory("prism-3p-hip");
    try (GpuCase c =
            new GpuCase(
                arm.name(),
                manager -> ThreePlayerRows.network(manager, arm.shape(), seed),
                batches.getFirst());
        NDManager gpu = NDManager.newBaseManager(Device.gpu(0))) {
      Path model = directory.resolve("model");
      c.network.save(model);
      PrismNetwork onGpu = PrismNetwork.load(model, gpu);
      Scores scores = new Scores();
      Scores longScores = new Scores();
      for (HostBatch batch : batches) {
        scores.add(c, onGpu, gpu, batch);
      }
      longScores.add(c, onGpu, gpu, longBatch);
      scores.check(failures, arm.name());
      longScores.check(failures, "long " + arm.name());
    } catch (RuntimeException failure) {
      failures.add(arm.name() + " stopped: " + failure);
    } finally {
      Snapshot.deleteTree(directory);
    }
    Assert.assertTrue(failures.isEmpty(), failures.toString());
  }

  /** 同じバッチを 2 回実行しても s がビット単位で同じ。行を並べ替え、前に別の行を足して組み直しても、同じ行の s はビット単位で同じ（イベントの多い行も含む）。 */
  public void rebatchingKeepsRowScoresBitwise() {
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(GpuCase.ROWS - 4, 900));
    rows.addAll(ThreePlayerRows.longRows(4));
    List<long[]> shuffled = new ArrayList<>(rows);
    Collections.shuffle(shuffled, new Random(9));
    List<long[]> prefixed = new ArrayList<>(ThreePlayerRows.simulatedRows(7, 700));
    prefixed.addAll(shuffled);
    HostBatch first = ThreePlayerRows.encode(rows);
    HostBatch second = ThreePlayerRows.encode(prefixed);
    for (ThreePlayerRows.Arm arm : ThreePlayerRows.Arm.values()) {
      try (GpuCase c =
          new GpuCase(
              arm.label(),
              manager -> ThreePlayerRows.network(manager, arm.shape(), 61 + arm.ordinal()),
              first)) {
        float[] a = c.forward(first);
        float[] b = c.forward(second);
        float[] again = c.forward(first);
        Assert.assertTrue(
            HipParity.bitwiseEqual(a, again), arm.label() + ": two runs of one batch differ");
        int compared = 0;
        for (int r = 0; r < rows.size(); r++) {
          int s = prefixed.indexOf(rows.get(r));
          Assert.assertEquals(second.plan().actions(s), first.plan().actions(r));
          for (int k = 0; k < first.plan().actions(r); k++) {
            Assert.assertEquals(
                Float.floatToRawIntBits(b[second.plan().firstAction(s) + k]),
                Float.floatToRawIntBits(a[first.plan().firstAction(r) + k]),
                arm.label() + " row " + r + " action " + k);
            compared++;
          }
        }
        log.info(
            "PRISM-3P-HIP-T5 rebatch arm {}: {} scores bitwise equal, two runs bitwise equal",
            arm.label(),
            compared);
      }
    }
  }

  /** HIP・eager（GPU の BF16、CPU の FP32）・CPU の参照の s をバッチの順に集める。行ごとの手の位置と数、手の扱いも一緒に持つ。 */
  private static final class Scores {
    private final List<float[]> hip = new ArrayList<>();
    private final List<float[]> bf16 = new ArrayList<>();
    private final List<float[]> fp32 = new ArrayList<>();
    private final List<float[]> reference = new ArrayList<>();
    private final List<HipParity.Actions> actions = new ArrayList<>();

    void add(GpuCase c, PrismNetwork onGpu, NDManager gpu, HostBatch batch) {
      hip.add(c.forward(batch));
      try (NDManager scope = gpu.newSubManager()) {
        bf16.add(
            EagerStages.scores(
                onGpu,
                new ParameterStore(gpu, false),
                EagerStages.inputs(scope, batch),
                Precision.BFLOAT16));
      }
      try (NDManager scope = c.manager.newSubManager()) {
        fp32.add(
            EagerStages.scores(
                c.network,
                new ParameterStore(c.manager, false),
                EagerStages.inputs(scope, batch),
                Precision.FLOAT32));
      }
      reference.add(new HipReference(c.pack).forward(batch));
      actions.add(HipParity.Actions.of(batch));
    }

    /** 比べた結果をログに出し、しきい値を外れたものを失敗の一覧に足す。 */
    void check(List<String> failures, String label) {
      HipParity.Actions joined = join(actions);
      float[] h = concatenate(hip);
      float[] b = concatenate(bf16);
      float[] f = concatenate(fp32);
      float[] r = concatenate(reference);
      HipParity.Report half = HipParity.compare(h, b, joined);
      log.info(
          "PRISM-3P-HIP-T5 {} HIP vs eager GPU BF16: {} passes={}",
          label,
          half.summary(),
          half.passes());
      log.info(
          "PRISM-3P-HIP-T5 {} HIP vs eager CPU FP32: {}",
          label,
          HipParity.compare(h, f, joined).summary());
      log.info(
          "PRISM-3P-HIP-T5 {} HIP vs CPU reference: {}",
          label,
          HipParity.compare(h, r, joined).summary());
      line(failures, label, "s vs eager GPU BF16", Diff.fp32(b, h), SCORE_LIMIT);
      line(failures, label, "s vs eager CPU FP32", Diff.fp32(f, h), SCORE_LIMIT);
      line(failures, label, "s vs CPU reference", Diff.fp32(r, h), SCORE_REFERENCE_LIMIT);
      argmax(failures, label, h, b, f, joined);
    }

    private static void line(
        List<String> failures, String label, String name, Diff diff, double limit) {
      log.info("PRISM-3P-HIP-T5 {} {}", label, diff.line(name));
      if (!(diff.relativeRms() <= limit)) {
        failures.add(label + " " + name + " " + diff.relativeRms());
      }
    }
  }

  /**
   * argmax の一致をログに出し、BF16 の丸めで順位が変わらない行の一致を確かめる。乱数で初期化したネットワークには、1 位と 2 位の手の差が
   * 丸めの差より小さい行が多い。そのため、全ての行の一致（しきい値 {@value HipParity#ARGMAX_LIMIT}）と自信のある行の一致はログに出すだけにする。
   * 確かめる行は、eager の FP32 での 1 位と 2 位の差が、その行の eager の BF16 と FP32 の s の差の最大値の 2 倍を超える行。その行で HIP の
   * argmax が eager の FP32 と一致する割合が {@value HipParity#ARGMAX_LIMIT} 以上であることを確かめる。そういう行が 1 つも無ければ失敗にする。
   */
  private static void argmax(
      List<String> failures,
      String label,
      float[] hip,
      float[] bf16,
      float[] fp32,
      HipParity.Actions actions) {
    HipParity.Report half = HipParity.compare(hip, bf16, actions);
    int decided = 0;
    int agree = 0;
    for (int r = 0; r < actions.rowFirst().length; r++) {
      int first = actions.rowFirst()[r];
      int n = actions.rowCount()[r];
      if (n < 2) {
        continue;
      }
      double noise = 0;
      for (int k = first; k < first + n; k++) {
        noise = Math.max(noise, Math.abs((double) bf16[k] - fp32[k]));
      }
      if (gap(fp32, first, n) > 2 * noise) {
        decided++;
        agree += argmax(hip, first, n) == argmax(fp32, first, n) ? 1 : 0;
      }
    }
    double rate = decided == 0 ? 1 : (double) agree / decided;
    log.info(
        "PRISM-3P-HIP-T5 {} argmax vs eager GPU BF16 {} on all {} rows (line {} {}), confident {}"
            + " on {} rows; vs eager FP32 {} on {} rows that eager's own BF16 rounding does not"
            + " reorder",
        label,
        String.format(Locale.ROOT, "%.5f", half.argmaxAgreement()),
        half.rows(),
        HipParity.ARGMAX_LIMIT,
        half.argmaxAgreement() >= HipParity.ARGMAX_LIMIT ? "met" : "missed",
        String.format(Locale.ROOT, "%.5f", half.confidentAgreement()),
        half.confidentRows(),
        String.format(Locale.ROOT, "%.5f", rate),
        decided);
    if (decided == 0 || rate < HipParity.ARGMAX_LIMIT) {
      failures.add(
          label
              + " argmax on rows that BF16 rounding does not reorder "
              + rate
              + " ("
              + decided
              + " rows)");
    }
  }

  /** 行の中で、1 位と 2 位の手の s の差。 */
  private static double gap(float[] scores, int first, int count) {
    double top = Double.NEGATIVE_INFINITY;
    double second = Double.NEGATIVE_INFINITY;
    for (int k = first; k < first + count; k++) {
      if (scores[k] > top) {
        second = top;
        top = scores[k];
      } else if (scores[k] > second) {
        second = scores[k];
      }
    }
    return top - second;
  }

  private static int argmax(float[] scores, int first, int count) {
    int best = first;
    for (int k = first + 1; k < first + count; k++) {
      if (scores[k] > scores[best]) {
        best = k;
      }
    }
    return best;
  }

  /** バッチごとの手の位置・数・扱いを、s をつないだ並びに合わせて 1 つにまとめる。 */
  private static HipParity.Actions join(List<HipParity.Actions> parts) {
    List<Integer> first = new ArrayList<>();
    List<Integer> count = new ArrayList<>();
    List<Role> roles = new ArrayList<>();
    int offset = 0;
    for (HipParity.Actions part : parts) {
      for (int r = 0; r < part.rowFirst().length; r++) {
        first.add(offset + part.rowFirst()[r]);
        count.add(part.rowCount()[r]);
      }
      roles.addAll(Arrays.asList(part.roles()));
      offset += part.roles().length;
    }
    return new HipParity.Actions(
        first.stream().mapToInt(Integer::intValue).toArray(),
        count.stream().mapToInt(Integer::intValue).toArray(),
        roles.toArray(new Role[0]));
  }

  /** 配列を順につなぐ。 */
  private static float[] concatenate(List<float[]> parts) {
    float[] out = new float[parts.stream().mapToInt(part -> part.length).sum()];
    int at = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, out, at, part.length);
      at += part.length;
    }
    return out;
  }
}
