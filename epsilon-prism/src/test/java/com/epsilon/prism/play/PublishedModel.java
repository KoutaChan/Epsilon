package com.epsilon.prism.play;

import ai.djl.Device;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.backend.KernelDevices;
import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;

/** 配布済みの方策と、固定した判断列における各合法手の参照スコア。 */
public record PublishedModel(String directory, Variant variant, PrismShape shape) {
  private static final String ROOT_PROPERTY = "prism.publishedModels";
  // CPU の行列積の丸め方によって FP32 の最下位付近に出る差を許す。
  private static final float CPU_FP32_SCORE_TOLERANCE = 5e-7f;

  @DataProvider
  public static Object[] fourPlayerModels() {
    return new Object[] {
      new PublishedModel("prism-v2b-v760", Variant.TENHOU_4P, new PrismShape.V2(256, 512, 256, 2))
    };
  }

  @DataProvider
  public static Object[] threePlayerModels() {
    var shape = new PrismShape.V3(256, 512, 512, 768, 1536, 1024, true);
    return new Object[] {new PublishedModel("prism3p-u3-v413", Variant.TENHOU_3P, shape)};
  }

  /** CPU のテストで読む配布モデル。置き場を渡していなければ飛ばす。 */
  public Path snapshot() {
    String root = System.getProperty(ROOT_PROPERTY);
    if (root == null)
      throw new SkipException("Set -D" + ROOT_PROPERTY + " to the published models directory");
    return Path.of(root).resolve(directory);
  }

  /** GPU の組のテストで読む配布モデル。置き場を渡していないか、モデルが無ければ失敗にする。 */
  public Path requiredSnapshot() {
    String root = System.getProperty(ROOT_PROPERTY);
    Assert.assertNotNull(root, "Set -D" + ROOT_PROPERTY + " to the published models directory");
    Path snapshot = Path.of(root).resolve(directory);
    Assert.assertTrue(Files.isDirectory(snapshot), "Missing published model " + snapshot);
    return snapshot;
  }

  /** 保存したモデルの CPU の eager のスコアを配布モデルの参照と照らす。FP32 は絶対誤差、BF16 はビットで照らす。 */
  public void assertEagerScores() throws IOException {
    assertScores(snapshot(), "cpu", Optional.empty(), Precision.values());
  }

  /**
   * この機械の GPU のランタイムのプロバイダーで通常の方策を開き、専用カーネルの参照スコアのビットと照らす。1 局面ずつの評価、固定の局面を
   * 全て 1 つのバッチにした提出、その並べ替えの 3 通りで照らす。配布モデルが無ければ失敗にする。
   */
  public void assertNativeScores() throws IOException {
    Path snapshot = requiredSnapshot();
    Optional<Backend> backend = Optional.of(KernelDevices.backend());
    assertScores(snapshot, "gpu0", backend, Precision.BFLOAT16);
    assertBatchedScores(snapshot, backend);
  }

  /**
   * @param expectedBackend 選ばれるはずの専用カーネルのバックエンド（eager なら空）
   */
  private void assertScores(
      Path snapshot, String device, Optional<Backend> expectedBackend, Precision... precisions)
      throws IOException {
    PrismShape saved = PrismNetwork.readShape(Snapshot.model(snapshot));
    Assert.assertEquals(saved, shape, directory);
    Kernels kernels = expectedBackend.isEmpty() ? Kernels.EAGER : Kernels.NATIVE;
    String path = KernelDevices.label(expectedBackend);
    var scores = PublishedScoreReference.load(path);
    ObservationBatch rows = PublishedPositions.sample(variant);
    Assert.assertEquals(
        PublishedPositions.fingerprint(rows),
        scores.getProperty(variant + ".inputs"),
        variant.name());
    for (Precision precision : precisions) {
      String key = directory + "." + path + "." + precision;
      String reference = scores.getProperty(key);
      Assert.assertNotNull(reference, key);
      var options = new PolicyOptions(device, precision, 1, kernels);
      try (var policy = ModelFamily.of(Series.PRISM).open(snapshot, variant, options)) {
        Assert.assertEquals(((PrismPolicy) policy).backend(), expectedBackend, key);
        float tolerance =
            expectedBackend.isEmpty() && precision == Precision.FLOAT32
                ? CPU_FP32_SCORE_TOLERANCE
                : 0.0f;
        assertScoreRows(policy, rows, reference.split(";"), key, tolerance);
      }
    }
  }

  /** 固定の局面を全て 1 つのバッチにし、そのままの順と並べ替えた順で提出して、局面ごとのスコアのビットを参照と照らす。 */
  private void assertBatchedScores(Path snapshot, Optional<Backend> backend) throws IOException {
    String path = KernelDevices.label(backend);
    String key = directory + "." + path + "." + Precision.BFLOAT16;
    String[] references = PublishedScoreReference.load(path).getProperty(key).split(";");
    ObservationBatch rows = PublishedPositions.sample(variant);
    int[] order = new int[rows.size()];
    for (int i = 0; i < order.length; i++) order[i] = i;
    int[] shuffled = order.clone();
    var random = new Random(13);
    for (int i = shuffled.length - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int swap = shuffled[i];
      shuffled[i] = shuffled[j];
      shuffled[j] = swap;
    }
    var settings =
        new InferenceSettings(
            List.of("gpu0"), Precision.BFLOAT16, Kernels.NATIVE, rows.size(), 1, 1, 1);
    try (var runtime = new NetworkRuntime(settings);
        var policy =
            Snapshot.open(
                snapshot,
                List.of(Device.gpu(0)),
                runtime,
                rows.size(),
                Precision.BFLOAT16,
                Kernels.NATIVE)) {
      Assert.assertEquals(policy.backend(), backend, key);
      var reader = new Observation(variant);
      for (int[] batch : List.of(order, shuffled)) {
        var requests = new ArrayList<PolicyRequest>();
        for (int index : batch) {
          reader.bind(rows, index);
          requests.add(new PolicyRequest(null, 0, index, null, null, true, 0, rows, index, reader));
        }
        var attempt = policy.tryAcquireIngress(0);
        Assert.assertTrue(attempt.acquired(), key);
        policy.submit(requests, attempt.ingress()).join();
        for (int i = 0; i < batch.length; i++) {
          assertRowScores(
              requests.get(i).scores, references[batch[i]], key + " batch row " + batch[i], 0.0f);
        }
      }
    }
  }

  private static void assertScoreRows(
      Policy policy, ObservationBatch rows, String[] references, String key, float tolerance) {
    Assert.assertEquals(rows.size(), references.length, key);
    var observation = new Observation(rows.variant());
    for (int row = 0; row < rows.size(); row++) {
      assertRowScores(
          policy.evaluate(observation.bind(rows, row)).values(),
          references[row],
          key + " row " + row,
          tolerance);
    }
  }

  /** 1 局面の全ての合法手のスコアを、参照の 16 進の並びと照らす。 */
  private static void assertRowScores(
      float[] actual, String reference, String label, float tolerance) {
    int[] expected =
        Arrays.stream(reference.trim().split("\\s+")).mapToInt(HexFormat::fromHexDigits).toArray();
    Assert.assertEquals(actual.length, expected.length, label);
    for (int action = 0; action < actual.length; action++) {
      String location = label + " action " + action;
      if (tolerance == 0.0f) {
        Assert.assertEquals(Float.floatToRawIntBits(actual[action]), expected[action], location);
      } else {
        Assert.assertEquals(
            actual[action], Float.intBitsToFloat(expected[action]), tolerance, location);
      }
    }
  }
}
