package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.HipReference;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * HIP の推論のバックエンドのテスト（CPU、native なし）。GPU の計画の代わりに、カーネルと同じ式・同じ丸めの CPU の参照
 * （{@link HipReference}）をスロットのレーンにした方策を使う。これで HIP のバックエンドの組み立てを確かめる。確かめるのは、保存した
 * ネットワークから作る重みの束とバージョン番号、ページ固定の blob の中のバッチの領域、バッチの順に並べた s と答え。
 *
 * <ul>
 *   <li>乱数を足した v2 の本番の形と v3 の形（H 256）のバージョンで、s が参照の s とビットで同じ。参照の s は、同じ重みの束で、同じ O を
 *       要求の順に 1 つのバッチにして直に計算したもの。トークンの上限でバッチを分けても（S の順に並べ替える）ビットで同じ。参照は行ごとに
 *       閉じた計算だから。σ は貪欲に選んだ手の one-hot。
 *   <li>重みの束は読んだネットワークと合う。s と eager の BF16 の s の差は相対 RMS 4e-2 以内。層 0 の q・k・v は折り込んだ表から 1 回の
 *       丸めで作るので、x と n も丸める eager とは丸めの位置が違う。この乱数を足したネットワークでは、折り込む前に 1.6e-2 だった差が
 *       2.6e-2 になり、ふだんの基準 2e-2 を超える。束とネットワークが食い違えば（置く場所や並びの誤り）、差は O(1) になる。
 *   <li>レーンに写した重みのバージョンが方策のバージョンと違えば、受付は例外で終わる（古い重みを使わない）。
 *   <li>HIP を選んでも、CPU とカーネルが対応しない幅では eager のバックエンドになる（CPU の方策は native を読まずに
 *       開ける）。GPU で libprism_hip.so が無ければ、英語の例外で止まる（eager に切り替えない）。
 * </ul>
 */
public class NetworkPolicyHipTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyHipTest.class);

  private static final int GAMES = 6;
  private static final int BATCHES = 16;
  private static final int VERSION = 3;

  /** CPU の方策の行の上限（推論のセクションの既定値）。 */
  private static final int MAX_BATCH = 8;

  /** eager の BF16 の s との相対 RMS の上限。層 0 の丸めの位置の違いによる差を含む（クラスの説明を参照）。 */
  private static final double EAGER_LIMIT = 4e-2;

  /** 方策を確かめる形。v2 の本番の形と、型ごとの FFN・候補層の 2 段目を持つ v3 の形。 */
  @DataProvider
  public Object[][] shapes() {
    return new Object[][] {{HipFixtures.Arm.V2_DEPTH_2}, {HipFixtures.Arm.V3}};
  }

  @Test(timeOut = 900_000, dataProvider = "shapes")
  public void servedScoresMatchTheReference(HipFixtures.Arm arm) throws Exception {
    Path root = Files.createTempDirectory("prism-hip-policy");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, arm.shape(), true);
      Path model = snapshot.resolve(Snapshot.MODEL);
      PriorSettings prior = PriorSettings.STANDARD;
      List<Device> cpu = List.of(Device.cpu());
      int smallest = InputSchema.FORMAT.fixedTokens() + Limits.MAX_EVENT_TOKENS;
      HipReference reference = new HipReference(WeightPack.load(model, VERSION));
      RequestBatch batch = new RequestBatch();
      double[] squares = new double[2];
      int[] counts = new int[3];
      try (NetworkPolicy whole =
              hip(model, prior, BatchEncoder.UNBOUNDED, ReferenceLane::new);
          NetworkPolicy split = hip(model, prior, smallest, ReferenceLane::new);
          NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        assertEquals(whole.inference(), InferencePath.HIP);
        NetworkPolicyTest.Direct eager =
            new NetworkPolicyTest.Direct(PrismNetwork.load(model, manager), Precision.BFLOAT16);
        NetworkPolicyTest.walk(
            GAMES,
            BATCHES,
            requests -> {
              List<PolicyRequest> copies = new ArrayList<>(requests.size());
              for (PolicyRequest request : requests) {
                copies.add(PolicyRequests.copy(request));
              }
              NetworkPolicyTest.ask(whole, requests);
              NetworkPolicyTest.ask(split, copies);
              HostInputs inputs = batch.encode(requests);
              float[] expected = reference.forward(inputs.input());
              float[][] bf16 = eager.scores(requests, manager);
              for (int row = 0; row < requests.size(); row++) {
                PolicyRequest request = requests.get(row);
                int n = request.legalCount();
                int first = inputs.plan().firstAction(row);
                String where = "row " + row + " " + Arrays.toString(request.legal());
                for (int a = 0; a < n; a++) {
                  assertEquals(
                      Float.floatToRawIntBits(request.scores[a]),
                      Float.floatToRawIntBits(expected[first + a]),
                      "HIP s " + a + " " + where);
                  assertEquals(
                      Float.floatToRawIntBits(copies.get(row).scores[a]),
                      Float.floatToRawIntBits(request.scores[a]),
                      "split HIP s " + a + " " + where);
                  double delta = (double) request.scores[a] - bf16[row][a];
                  squares[0] += delta * delta;
                  squares[1] += (double) bf16[row][a] * bf16[row][a];
                }
                assertEquals(request.sigma[Greedy.argmax(request.scores, request.prior, n)], 1f, where);
                counts[1] += n;
              }
              counts[0]++;
              counts[2] = Math.max(counts[2], requests.size());
            });
        NetworkPolicy.Throughput parts = split.throughput();
        double relative = Math.sqrt(squares[0] / squares[1]);
        log.info(
            "PRISM-HIP-POLICY arm={} batches={} actions={} widest={} chunks {} -> {} decisions={}"
                + " vs eager BF16 relative RMS={}",
            arm.label(),
            counts[0],
            counts[1],
            counts[2],
            whole.throughput().chunks(),
            parts.chunks(),
            whole.throughput().decisions(),
            String.format("%.3e", relative));
        assertTrue(parts.chunks() > parts.batches(), "the small budget splits batches");
        assertTrue(counts[2] > 3, "batches must hold several rows: " + counts[2]);
        assertTrue(squares[1] > 0, "the perturbed network must give nonzero scores");
        assertTrue(relative <= EAGER_LIMIT, "HIP s against eager BF16: " + relative);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test(timeOut = 900_000)
  public void staleWeightCopiesFailTheBatch() throws Exception {
    Path root = Files.createTempDirectory("prism-hip-stale");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, PrismShape.standard(), false);
      Path model = snapshot.resolve(Snapshot.MODEL);
      try (NetworkPolicy policy =
          hip(
              model,
              PriorSettings.STANDARD,
              BatchEncoder.UNBOUNDED,
              pack -> new ReferenceLane(pack, pack.version() - 1))) {
        // 要求の観測は打ち進めると変わるので、受付は打ち進める途中で送る
        int[] asked = new int[1];
        NetworkPolicyTest.walk(
            1,
            1,
            requests -> {
              CompletionException failure =
                  expectThrows(
                      CompletionException.class, () -> NetworkPolicyTest.ask(policy, requests));
              Throwable cause = failure.getCause();
              assertTrue(cause instanceof IllegalStateException, String.valueOf(cause));
              assertTrue(cause.getMessage().contains("stale"), cause.getMessage());
              asked[0]++;
            });
        assertEquals(asked[0], 1);
        assertEquals(policy.throughput().decisions(), 0L);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test(timeOut = 900_000)
  public void hipFallsBackToEagerWhereTheKernelsDoNotRun() throws Exception {
    PrismShape standard = PrismShape.standard();
    assertEquals(InferencePath.HIP.resolve(Device.gpu(0), standard), InferencePath.HIP);
    assertEquals(InferencePath.HIP.resolve(Device.gpu(1), PrismShape.standard(3)), InferencePath.HIP);
    assertEquals(InferencePath.HIP.resolve(Device.cpu(), standard), InferencePath.EAGER);
    assertEquals(InferencePath.HIP.resolve(Device.gpu(0), new PrismShape(384, 512, 256, 2, null)), InferencePath.EAGER);
    assertEquals(InferencePath.EAGER.resolve(Device.gpu(0), standard), InferencePath.EAGER);
    Path root = Files.createTempDirectory("prism-hip-cpu");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, new PrismShape(64, 128, 64, 2, null), false);
      PrismSettings settings = TestSettings.of("inference.kernels=NATIVE");
      InferencePath inference = InferencePath.of(settings.inference().kernels());
      assertEquals(inference, InferencePath.HIP);
      try (PolicyService opened =
          Snapshot.open(
              snapshot,
              List.of(Device.cpu()),
              null,
              settings.inference().maxBatch(),
              settings.inference().precision(),
              inference)) {
        NetworkPolicy policy = (NetworkPolicy) opened;
        assertEquals(policy.inference(), InferencePath.EAGER);
        NetworkPolicyTest.walk(
            2,
            1,
            requests -> {
              NetworkPolicyTest.ask(policy, requests);
              for (PolicyRequest request : requests) {
                assertEquals(request.scores.length, request.legalCount());
              }
            });
        assertTrue(policy.throughput().decisions() > 0, "the eager policy must answer");
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** GPU で HIP を選んだのに libprism_hip.so が無い（-Pnative-hip でビルドしていない）と、GPU に触る前に英語の例外で止まる（eager に切り替えない）。 */
  @Test(timeOut = 900_000)
  public void hipWithoutTheNativeLibraryStops() throws Exception {
    if (HipLibrary.class.getResource("/native/linux-x86_64/libprism_hip.so") != null) {
      throw new SkipException("libprism_hip.so is on the classpath (built with -Pnative-hip)");
    }
    Path root = Files.createTempDirectory("prism-hip-missing");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, PrismShape.standard(), false);
      IllegalStateException failure =
          expectThrows(
              IllegalStateException.class,
              () ->
                  NetworkPolicy.open(
                      PolicyParams.of(1),
                      snapshot.resolve(Snapshot.MODEL),
                      List.of(Device.gpu(0)),
                      null,
                      MAX_BATCH,
                      Precision.BFLOAT16,
                      InferencePath.HIP));
      assertTrue(failure.getMessage().contains("libprism_hip.so"), failure.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** スロットのレーンを重みの束だけから作るインターフェース。 */
  @FunctionalInterface
  private interface PackLanes {
    HipInference.Lane open(WeightPack pack);
  }

  /** バージョン {@link #VERSION} の HIP の方策を、呼び出したスレッドで評価する形で開く。レーンはテストで渡したもの。 */
  private static NetworkPolicy hip(Path model, PriorSettings prior, int tokenBudget, PackLanes lanes) throws Exception {
    return NetworkPolicy.hip(
        PolicyParams.of(VERSION),
        model,
        List.of(Device.cpu()),
        null,
        MAX_BATCH,
        prior,
        tokenBudget,
        (device, pack) -> lanes.open(pack));
  }
}
