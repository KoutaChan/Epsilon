package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.prism.Limits;
import com.epsilon.prism.backend.HipReference;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * HIP の推論の道（CPU、native なし）。GPU の計画の代わりに、カーネルと同じ式・同じ丸めの CPU の参照（{@link HipReference}）を実行枠の道具にした
 * 方策で、HIP の道の組み立て（保存した網から作る重みの束と版の番号、ページ固定の blob の塊の区画、塊の順に詰めた s の並びと答え）を確かめる。
 *
 * <ul>
 *   <li>乱数で動かした標準の形（H 256）の版の s が、同じ重みの束の参照で同じ O を問い合わせの順の 1 つの塊にして直に回した s とビットで同じ。詰めた
 *       トークンの上限で塊を分けても（S の順に並べ替える）ビットで同じ（参照は行ごとに閉じた計算）。σ は貪欲の one-hot。
 *   <li>重みの束は読んだ網と合う: s が eager の BF16 の s と相対 RMS 2e-2 以内（HipReferenceTest の段の基準）。
 *   <li>道具の写しの版が方策の版と違えば、受け付けは例外で終わる（古い写しを使わない）。
 *   <li>HIP を選んでも、CPU・前線の腕・特化のない幅では eager の道になる（CPU の方策は native を読まずに開ける）。GPU で libprism_hip.so が
 *       なければ、英語の例外で止まる（eager に落ちない）。
 * </ul>
 */
public class NetworkPolicyHipTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyHipTest.class);

  private static final int GAMES = 6;
  private static final int BATCHES = 16;
  private static final int VERSION = 3;

  /** 段の相対 RMS の上限。 */
  private static final double STAGE_LIMIT = 2e-2;

  @Test(timeOut = 900_000)
  public void servedScoresMatchTheReference() throws Exception {
    Path root = Files.createTempDirectory("prism-hip-policy");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, PrismShape.standard(), true);
      Path model = snapshot.resolve(Snapshot.MODEL);
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      PriorSettings prior = config.priorSettings();
      List<Device> cpu = List.of(Device.cpu());
      int smallest = BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS;
      HipReference reference = new HipReference(WeightPack.load(model, VERSION));
      RequestBatch batch = new RequestBatch(false);
      double[] squares = new double[2];
      int[] counts = new int[3];
      try (NetworkPolicy whole =
              hip(model, config, prior, BatchEncoder.UNBOUNDED, ReferenceLane::new);
          NetworkPolicy split = hip(model, config, prior, smallest, ReferenceLane::new);
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
                copies.add(NetworkPolicyTest.copy(request));
              }
              NetworkPolicyTest.ask(whole, requests);
              NetworkPolicyTest.ask(split, copies);
              HostInputs inputs = batch.encode(requests);
              float[] expected = reference.forward(inputs.input());
              float[][] bf16 = eager.scores(requests, manager);
              for (int row = 0; row < requests.size(); row++) {
                PolicyRequest request = requests.get(row);
                int n = request.legal().size();
                int first = inputs.plan().firstAction(row);
                String where = "row " + row + " " + request.legal();
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
            "PRISM-HIP-POLICY batches={} actions={} widest={} chunks {} -> {} decisions={}"
                + " vs eager BF16 relative RMS={}",
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
        assertTrue(relative <= STAGE_LIMIT, "HIP s against eager BF16: " + relative);
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
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      try (NetworkPolicy policy =
          hip(
              model,
              config,
              config.priorSettings(),
              BatchEncoder.UNBOUNDED,
              pack -> new ReferenceLane(pack, pack.version() - 1))) {
        // 問い合わせの観測は打ち進めると変わるので、受け付けは歩みの中で問う
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
    assertEquals(InferencePath.HIP.resolve(Device.gpu(0), standard.withFrontier(true)), InferencePath.EAGER);
    assertEquals(InferencePath.HIP.resolve(Device.gpu(0), standard.withHidden(384)), InferencePath.EAGER);
    assertEquals(InferencePath.EAGER.resolve(Device.gpu(0), standard), InferencePath.EAGER);
    Path root = Files.createTempDirectory("prism-hip-cpu");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, new PrismShape(64, 128, 64, 2, false), false);
      PrismConfig config =
          PrismConfig.fromSettings(EpsilonSettings.of(Map.of("epsilon.prism.inference", "HIP")));
      assertEquals(config.inference(), InferencePath.HIP);
      try (PolicyService opened =
          Snapshot.open(
              snapshot,
              List.of(Device.cpu()),
              null,
              config.inferBatch(),
              config.priorSettings(),
              config.inference())) {
        NetworkPolicy policy = (NetworkPolicy) opened;
        assertEquals(policy.inference(), InferencePath.EAGER);
        NetworkPolicyTest.walk(
            2,
            1,
            requests -> {
              NetworkPolicyTest.ask(policy, requests);
              for (PolicyRequest request : requests) {
                assertEquals(request.scores.length, request.legal().size());
              }
            });
        assertTrue(policy.throughput().decisions() > 0, "the eager policy must answer");
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** GPU で HIP を選び、libprism_hip.so がない（-Pnative-hip で組み立てていない）なら、GPU に触る前に英語の例外で止まる（eager に落ちない）。 */
  @Test(timeOut = 900_000)
  public void hipWithoutTheNativeLibraryStops() throws Exception {
    if (HipLibrary.class.getResource("/native/linux-x86_64/libprism_hip.so") != null) {
      throw new SkipException("libprism_hip.so is on the classpath (built with -Pnative-hip)");
    }
    Path root = Files.createTempDirectory("prism-hip-missing");
    try {
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, PrismShape.standard(), false);
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      IllegalStateException failure =
          expectThrows(
              IllegalStateException.class,
              () ->
                  NetworkPolicy.open(
                      PolicyParams.of(1),
                      snapshot.resolve(Snapshot.MODEL),
                      List.of(Device.gpu(0)),
                      null,
                      config.inferBatch(),
                      config.priorSettings(),
                      InferencePath.HIP));
      assertTrue(failure.getMessage().contains("libprism_hip.so"), failure.getMessage());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 実行枠の道具を作る口の、重みの束だけを使う形。 */
  @FunctionalInterface
  private interface PackLanes {
    HipInference.Lane open(WeightPack pack);
  }

  /** 版 {@link #VERSION} の HIP の方策を呼び出しスレッドの口で開く（道具は試験の参照）。 */
  private static NetworkPolicy hip(
      Path model, PrismConfig config, PriorSettings prior, int tokenBudget, PackLanes lanes) throws Exception {
    return NetworkPolicy.hip(
        PolicyParams.of(VERSION),
        model,
        List.of(Device.cpu()),
        null,
        config.inferBatch(),
        prior,
        tokenBudget,
        (device, pack) -> lanes.open(pack));
  }
}
