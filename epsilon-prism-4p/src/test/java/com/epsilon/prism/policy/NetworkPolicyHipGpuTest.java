package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.action.Action;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.backend.HipParity;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * HIP の方策と eager の方策で s が一致するかのテスト（テストグループ rocm。GPU 1 枚）。同じバージョンを 2 つの推論のバックエンドで開く。
 * バージョンは -Dprism.snapshot で渡し、無ければ -Dprism.arm の形のネットワークに乱数を足して作る。乱数の種の違う半荘を並べた同じ受付を
 * 両方に送り、s の差（HIP − eager）の中央値・p99・最大、手の種類ごとの符号つきの平均、argmax の一致率を {@link HipParity} の基準で出す。
 *
 * <p>乱数のネットワークでは eager の BF16 の丸めの差が大きく、基準と比べる意味が薄い。だから基準の合否はログに書くだけにする。テストが
 * 求めるのは、HIP のバックエンドで開けたこと、全ての要求の答えが有限なこと、行の取り違えが無いと言える差（相対 RMS
 * {@value #SANITY_LIMIT} 以下）の 3 つだけ。
 */
public class NetworkPolicyHipGpuTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyHipGpuTest.class);

  /** 行や手の取り違えを見分ける相対 RMS の上限。BF16 の丸めの差は 1e-2 前後。 */
  private static final double SANITY_LIMIT = 5e-2;

  /** -Dprism.snapshot が無いときに作るバージョンの形のアームを渡すシステムプロパティ（{@link HipFixtures.Arm} の名前。既定は v2 の本番の形）。 */
  private static final String ARM_PROPERTY = "prism.arm";

  @Test(groups = "rocm")
  public void hipPolicyMatchesEager() throws Exception {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("needs a GPU");
    }
    int games = Integer.getInteger("prism.games", 64);
    int batches = Integer.getInteger("prism.batches", 200);
    Path root = Files.createTempDirectory("prism-hip-gpu");
    try {
      String given = System.getProperty("prism.snapshot");
      HipFixtures.Arm arm = HipFixtures.Arm.valueOf(System.getProperty(ARM_PROPERTY, HipFixtures.Arm.V2_DEPTH_2.name()));
      Path snapshot = given == null ? root.resolve("1") : Path.of(given);
      if (given == null) {
        NetworkPolicyTest.write(snapshot, arm.shape(), true);
      }
      PrismShape shape = PrismNetwork.readShape(Snapshot.model(snapshot));
      InferenceSettings settings = TestSettings.of().inference();
      List<Device> gpu = List.of(Device.gpu(0));
      // 推論のセクションの既定の行の上限は CPU 向けの値なので、1 回に送る行（半荘の数）が入る大きさにする。HIP は s のバッファをこの大きさで取る
      int maxBatch = Math.max(settings.maxBatch(), games);
      Collected collected = new Collected();
      NetworkPolicy.Throughput eagerWork;
      NetworkPolicy.Throughput hipWork;
      try (var execution = new NetworkRuntime(settings.encoders(), settings.slots(), settings.readyBatches());
          var eager = open(snapshot, gpu, execution, maxBatch, InferencePath.EAGER);
          var hip = open(snapshot, gpu, execution, maxBatch, InferencePath.HIP)) {
        assertEquals(eager.inference(), InferencePath.EAGER);
        assertEquals(hip.inference(), InferencePath.HIP);
        NetworkPolicyTest.walk(
            games,
            batches,
            requests -> {
              List<PolicyRequest> copies = new ArrayList<>(requests.size());
              for (PolicyRequest request : requests) {
                copies.add(PolicyRequests.copy(request));
              }
              NetworkPolicyTest.ask(eager, requests);
              NetworkPolicyTest.ask(hip, copies);
              for (int row = 0; row < requests.size(); row++) {
                collected.add(copies.get(row), requests.get(row));
              }
            });
        eagerWork = eager.throughput();
        hipWork = hip.throughput();
      }
      HipParity.Report report = collected.compare();
      log.info(
          "PRISM-HIP-POLICY-T5 snapshot={} shape={} decisions={} HIP vs eager (GPU BF16): {} relativeRMS={} T5={}",
          given == null ? "perturbed " + arm.label() : given,
          shape,
          report.rows(),
          report.summary(),
          String.format(Locale.ROOT, "%.3e", collected.relativeRms()),
          report.passes() ? "PASS" : "FAIL");
      log.info(
          "PRISM-HIP-POLICY-WORK one batch at a time: eager {} decisions {} s, HIP {} decisions {} s",
          eagerWork.decisions(),
          String.format(Locale.ROOT, "%.1f", eagerWork.seconds()),
          hipWork.decisions(),
          String.format(Locale.ROOT, "%.1f", hipWork.seconds()));
      assertTrue(report.rows() > 0, "no decisions were compared");
      assertTrue(collected.finite(), "HIP or eager returned a non-finite score");
      assertTrue(
          collected.relativeRms() <= SANITY_LIMIT, "HIP s against eager s: " + collected.relativeRms());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  private static NetworkPolicy open(
      Path snapshot,
      List<Device> devices,
      NetworkRuntime execution,
      int maxBatch,
      InferencePath inference)
      throws Exception {
    // HIP は BF16 で計算するので、eager も GPU の BF16 で比べる
    return Snapshot.open(snapshot, devices, execution, maxBatch, Precision.BFLOAT16, inference);
  }

  /** 2 つの方策の答えを、HipParity が受ける形（行の手の s を続けて並べた列と、手の種類）に集める。 */
  private static final class Collected {
    private final List<Float> hip = new ArrayList<>();
    private final List<Float> eager = new ArrayList<>();
    private final List<Integer> first = new ArrayList<>();
    private final List<Integer> count = new ArrayList<>();
    private final List<ActionRole> roles = new ArrayList<>();

    void add(PolicyRequest hipRequest, PolicyRequest eagerRequest) {
      int[] legal = eagerRequest.legal();
      first.add(hip.size());
      count.add(legal.length);
      for (int a = 0; a < legal.length; a++) {
        hip.add(hipRequest.scores[a]);
        eager.add(eagerRequest.scores[a]);
        roles.add(ActionRole.of(Action.of(legal[a]).type()));
      }
    }

    HipParity.Report compare() {
      return HipParity.compare(
          floats(hip),
          floats(eager),
          new HipParity.Actions(
              first.stream().mapToInt(Integer::intValue).toArray(),
              count.stream().mapToInt(Integer::intValue).toArray(),
              roles.toArray(ActionRole[]::new)));
    }

    double relativeRms() {
      double squares = 0;
      double norm = 0;
      for (int a = 0; a < hip.size(); a++) {
        double delta = (double) hip.get(a) - eager.get(a);
        squares += delta * delta;
        norm += (double) eager.get(a) * eager.get(a);
      }
      return norm > 0 ? Math.sqrt(squares / norm) : Math.sqrt(squares);
    }

    boolean finite() {
      for (int a = 0; a < hip.size(); a++) {
        if (!Float.isFinite(hip.get(a)) || !Float.isFinite(eager.get(a))) {
          return false;
        }
      }
      return true;
    }

    private static float[] floats(List<Float> values) {
      float[] result = new float[values.size()];
      for (int i = 0; i < result.length; i++) {
        result[i] = values.get(i);
      }
      return result;
    }
  }
}
