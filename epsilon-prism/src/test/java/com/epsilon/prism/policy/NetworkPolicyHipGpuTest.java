package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import com.epsilon.core.Action;
import com.epsilon.prism.backend.HipParity;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
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
 * HIP の方策と eager の方策の s の一致（rocm の組。GPU 1 枚）。同じ版（-Dprism.snapshot。なければ標準の形・深さ 2 の網に乱数を足した
 * 版）を 2 つの推論の道で開き、種の違う半荘を並べた同じ受け付けを両方に問い、s の差（HIP − eager）の中央値・p99・最大・手の役割ごとの
 * 符号つきの平均と argmax の一致率を、{@link HipParity} の基準で出す。
 *
 * <p>乱数の網では eager の BF16 の丸めの差が大きく、基準と比べる意味が薄い。だから基準の合否は記録に書くだけにし、試験が求めるのは、HIP の道で
 * 開けたこと、全ての問い合わせの答えが有限なこと、行の取り違えのない大きさ（相対 RMS {@value #SANITY_LIMIT} 以下）だけにする。
 */
public class NetworkPolicyHipGpuTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyHipGpuTest.class);

  /** 行や手の取り違えを見分ける相対 RMS の上限（BF16 の丸めの差は 1e-2 前後）。 */
  private static final double SANITY_LIMIT = 5e-2;

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
      Path snapshot = given == null ? root.resolve("1") : Path.of(given);
      if (given == null) {
        NetworkPolicyTest.write(snapshot, PrismShape.standard(), true);
      }
      PrismShape shape = PrismNetwork.readShape(Snapshot.model(snapshot));
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      List<Device> gpu = List.of(Device.gpu(0));
      Collected collected = new Collected();
      NetworkPolicy.Throughput eagerWork;
      NetworkPolicy.Throughput hipWork;
      try (var execution =
              new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
          var eager = open(snapshot, gpu, execution, config, InferencePath.EAGER);
          var hip = open(snapshot, gpu, execution, config, InferencePath.HIP)) {
        assertEquals(eager.inference(), InferencePath.EAGER);
        assertEquals(hip.inference(), InferencePath.HIP);
        NetworkPolicyTest.walk(
            games,
            batches,
            requests -> {
              List<PolicyRequest> copies = new ArrayList<>(requests.size());
              for (PolicyRequest request : requests) {
                copies.add(NetworkPolicyTest.copy(request));
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
          "PRISM-HIP-POLICY-T5 snapshot={} depth={} decisions={} HIP vs eager (GPU BF16): {} relativeRMS={} T5={}",
          given == null ? "perturbed standard" : given,
          shape.depth(),
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
      PolicyExecutionContext execution,
      PrismConfig config,
      InferencePath inference)
      throws Exception {
    return (NetworkPolicy)
        Snapshot.open(
            snapshot,
            devices,
            execution,
            config.inferBatch(),
            config.priorSettings(),
            inference);
  }

  /** 2 つの方策の答えを、HipParity の並び（行の手を続けた s と手の役割）に集める。 */
  private static final class Collected {
    private final List<Float> hip = new ArrayList<>();
    private final List<Float> eager = new ArrayList<>();
    private final List<Integer> first = new ArrayList<>();
    private final List<Integer> count = new ArrayList<>();
    private final List<ActionRole> roles = new ArrayList<>();

    void add(PolicyRequest hipRequest, PolicyRequest eagerRequest) {
      List<Action> legal = eagerRequest.legal();
      first.add(hip.size());
      count.add(legal.size());
      for (int a = 0; a < legal.size(); a++) {
        hip.add(hipRequest.scores[a]);
        eager.add(eagerRequest.scores[a]);
        roles.add(ActionRole.of(legal.get(a).type()));
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
