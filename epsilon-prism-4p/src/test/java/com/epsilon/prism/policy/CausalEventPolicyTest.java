package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.Device;
import com.epsilon.agent.observe.EventRecords;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.runtime.EventCacheSettings;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * イベントを公開の順に保存する方策に、公開の接頭辞が伸びていく判断を照会する。この機械の GPU のランタイムのプロバイダーで開いた方策で、
 * ページを使う推論（CAUSAL）のスコアが、使わない推論（OFF）と全ての合法手で同じビットになることを確かめる（gpu の組。GPU 1 枚）。
 * モデルは V4_M の形の重みに乱数を足して作る。
 */
public class CausalEventPolicyTest {
  /** 照会する判断の数。 */
  private static final int OBSERVATIONS = 32;

  @Test(groups = "gpu")
  public void comparesGrowingPublishedPrefixes() throws Exception {
    ObservationBatch records = new ObservationBatch(TENHOU_4P, 4096);
    EventRecords path = new EventRecords(TENHOU_4P);
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    Observation reader = new Observation(TENHOU_4P);
    Path root = Files.createTempDirectory("prism-causal-policy");
    try {
      new TableWalk(TENHOU_4P, TableWalk.Policy.CALLS)
          .run(
              731L,
              1,
              new TableWalk.Visitor() {
                @Override
                public void decision(com.epsilon.mahjong.engine.Engine engine, int slot) {
                  var decisions = engine.decisions();
                  if (records.size() >= OBSERVATIONS || decisions.legalCount(slot) < 2) return;
                  int seat = decisions.seat(slot);
                  int index =
                      writer.capture(engine.view(seat), decisions, slot, records, path, reader);
                  records.rememberEvents(index, path.prefix(seat, reader.kanDoraIndicators()));
                }

                @Override
                public void roundEnd(com.epsilon.mahjong.engine.Engine engine) {
                  path.reset();
                }
              });
      Assert.assertTrue(records.size() > 0);
      Path snapshot = root.resolve("1");
      NetworkPolicyTest.write(snapshot, ShapeArm.V4_M.shape(), true);
      var expected = score(snapshot, records, EventCacheSettings.Mode.OFF);
      var actual = score(snapshot, records, EventCacheSettings.Mode.CAUSAL);
      Assert.assertEquals(actual.size(), expected.size());
      for (int row = 0; row < expected.size(); row++) {
        float[] a = actual.get(row), b = expected.get(row);
        Assert.assertEquals(a.length, b.length);
        for (int i = 0; i < b.length; i++) {
          Assert.assertTrue(Float.isFinite(a[i]) && Float.isFinite(b[i]));
          Assert.assertEquals(
              Float.floatToRawIntBits(a[i]),
              Float.floatToRawIntBits(b[i]),
              "row " + row + " action " + i);
        }
      }
    } finally {
      records.clear();
      path.releasePrefixes();
      Snapshot.deleteTree(root);
    }
  }

  /** 方策を専用カーネルで開き、判断を 2 つずつ、同じ接頭辞の判断と前に公開した判断と一緒に照会して、全てのスコアを照会の順に返す。 */
  private static List<float[]> score(
      Path snapshot, ObservationBatch records, EventCacheSettings.Mode mode) throws Exception {
    var settings =
        new InferenceSettings(
            List.of("gpu0"),
            Precision.BFLOAT16,
            Kernels.NATIVE,
            4,
            1,
            2,
            2,
            new EventCacheSettings(mode, 1024, 0.1));
    var result = new ArrayList<float[]>();
    var reader = new Observation(TENHOU_4P);
    try (var runtime = new NetworkRuntime(settings);
        var policy =
            Snapshot.open(
                snapshot, List.of(Device.gpu(0)), runtime, 4, Precision.BFLOAT16, Kernels.NATIVE)) {
      Assert.assertNotEquals(policy.inference(), Kernels.EAGER);
      Assert.assertEquals(policy.reuseEventPrefix(), mode == EventCacheSettings.Mode.CAUSAL);
      for (int at = 0; at < records.size(); at += 2) {
        // 同じ接頭辞を持つ行と、前の受付で公開済みの行を一緒に照会する。
        int[] batch = {at, Math.min(at + 1, records.size() - 1), at, Math.max(0, at - 2)};
        int lane = mode == EventCacheSettings.Mode.OFF ? 0 : (at / 2) % 2;
        var requests = new ArrayList<PolicyRequest>();
        for (int index : batch) {
          reader.bind(records, index);
          requests.add(
              new PolicyRequest(null, lane, index, null, null, true, lane, records, index, reader));
        }
        var attempt = policy.tryAcquireIngress(lane);
        Assert.assertTrue(attempt.acquired());
        policy.submit(requests, attempt.ingress()).join();
        for (int i = 0; i < batch.length; i++) {
          reader.bind(records, batch[i]);
          float[] values = requests.get(i).scores;
          Assert.assertEquals(values.length, reader.legalActions());
          result.add(values);
        }
      }
    }
    return result;
  }
}
