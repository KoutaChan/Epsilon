package com.epsilon.prism.collect;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * GRP の Φ の値は公開系列だけで決まる: 要求の来る間隔（バッチの組み方）とメモの追い出しを変えても、同じ系列には最後のビットまで同じ値を返し、それは
 * {@link EpsilonGrpInference#maxBatch} 行ずつの評価と一致する。ドライバのスレッドがメモを持ち、複数のスレッドが要求を投げる収集と同じ形で確かめる。
 */
public class FixedGrpPhiDeterminismTest {
  private static final int PRODUCERS = 8, PREFIXES = 2000, REQUESTS = 1500, MEMO = 512;

  @Test(groups = "native")
  public void valuesDependOnlyOnThePrefix() throws Exception {
    List<float[]> prefixes = prefixes(new SplittableRandom(7));
    try (Model model = Model.newInstance("grp", Device.cpu(), "PyTorch")) {
      var block = new EpsilonGrpNetwork();
      model.setBlock(block);
      block.initialize(
          model.getNDManager(),
          DataType.FLOAT32,
          new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
          new Shape(-1));
      var inference = new EpsilonGrpInference(model, 64);
      double[][] expected = expected(inference, prefixes);
      // 間隔 0 は大きなバッチ、2 ms は 1〜数件のバッチ（GRU の別の計算になる 1〜3 行）を作る
      for (long gapNanos : new long[] {0, 2_000_000}) {
        double[][] values = collect(inference, prefixes, gapNanos);
        for (int id = 0; id < PREFIXES; id++)
          if (values[id] != null)
            Assert.assertEquals(
                Arrays.stream(values[id]).mapToLong(Double::doubleToRawLongBits).toArray(),
                Arrays.stream(expected[id]).mapToLong(Double::doubleToRawLongBits).toArray(),
                "prefix " + id + " gap " + gapNanos);
      }
    }
  }

  private static double[][] collect(EpsilonGrpInference inference, List<float[]> prefixes, long gapNanos)
      throws Exception {
    ExecutorService driver = Executors.newSingleThreadExecutor();
    double[][] values = new double[PREFIXES][];
    int[] conflicts = {0};
    CountDownLatch done = new CountDownLatch(PRODUCERS * REQUESTS);
    try (var phi = new FixedGrpPhi(inference, EpsilonUtilityProfile.TENHOU, driver::execute, 256, MEMO)) {
      List<Thread> producers = new ArrayList<>();
      for (int p = 0; p < PRODUCERS; p++) {
        var random = new SplittableRandom(100 + p);
        producers.add(
            Thread.ofPlatform()
                .start(
                    () -> {
                      for (int i = 0; i < REQUESTS; i++) {
                        int id = random.nextInt(PREFIXES);
                        driver.execute(
                            () ->
                                phi.request(
                                    prefixes.get(id),
                                    new int[4],
                                    result -> {
                                      if (values[id] == null) values[id] = result;
                                      else if (!Arrays.equals(values[id], result)) conflicts[0]++;
                                      done.countDown();
                                    }));
                        if (gapNanos > 0) LockSupport.parkNanos(random.nextLong(2 * gapNanos));
                      }
                    }));
      }
      Assert.assertTrue(done.await(5, TimeUnit.MINUTES));
      for (Thread producer : producers) producer.join();
      // 間隔のある組ではメモが溢れて追い出し、同じ系列を別のバッチで評価し直す
      if (gapNanos > 0) Assert.assertTrue(phi.stats().evaluated() > PREFIXES, "the memo must evict and re-evaluate");
    } finally {
      driver.shutdown();
      driver.awaitTermination(1, TimeUnit.MINUTES);
    }
    Assert.assertEquals(conflicts[0], 0, "re-evaluation after eviction changed a value");
    return values;
  }

  /** maxBatch 行ずつ（最後は写しで埋める）に通した値。 */
  private static double[][] expected(EpsilonGrpInference inference, List<float[]> prefixes) {
    var padded = new ArrayList<>(prefixes);
    while (padded.size() % inference.maxBatch() != 0) padded.add(prefixes.getLast());
    List<float[]> marginals = inference.predictMarginalProbabilities(padded);
    double[][] out = new double[PREFIXES][4];
    for (int id = 0; id < PREFIXES; id++)
      for (int seat = 0; seat < 4; seat++)
        out[id][seat] =
            EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(
                EpsilonUtilityProfile.TENHOU, marginals.get(id), seat * 4);
    return out;
  }

  /** 局の数 1〜12 の公開系列（点数は和了と放銃の移動、局番号は進むか本場が積まれる）。 */
  private static List<float[]> prefixes(SplittableRandom random) {
    List<float[]> out = new ArrayList<>(PREFIXES);
    for (int n = 0; n < PREFIXES; n++) {
      int steps = 1 + random.nextInt(12);
      float[] prefix = new float[steps * EpsilonGrpFeature.FEATURE_SIZE];
      int[] scores = {25000, 25000, 25000, 25000};
      int kyoku = 0, honba = 0;
      for (int step = 0; step < steps; step++) {
        float[] feature = EpsilonGrpFeature.fromProgress(kyoku, honba, 0, scores);
        System.arraycopy(feature, 0, prefix, step * feature.length, feature.length);
        int winner = random.nextInt(4), loser = random.nextInt(4), points = 100 * (10 + random.nextInt(120));
        if (winner != loser) {
          scores[winner] += points;
          scores[loser] -= points;
        }
        if (random.nextInt(3) == 0) honba++;
        else {
          honba = 0;
          kyoku = Math.min(kyoku + 1, 11);
        }
      }
      out.add(prefix);
    }
    return out;
  }
}
