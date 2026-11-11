package com.epsilon.prism.hip;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;

import java.lang.ref.Reference;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * K3 の速さのゲート。GPU 1 枚で測る。実効の速さは 2·トークン数·(H·A + 2·H·F) / カーネルの時間で、
 * 時間は GPU のイベントで測った中央値（転送は含めない）。型ごとの FFN では、F に区間ごとの幅を使う。最後の全トークンの層では、イベントの
 * トークンごとに、キーとバリューの組の数 × 2·H·2A を足す。TFLOPS は MFLOP/µs と同じ単位。
 */
@Test(groups = "rocm")
public class MlpBenchmarkTest {

  private static final Logger log = LoggerFactory.getLogger(MlpBenchmarkTest.class);

  private static final int ROWS = 2048;
  private static final int TOKENS_PER_ROW = 74;
  private static final int FFN = 512;
  private static final double GATE_TFLOPS = 15.0;
  private static final double TYPED_GATE = 0.95;
  private static final int WARMUP = 10;
  private static final int ITERATIONS = 50;

  /** 計画の上で 1 つのカーネルの時間を測る。閉じると計画も閉じる。 */
  private record Measured(HipPlan plan, MlpCase c, float[] millis) implements AutoCloseable {

    /** 計画を作り、重みと入力を GPU に写す。写し元のバッファは、写し終わるまで回収させない。 */
    static Measured open(HipLibrary library, PtStream stream, MlpCase c) {
      HipPlan plan = HipPlan.create(library, stream, c.shape());
      ByteBuffer residual = c.input(HipRegion.RESIDUAL);
      ByteBuffer attention = c.input(HipRegion.ATTENTION);
      plan.uploadWeights(c.pack(), 1);
      plan.copyIn(HipRegion.RESIDUAL, residual, c.bytes(HipRegion.RESIDUAL));
      plan.copyIn(HipRegion.ATTENTION, attention, c.bytes(HipRegion.ATTENTION));
      plan.synchronize();
      Reference.reachabilityFence(residual);
      Reference.reachabilityFence(attention);
      plan.profile(true);
      return new Measured(plan, c, new float[ITERATIONS]);
    }

    /** カーネルを 1 回回す。{@code i} はウォームアップも含めて数えた回の番号で、ウォームアップの回の時間は捨てる。 */
    void run(int i) {
      plan.run(c.kernel, c.layer, c.batch());
      float ms = plan.kernelMillis()[c.kernel.ordinal()];
      if (i >= WARMUP) {
        millis[i - WARMUP] = ms;
      }
    }

    /** 時間の中央値（ms）。 */
    double median() {
      float[] sorted = millis.clone();
      Arrays.sort(sorted);
      return sorted[ITERATIONS / 2];
    }

    /** 時間の中央値で求めた実効の TFLOPS。 */
    double tflops() {
      return flops(c) / (median() * 1e-3) / 1e12;
    }

    @Override
    public void close() {
      plan.close();
    }
  }

  /** 本番の形（H=256・F=512、1 バッチ 2048 行 × 平均 74 トークン）で、実効の速さが 15 TFLOPS 以上になる。 */
  public void reachesTheGate() {
    int tokens = ROWS * TOKENS_PER_ROW;
    HipArchitecture shared = TestPack.architecture(FFN, TestPack.FOUR_PLAYERS);
    MlpCase c = MlpCase.random(7, shared, HipKernel.MLP_ALL, 0, 1, tokens - shared.counts().fixed());
    HipLibrary library = HipLibrary.load();
    double tflops;
    float[] millis;
    try (PtStream stream = newStream();
        Measured measured = Measured.open(library, stream, c)) {
      for (int i = 0; i < WARMUP + ITERATIONS; i++) {
        measured.run(i);
      }
      tflops = measured.tflops();
      millis = measured.millis().clone();
    }
    Arrays.sort(millis);
    log.info(
        "K3 {} tokens: median {} ms (min {}, max {}) = {} TFLOPS effective; gate {} TFLOPS: {}",
        tokens,
        String.format("%.3f", millis[ITERATIONS / 2]),
        String.format("%.3f", millis[0]),
        String.format("%.3f", millis[ITERATIONS - 1]),
        String.format("%.1f", tflops),
        String.format("%.0f", GATE_TFLOPS),
        tflops >= GATE_TFLOPS ? "PASS" : "FAIL");
    assertTrue(tflops >= GATE_TFLOPS, "K3 effective " + tflops + " TFLOPS is below the gate " + GATE_TFLOPS);
  }

  /**
   * 型ごとの FFN の K3 の実効の速さが、共有の FFN の K3 の 0.95 倍以上になる。型ごとの K3 は V3 の局・席・牌の層で、FFN の幅は
   * 局・席 1536・牌 768。これに 2048 行分の局・席・牌のトークンを回す。共有の K3 は深さ 3 の形の層 0 で、型ごとの K3 と同じく全ての
   * トークンで x と次の層の Attention の入力を書く。こちらは同じ数のトークンを、本番の幅 512 と、型ごとの区間と同じ幅 1536・768 で回す。
   *
   * <p>同じ幅との比べでは、幅ごとの時間を区間のトークンの割合で重み付けして足し、型ごとの K3 が同じ計算にかける時間とみなす。
   * ワークグループの数がそろうので、最後の半端なワークグループの分もそろう。カーネルは回ごとに順番に回し、クロックの揺れを打ち消す。
   */
  public void typedKeepsTheSharedEfficiency() {
    HipArchitecture typed = MlpCase.typed(TestPack.FOUR_PLAYERS, 1536);
    MlpCase typedCase = MlpCase.random(11, typed, HipKernel.MLP_FIXED, 1, ROWS, 0);
    int tokens = typedCase.fixedTokens();
    int contextWidth = typed.layer(1).ffn().width(TokenGroup.CONTEXT);
    int tileWidth = typed.layer(1).ffn().width(TokenGroup.TILE);
    HipLibrary library = HipLibrary.load();
    List<Measured> measured = new ArrayList<>();
    try (PtStream stream = newStream()) {
      try {
        measured.add(Measured.open(library, stream, typedCase));
        for (int width : new int[] {FFN, contextWidth, tileWidth}) {
          HipArchitecture shared = MlpCase.sharedDepth3(width);
          int events = tokens - shared.counts().fixed();
          measured.add(Measured.open(library, stream, MlpCase.random(12 + width, shared, HipKernel.MLP_ALL, 0, 1, events)));
        }
        for (int i = 0; i < WARMUP + ITERATIONS; i++) {
          for (Measured kernel : measured) {
            kernel.run(i);
          }
        }
      } finally {
        for (Measured kernel : measured) {
          kernel.close();
        }
      }
    }
    Measured typedK3 = measured.get(0);
    Measured production = measured.get(1);
    double share = (double) typedCase.contextTokens() / tokens;
    double matchedMillis = share * measured.get(2).median() + (1 - share) * measured.get(3).median();
    double matched = flops(typedCase) / (matchedMillis * 1e-3) / 1e12;
    double toProduction = typedK3.tflops() / production.tflops();
    double toMatched = typedK3.tflops() / matched;
    log.info(
        "K3 typed FFN {}/{} on {} fixed tokens: median {} ms = {} TFLOPS; shared F {} {} TFLOPS (ratio {}), shared F {}"
            + " {} TFLOPS and F {} {} TFLOPS give {} TFLOPS at the typed widths (ratio {}); gate {}: {}",
        contextWidth,
        tileWidth,
        tokens,
        String.format("%.3f", typedK3.median()),
        String.format("%.1f", typedK3.tflops()),
        FFN,
        String.format("%.1f", production.tflops()),
        String.format("%.3f", toProduction),
        contextWidth,
        String.format("%.1f", measured.get(2).tflops()),
        tileWidth,
        String.format("%.1f", measured.get(3).tflops()),
        String.format("%.1f", matched),
        String.format("%.3f", toMatched),
        TYPED_GATE,
        toProduction >= TYPED_GATE && toMatched >= TYPED_GATE ? "PASS" : "FAIL");
    assertTrue(toProduction >= TYPED_GATE, "typed K3 reaches " + toProduction + " of the production shared K3");
    assertTrue(toMatched >= TYPED_GATE, "typed K3 reaches " + toMatched + " of the shared K3 at the same widths");
  }

  /**
   * カーネルの FLOP 数。mlpStage を回すトークンの分（型ごとの FFN では区間ごとの F で数える）に、イベントのキーとバリューの射影の分を
   * 足す。
   */
  private static double flops(MlpCase c) {
    double h = c.architecture.hidden();
    double a = c.architecture.attention();
    double total = 0;
    for (int t = 0; t < c.processedTokens(); t++) {
      total += 2.0 * (h * a + 2.0 * h * c.ffns.get(c.group(t)).width());
    }
    return total + (double) c.eventKeyValues().size() * c.events * 2.0 * h * 2.0 * a;
  }

  private static PtStream newStream() {
    return ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0));
  }
}
