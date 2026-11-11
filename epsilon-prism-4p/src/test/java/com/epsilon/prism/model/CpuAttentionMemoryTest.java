package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism4p.input.InputSchema;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * CPU の順伝播と逆伝播のメモリを確かめる。イベントの多い行のバッチを、系譜の始めの形（深さ 2）のネットワークに CPU で通す。RSS のピークの
 * 増え（/proc/self の VmHWM）を (行, クエリ, キー) の組の数で割った値が {@link #BYTES_PER_PAIR} に収まることを、推論と学習で確かめる
 * （Linux だけ）。djl-rocm の CPU の packed 演算は行のまとまりごとに計算するので、Attention が組ごとに持つのはスコアの数個分で済む。
 * 組ごとに表の行と相対のバリューを持つ計算では、この上限を数倍超える。
 */
public class CpuAttentionMemoryTest {
  private static final Logger log = LoggerFactory.getLogger(CpuAttentionMemoryTest.class);

  private static final int ROWS = 16;
  private static final int EVENTS = 100;

  /** 組あたりの RSS のピークの増えの上限（バイト）。 */
  private static final long BYTES_PER_PAIR = 2500;

  private static final Path CLEAR_REFS = Path.of("/proc/self/clear_refs");
  private static final Path STATUS = Path.of("/proc/self/status");

  @Test
  public void peakMemoryPerPairStaysWithinAFewScores() {
    if (!Files.isWritable(CLEAR_REFS)) {
      throw new SkipException("peak RSS needs " + CLEAR_REFS);
    }
    PrismShape shape = PrismShape.standard(2);
    Batch batch = Batches.build(Collections.nCopies(ROWS, Batches.turn(EVENTS)), 7);
    List<ValueLabel> labels = Batches.values(batch, 11);
    boolean[] measured = ValueLabel.measured(labels);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    long tokens = InputSchema.FORMAT.fixedTokens() + EVENTS;
    long pairs = ROWS * tokens * tokens;
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(5);
      PrismNetwork network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      ParameterStore store = new ParameterStore(manager, false);
      Runnable inference =
          () -> {
            try (NDManager scope = manager.newSubManager();
                InferenceMode mode = manager.getEngine().newInferenceMode()) {
              network
                  .forward(store, batch.inputs(scope), Pass.inference(Precision.FLOAT32))
                  .scores()
                  .toFloatArray();
            }
          };
      Runnable training =
          () -> {
            try (NDManager scope = manager.newSubManager();
                GradientCollector collector = manager.getEngine().newGradientCollector()) {
              Outputs outputs =
                  network.forward(
                      store, batch.inputs(scope), Pass.training(Precision.FLOAT32, measured, null));
              collector.backward(loss.total(scope, outputs, labels, null, 1f, labels.size(), 1f));
            }
          };
      inference.run();
      training.run();
      long inferencePeak = peak(inference);
      long trainingPeak = peak(training);
      log.info(
          "CPU attention peak RSS per pair: inference {} B, training {} B ({} pairs)",
          inferencePeak / pairs,
          trainingPeak / pairs,
          pairs);
      Assert.assertTrue(inferencePeak < BYTES_PER_PAIR * pairs, "inference peak " + inferencePeak + " B");
      Assert.assertTrue(trainingPeak < BYTES_PER_PAIR * pairs, "training peak " + trainingPeak + " B");
    }
  }

  /** ピークを今の RSS に戻してから計算し、RSS のピークの増えを返す（バイト）。 */
  private static long peak(Runnable pass) {
    try {
      Files.writeString(CLEAR_REFS, "5");
      long before = status("VmRSS");
      pass.run();
      return status("VmHWM") - before;
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  private static long status(String field) throws IOException {
    for (String line : Files.readAllLines(STATUS)) {
      if (line.startsWith(field + ":")) {
        return Long.parseLong(line.replaceAll("[^0-9]", "")) * 1024;
      }
    }
    throw new IllegalStateException("no " + field + " in " + STATUS);
  }
}
