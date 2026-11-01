package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import com.epsilon.prism.model.Batches.Batch;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * CPU の 1 歩の時間の測り（通常の試験には入れず、サーバで {@code -Dtest=PrismStepBench} で回す）。標準の形（H 256・F 512・G 256）、
 * 出来事 35 個（S = 74）の行で、順伝播・損失（差分整合＋補助）・逆伝播・Adam の 1 歩を測り、行あたりの時間も出す。
 *
 * <p>CPU の packedRelationScaledDotProductAttention は参照の実装で [行, S, S, 頭 + A] を区画ごとに作るので、メモリを守るため行の数を
 * 小さくして測り、256 行の時間は行に比例するとみなして出す。
 */
public class PrismStepBench {
  private static final Logger log = LoggerFactory.getLogger(PrismStepBench.class);

  private static final int ROWS = 32;
  private static final int EVENTS = 35;
  private static final int WARMUP = 1;
  private static final int MEASURED = 3;

  @Test
  public void measureCpuStep() {
    for (int depth = PrismShape.MIN_DEPTH; depth <= PrismShape.MAX_DEPTH; depth++) {
      measure(depth);
    }
  }

  private static void measure(int depth) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(7);
      PrismNetwork network = new PrismNetwork(PrismShape.standard(depth));
      network.initialize(manager);
      Batches.perturb(network, 0.02f, 11);
      ParameterStore store = new ParameterStore(manager, false);
      List<Batches.Row> rows = new ArrayList<>();
      for (int r = 0; r < ROWS; r++) {
        rows.add(
            switch (r % 8) {
              case 0, 1, 2, 3 -> Batches.turn(EVENTS);
              case 4, 5 -> Batches.response(EVENTS);
              case 6 -> Batches.chankan(EVENTS);
              default -> Batches.postCall(EVENTS);
            });
      }
      Batch batch = Batches.build(rows, 13);
      List<ValueLabel> labels = Batches.values(batch, 17);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 19);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-4f);
      long trainNanos = 0;
      long inferNanos = 0;
      for (int step = 0; step < WARMUP + MEASURED; step++) {
        long start = System.nanoTime();
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              network.forward(
                  store, batch.inputs(stepScope), Pass.training(Precision.FLOAT32, measured, auxiliary));
          collector.backward(loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f));
        }
        Batches.step(network, optimizer);
        long trained = System.nanoTime();
        try (NDManager stepScope = scope.newSubManager()) {
          network
              .forward(store, batch.inputs(stepScope), Pass.inference(Precision.FLOAT32))
              .scores()
              .toFloatArray();
        }
        long inferred = System.nanoTime();
        if (step >= WARMUP) {
          trainNanos += trained - start;
          inferNanos += inferred - trained;
        }
      }
      double trainMs = trainNanos / 1e6 / MEASURED;
      double inferMs = inferNanos / 1e6 / MEASURED;
      log.info(
          "PRISM-STEP depth={} rows={} tokens={} transitions={} train_ms={} ({} ms/row, 256 rows ~{} ms)"
              + " infer_ms={} ({} ms/row)",
          depth,
          ROWS,
          TokenType.FIXED_TOKENS + EVENTS,
          batch.transitions(),
          String.format("%.1f", trainMs),
          String.format("%.2f", trainMs / ROWS),
          String.format("%.0f", trainMs / ROWS * 256),
          String.format("%.1f", inferMs),
          String.format("%.2f", inferMs / ROWS));
    }
  }
}
