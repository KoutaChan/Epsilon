package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.run.RunFiles;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 本物のコホートの錨の s の照合（rocm の組。GPU 2 枚。-Dprism.run と -Dprism.cohort がなければ飛ばす）。run のコホート t の生徒の全ての行
 * （{@link Selection#LABELS_AND_ROOTS}）で、先に出した s（{@link AnchorScores#compute}、2 枚）と、当てはめの並べ替えの順に詰めた塊を
 * HIP の錨（{@link Anchor.Hip}、1 枚）に通した s が、全ての手でビットで同じ。錨の網は版 t の学習の網。
 */
@Test(groups = "rocm")
public class AnchorScoresGpuCheck {
  private static final Logger log = LoggerFactory.getLogger(AnchorScoresGpuCheck.class);

  private static final int THREADS = 16;

  public void precomputedScoresMatchTheFitAnchorBitwise() throws IOException {
    String run = System.getProperty("prism.run");
    String cohort = System.getProperty("prism.cohort");
    if (run == null || cohort == null) {
      throw new SkipException("needs -Dprism.run and -Dprism.cohort");
    }
    int t = Integer.parseInt(cohort);
    Path network = Snapshot.iterate(RunFiles.snapshot(Path.of(run), t));
    try (CohortStore store = new CohortStore()) {
      store.addCohort(CohortWriter.directory(Path.of(run), t));
      int[] rows = store.select(t, t, Selection.LABELS_AND_ROOTS);
      long started = System.nanoTime();
      AnchorScores scores =
          AnchorScores.compute(
              store,
              rows,
              network,
              InferencePath.HIP,
              NetworkDevices.of(Device.gpu(0), Device.gpu(1)),
              THREADS);
      double precomputed = (System.nanoTime() - started) / 1e9;
      scores.requireNetwork(network);
      BatchPlan.Capacity capacity = StepFeed.capacity(FitPlan.TOKEN_BUDGET);
      int[] order = Fit.shuffle(rows, 11);
      long compared = 0;
      long mismatched = 0;
      Device gpu = Device.gpu(0);
      PtEngine engine = (PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME);
      try (PtNDManager pinned =
              (PtNDManager) NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME);
          PtStream stream = engine.newStream(gpu);
          Anchor anchor = Anchor.open(network, InferencePath.HIP, gpu, stream, null, capacity);
          ExecutorService readers = Executors.newFixedThreadPool(THREADS)) {
        HostInputs batch =
            HostInputs.allocate(
                capacity, false, bytes -> pinned.allocatePinned(bytes, DataType.UINT8).getByteBuffer());
        Encoder.Inputs inputs = new Encoder.Inputs(false);
        for (int from = 0; from < order.length; from += 4096) {
          int[] ids = Arrays.copyOfRange(order, from, Math.min(order.length, from + 4096));
          Row[] read = store.read(ids, readers, THREADS);
          int first = 0;
          while (first < read.length) {
            BatchPlan plan = batch.plan();
            plan.clear();
            int end = first;
            while (end < read.length && plan.add(inputs.bind(read[end])) >= 0) {
              end++;
            }
            plan.seal();
            for (int row = first; row < end; row++) {
              inputs.write(batch, row - first, read[row]);
            }
            Anchor.Scores reference = anchor.scores(batch);
            for (int row = first; row < end; row++) {
              float[] own = scores.scores(ids[row]);
              int at = reference.starts()[row - first];
              for (int a = 0; a < own.length; a++) {
                if (Float.floatToRawIntBits(own[a])
                    != Float.floatToRawIntBits(reference.values()[at + a])) {
                  mismatched++;
                }
                compared++;
              }
            }
            first = end;
          }
        }
      }
      log.info(
          "PRISM-ANCHOR-CHECK cohort={} rows={} actions={} mismatched={} precomputed {}s",
          t,
          rows.length,
          compared,
          mismatched,
          String.format(Locale.ROOT, "%.1f", precomputed));
      assertEquals(mismatched, 0, "anchor scores differ from the fit anchor");
    }
  }
}
