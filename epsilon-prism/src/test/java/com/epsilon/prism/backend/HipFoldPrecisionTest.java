package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * L0-FOLD の通しの精度を GPU で測る（rocm の組。GPU 1 枚）。網は乱数で動かした深さ 2 の網（模擬の半荘の {@value #FIXTURE_ROWS} 行）と、系の
 * 性質 {@value #SNAPSHOT} があればその学んだ版（{@value #COHORT} の封印済みコホートのラベル行から {@value #ROWS} 行（既定 8192）を等間隔に。
 * なければ模擬の行）。行を {@value #BATCH_ROWS} 行の塊に分け、塊ごとに次の s を出して FP64 の順伝播（{@link ExactReference}）の s と比べる:
 *
 * <ul>
 *   <li>HIP: GPU の順伝播（K1 の L0-FOLD）
 *   <li>A: CPU の参照の畳む前の式（{@link HipReference.Layer0#PROJECTED}。L0-FOLD の前の本番の式の CPU の写し。精度の線の基準）
 *   <li>R: CPU の参照の既定（{@link HipReference.Layer0#FOLDED}。HIP の K1 とビットで同じ qkv0）
 * </ul>
 *
 * <p>記録の行「PRISM-HIP-FOLD &lt;網&gt; &lt;版&gt; vs FP64」に argmax の一致と確信の行の一致（上位 2 手の差は FP64 の側で測る）、s の相対
 * RMS、有限でない値の数を、「HIP vs R」に HIP と R の argmax の一致と s の相対 RMS を出す。HIP の s は有限であることだけを求める（精度の
 * 線は窓の判定が当てる: 学んだ版の HIP の argmax が A − 0.001 以上、確信の行の一致 0.995 以上、s の相対 RMS が A の 1.05 倍以下）。
 */
@Test(groups = "rocm")
public class HipFoldPrecisionTest {
  private static final Logger log = LoggerFactory.getLogger(HipFoldPrecisionTest.class);

  private static final String SNAPSHOT = "prism.snapshot";
  private static final String COHORT = "prism.l0fold.cohort";
  private static final String ROWS = "prism.l0fold.rows";
  private static final int FIXTURE_ROWS = 1024;
  private static final int BATCH_ROWS = 256;
  private static final int DEPTH = 2;
  private static final long SEED = 4300;
  private static final long VERSION = 1;
  private static final String[] RUNS = {"HIP", "A", "R"};

  public void foldAgainstFp64() throws Exception {
    List<long[]> simulated = HipFixtures.simulatedRows(FIXTURE_ROWS, SEED);
    measure("fixture depth " + DEPTH, manager -> HipFixtures.network(manager, DEPTH, SEED + DEPTH), simulated);
    String snapshot = System.getProperty(SNAPSHOT);
    if (snapshot != null) {
      String cohort = System.getProperty(COHORT);
      List<long[]> rows = cohort == null ? simulated : cohortRows(Path.of(cohort), Integer.getInteger(ROWS, 8192));
      Path model = Snapshot.model(Path.of(snapshot));
      measure("snapshot " + snapshot, manager -> PrismNetwork.load(model, manager), rows);
    }
  }

  /** 封印済みのコホートのラベル行から count 行を等間隔に選んだ O。 */
  private static List<long[]> cohortRows(Path directory, int count) throws IOException {
    try (CohortStore store = new CohortStore()) {
      store.addCohort(directory);
      int iteration = store.cohorts().get(0);
      int[] labels = store.select(iteration, iteration, Selection.LABELS);
      List<long[]> rows = new ArrayList<>(count);
      for (int i = 0; i < Math.min(count, labels.length); i++) {
        rows.add(store.read(labels[(int) ((long) i * labels.length / count)]).observation());
      }
      log.info("PRISM-HIP-FOLD cohort {} iteration {}: {} label rows, {} picked", directory, iteration, labels.length,
          rows.size());
      return rows;
    }
  }

  /** 網の作り方。 */
  private interface NetworkSource {
    PrismNetwork open(NDManager manager) throws IOException;
  }

  /** 1 つの網で行 rows の 3 通りの s を FP64 と比べる（記録の行の頭は label）。 */
  private static void measure(String label, NetworkSource source, List<long[]> rows)
      throws IOException, InterruptedException, ExecutionException {
    long started = System.nanoTime();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        PtStream stream = ((PtEngine) Engine.getEngine(PtEngine.ENGINE_NAME)).newStream(Device.gpu(0))) {
      PrismNetwork network = source.open(manager);
      WeightLayout layout = WeightLayout.of(network);
      WeightPack pack = WeightPack.build(layout, network, VERSION);
      List<HostBatch> batches = new ArrayList<>();
      for (int from = 0; from < rows.size(); from += BATCH_ROWS) {
        batches.add(HipFixtures.encode(rows.subList(from, Math.min(rows.size(), from + BATCH_ROWS))));
      }
      List<float[]> hip = new ArrayList<>();
      try (HipBackend backend = HipBackend.open(HipLibrary.load(), stream, layout, GpuCase.capacity(BATCH_ROWS))) {
        backend.publish(pack);
        for (HostBatch batch : batches) {
          float[] s = backend.forward(batch, VERSION);
          for (float value : s) {
            Assert.assertTrue(Float.isFinite(value), label + " HIP score " + value);
          }
          hip.add(s);
        }
      }
      Tally[] tallies = {new Tally(), new Tally(), new Tally()};
      Tally versusR = new Tally();
      ThreadLocal<HipReference> references = ThreadLocal.withInitial(() -> new HipReference(pack));
      ExecutorService executor = Executors.newFixedThreadPool(Math.max(1, Runtime.getRuntime().availableProcessors()));
      try {
        List<Future<?>> futures = new ArrayList<>();
        for (int b = 0; b < batches.size(); b++) {
          HostBatch batch = batches.get(b);
          float[] gpu = hip.get(b);
          futures.add(executor.submit(() -> {
            HipReference reference = references.get();
            HipParity.Actions actions = HipParity.Actions.of(batch);
            double[] exact = new ExactReference(reference).forward(batch);
            float[] expected = new float[exact.length];
            for (int a = 0; a < exact.length; a++) {
              expected[a] = (float) exact[a];
            }
            reference.layer0 = HipReference.Layer0.PROJECTED;
            float[] projected = reference.forward(batch);
            reference.layer0 = HipReference.Layer0.FOLDED;
            float[] folded = reference.forward(batch);
            float[][] runs = {gpu, projected, folded};
            for (int v = 0; v < runs.length; v++) {
              tallies[v].add(runs[v], expected, exact, actions);
            }
            versusR.add(gpu, folded, null, actions);
            return null;
          }));
        }
        for (Future<?> future : futures) {
          future.get();
        }
      } finally {
        executor.shutdownNow();
      }
      for (int v = 0; v < RUNS.length; v++) {
        Tally t = tallies[v];
        log.info(
            "PRISM-HIP-FOLD {} {} vs FP64: argmax {} confident {} (rows {}, confident {}); s rel RMS {}; non-finite {}",
            label,
            RUNS[v],
            f5(t.ratio(t.agree, t.rows)),
            f5(t.ratio(t.confidentAgree, t.confident)),
            t.rows,
            t.confident,
            String.format(Locale.ROOT, "%.4e", t.relativeRms()),
            t.nonFinite);
      }
      log.info(
          "PRISM-HIP-FOLD {} HIP vs R: argmax {} (rows {}); s rel RMS {}",
          label,
          f5(versusR.ratio(versusR.agree, versusR.rows)),
          versusR.rows,
          String.format(Locale.ROOT, "%.4e", versusR.relativeRms()));
      log.info("PRISM-HIP-FOLD {} done: {} rows in {} s", label, rows.size(),
          String.format(Locale.ROOT, "%.1f", (System.nanoTime() - started) / 1e9));
    }
  }

  /** 1 つの版の集計（塊をまたいで足す）。 */
  private static final class Tally {
    double error;
    double norm;
    long nonFinite;
    long rows;
    long agree;
    long confident;
    long confidentAgree;

    /** expected と比べる（exact があれば相対 RMS はそれと、なければ expected と）。 */
    synchronized void add(float[] actual, float[] expected, double[] exact, HipParity.Actions actions) {
      for (int i = 0; i < actual.length; i++) {
        if (!Float.isFinite(actual[i])) {
          nonFinite++;
          continue;
        }
        double e = exact == null ? expected[i] : exact[i];
        double d = actual[i] - e;
        error += d * d;
        norm += e * e;
      }
      HipParity.Report report = HipParity.compare(actual, expected, actions);
      rows += report.rows();
      agree += Math.round(report.argmaxAgreement() * report.rows());
      confident += report.confidentRows();
      confidentAgree += Math.round(report.confidentAgreement() * report.confidentRows());
    }

    double relativeRms() {
      return Math.sqrt(error / norm);
    }

    double ratio(long numerator, long denominator) {
      return denominator == 0 ? 1 : (double) numerator / denominator;
    }
  }

  private static String f5(double value) {
    return String.format(Locale.ROOT, "%.5f", value);
  }
}
