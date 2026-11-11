package com.epsilon.prism.model;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.ThreePlayerRows;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism3p.input.InputSchema;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 三人打ちの GPU での学習の経路を確かめる（rocm のグループ。GPU 1 枚）。行はエンジンで打った三人打ちの判断（全ての手の種類と、公開の
 * イベントが多い行を含む）。
 *
 * <ul>
 *   <li>関係のコード: GPU の学習で使う自前の演算（{@link RelationCodes}、T0）の結果が、eager の {@link RelationClass} とパディングの位置も
 *       含めた全てのワードでビット単位で同じ。席の区分のコードは三人打ちの式の 1〜9 が全て出て、10 以上は出ない。
 *       {@code -Dprism.bench.rows=<rows/t>} があれば、三人打ちのコホートの実際の行を本番と同じ詰め方でバッチにしたものも比べる。
 *   <li>Attention の記録: BF16 の HIP の経路（T0・T1・T2）の損失・s・全ての勾配を、djl-rocm の演算の経路、および CPU の参照の Attention
 *       （{@link ReferenceAttention}）を差し込んだ経路と比べる。v2 の深さ 2 と型ごとの FFN の 3 つのアームを、ネットワークと行の種を変えて
 *       {@value #SEEDS} 回ずつ実行し、差の平均をしきい値と比べる: 損失 ≤ 1e-3、s ≤ 1e-2、勾配の全体 ≤ 2e-2（2 つの BF16 の経路の差は丸めの
 *       ずれによる揺れで、1 つのネットワークでは 2e-2 を超えることもある）。FP32 の djl の経路を真の値として、HIP の経路の s のずれの平均が
 *       djl の経路の 1.25 倍以内。どの種でも、どの経路でも、勾配がちょうど 0 のパラメーターがない。
 * </ul>
 */
@Test(groups = "rocm")
public class ThreePlayerAttentionTapeGpuTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerAttentionTapeGpuTest.class);

  /** 実際の行のディレクトリを渡すシステムプロパティ（FitStepGpuBench と同じ）。 */
  private static final String ROWS_PROPERTY = "prism.bench.rows";

  /** 実際の行で比べるときの、1 ステップの行の数とステップの数。 */
  private static final int STEP_ROWS = 4096;
  private static final int STEPS = 2;

  /** バッチに入れる、ふつうの判断の行の数。 */
  private static final int ROWS = 96;

  /** バッチに入れる、公開のイベントが多い行の数。 */
  private static final int LONG_ROWS = 8;

  /** アームごとに変える、ネットワークと行の種の数。 */
  private static final int SEEDS = 4;

  /** 三人打ちの席の区分のコードの上限（クエリとキーの両方に持ち主の席がある: 1〜3、キーだけ: 4〜6、クエリだけ: 7〜9）。 */
  private static final int SEAT_CODES = 3 * InputSchema.FORMAT.seats();

  private List<long[]> longRows;

  @BeforeClass
  public void rows() {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    longRows = ThreePlayerRows.longRows(LONG_ROWS);
  }

  /** 種 {@code seed} のバッチ（ふつうの行の後に、公開のイベントが多い行を足す）。 */
  private HostBatch batch(int seed) {
    List<long[]> rows = new ArrayList<>(ThreePlayerRows.mixedRows(ROWS, 900 + 13L * seed));
    rows.addAll(longRows);
    return ThreePlayerRows.encode(rows);
  }

  public void relationCodesMatchEager() throws IOException {
    Device gpu = Device.gpu(0);
    long[] seat = new long[16];
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      HipAttention kernel = HipAttention.on(HipLibrary.load(), stream);
      HostBatch engine = batch(0);
      compare(manager, kernel, "engine", engine, seat);
      String directory = System.getProperty(ROWS_PROPERTY);
      if (directory != null) {
        for (List<long[]> piece : pieces(Path.of(directory))) {
          compare(manager, kernel, "rows", ThreePlayerRows.encode(piece), seat);
        }
      }
    }
    log.info("PRISM-3P-T0 seat segment codes {}", Arrays.toString(seat));
    for (int code = 1; code <= SEAT_CODES; code++) {
      Assert.assertTrue(seat[code] > 0, "seat code " + code + " never appears");
    }
    for (int code = SEAT_CODES + 1; code < seat.length; code++) {
      Assert.assertEquals(seat[code], 0L, "seat code " + code + " is outside the three-player formula");
    }
  }

  /** バッチのコードを eager と自前の演算で作り、全てのワードを照合する（違えば最初に違う位置をログに出す）。席の区分のコードも数える。 */
  private static void compare(NDManager manager, RelationCodes kernel, String name, HostBatch batch, long[] seat) {
    try (NDManager scope = manager.newSubManager()) {
      NetworkInputs inputs = NetworkInputs.of(scope, PinnedUploads.on(scope), batch);
      BatchIndex index = BatchIndex.send(inputs, Pass.inference(Precision.BFLOAT16));
      ShortBuffer expected = words(RelationClass.codes(inputs, index));
      ShortBuffer actual = words(RelationClass.codes(kernel, inputs, index));
      Assert.assertEquals(actual.limit(), expected.limit(), "code words");
      int mismatched = 0;
      int first = -1;
      for (int w = 0; w < expected.limit(); w++) {
        if (expected.get(w) != actual.get(w)) {
          mismatched++;
          first = first < 0 ? w : first;
        }
        if (w % RelationCodes.WORDS == 0) {
          seat[((expected.get(w) & 0xffff) >> 12) & 15]++;
        }
      }
      int size = index.tokens;
      log.info(
          "PRISM-3P-T0 {} rows={} S={} words={} mismatched={}", name, inputs.rows(), size, expected.limit(), mismatched);
      if (first >= 0) {
        int pair = first / RelationCodes.WORDS;
        log.info(
            "PRISM-3P-T0   first row={} i={} j={} word={} eager={} kernel={}",
            pair / (size * size),
            pair / size % size,
            pair % size,
            first % RelationCodes.WORDS,
            String.format("%04x", expected.get(first) & 0xffff),
            String.format("%04x", actual.get(first) & 0xffff));
      }
      Assert.assertEquals(mismatched, 0, name + " codes differ");
    }
  }

  /** コードの INT16 のワード（ホストへ転送する）。 */
  private static ShortBuffer words(NDArray codes) {
    return codes.toByteBuffer().order(ByteOrder.nativeOrder()).asShortBuffer();
  }

  /** 実際の行をステップごとに取り、トークンの数の順に並べて、トークンの上限まで詰めたバッチ（O のワードの並び）。 */
  private static List<List<long[]>> pieces(Path directory) throws IOException {
    List<List<long[]>> pieces = new ArrayList<>();
    Observation observation = new Observation(TENHOU_3P);
    try (CohortStore store = new CohortStore()) {
      store.addCohort(directory);
      Assert.assertEquals(store.variant(), TENHOU_3P, "the rows must be three-player rows");
      int cohort = store.cohorts().get(0);
      int[] rows = Fit.shuffle(store.select(cohort, cohort, Selection.LABELS), 5);
      for (int step = 0; step < STEPS && (step + 1) * STEP_ROWS <= rows.length; step++) {
        List<long[]> sorted = new ArrayList<>(STEP_ROWS);
        for (int r = step * STEP_ROWS; r < (step + 1) * STEP_ROWS; r++) {
          sorted.add(store.read(rows[r]).observation());
        }
        sorted.sort(Comparator.comparingInt(words -> observation.bind(words, 0).publicEvents()));
        List<long[]> piece = new ArrayList<>();
        for (long[] words : sorted) {
          int tokens = InputSchema.FORMAT.fixedTokens() + observation.bind(words, 0).publicEvents();
          if (!piece.isEmpty() && (long) (piece.size() + 1) * tokens > FitPlan.TOKEN_BUDGET) {
            pieces.add(piece);
            piece = new ArrayList<>();
          }
          piece.add(words);
        }
        pieces.add(piece);
      }
    }
    return pieces;
  }

  @DataProvider
  public Object[][] arms() {
    return new Object[][] {{ShapeArm.V2}, {ShapeArm.V3}, {ShapeArm.V3_ONE_STAGE}, {ShapeArm.V3_CONTEXT_512}};
  }

  @Test(dataProvider = "arms")
  public void tapeMatchesTheDjlOperatorAndTheReference(ShapeArm arm) {
    PrismShape shape = arm.shape(arm == ShapeArm.V2 ? PrismShape.DEFAULT_DEPTH : ShapeArm.V3_DEPTH);
    double[][] djl = new double[3][SEEDS];
    double[][] host = new double[3][SEEDS];
    double[] djlRounding = new double[SEEDS];
    double[] hipRounding = new double[SEEDS];
    List<String> silent = new ArrayList<>();
    Device gpu = Device.gpu(0);
    for (int seed = 0; seed < SEEDS; seed++) {
      HostBatch batch = batch(seed);
      try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
          PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
          PtStreamScope ignored = stream.openScope()) {
        PrismNetwork network = ThreePlayerRows.network(manager, shape, 3 + seed);
        HipAttention attention = HipAttention.on(HipLibrary.load(), stream);
        ThreePlayerSteps.Outcome truth = run(network, manager, batch, null, Precision.FLOAT32);
        ThreePlayerSteps.Outcome eager = run(network, manager, batch, null, Precision.BFLOAT16);
        ThreePlayerSteps.Outcome reference = run(network, manager, batch, new ReferenceAttention(), Precision.BFLOAT16);
        ThreePlayerSteps.Outcome hip = run(network, manager, batch, attention, Precision.BFLOAT16);
        double hipMillis = run(network, manager, batch, attention, Precision.BFLOAT16).millis();
        double djlMillis = run(network, manager, batch, null, Precision.BFLOAT16).millis();
        put(djl, seed, eager, hip);
        put(host, seed, reference, hip);
        djlRounding[seed] = ThreePlayerSteps.relative(truth.scores(), eager.scores());
        hipRounding[seed] = ThreePlayerSteps.relative(truth.scores(), hip.scores());
        for (ThreePlayerSteps.Outcome outcome : List.of(eager, reference, hip)) {
          List<String> names = ThreePlayerSteps.silent(outcome);
          if (!names.isEmpty()) {
            silent.add("seed " + seed + (outcome == hip ? " HIP " : outcome == eager ? " djl " : " reference ") + names);
          }
        }
        log.info(
            "PRISM-3P-TAPE {} seed={} rows={} loss fp32={} djl={} reference={} hip={} vs djl: loss rel={} s rel={}"
                + " gradients rel={} vs reference: loss rel={} s rel={} gradients rel={} from FP32 s djl={} hip={}"
                + " ms/step djl={} hip={}",
            arm,
            seed,
            batch.plan().rows(),
            String.format("%.6f", truth.loss()),
            String.format("%.6f", eager.loss()),
            String.format("%.6f", reference.loss()),
            String.format("%.6f", hip.loss()),
            String.format("%.3g", djl[0][seed]),
            String.format("%.4f", djl[1][seed]),
            String.format("%.4f", djl[2][seed]),
            String.format("%.3g", host[0][seed]),
            String.format("%.4f", host[1][seed]),
            String.format("%.4f", host[2][seed]),
            String.format("%.4f", djlRounding[seed]),
            String.format("%.4f", hipRounding[seed]),
            String.format("%.1f", djlMillis),
            String.format("%.1f", hipMillis));
      }
    }
    log.info(
        "PRISM-3P-TAPE-SUMMARY {} seeds={} vs djl mean loss rel={} s rel={} (max {}) gradients rel={} (max {})"
            + " vs reference mean loss rel={} s rel={} (max {}) gradients rel={} (max {}) from FP32 s djl={} hip={}",
        arm,
        SEEDS,
        String.format("%.3g", mean(djl[0])),
        String.format("%.4f", mean(djl[1])),
        String.format("%.4f", max(djl[1])),
        String.format("%.4f", mean(djl[2])),
        String.format("%.4f", max(djl[2])),
        String.format("%.3g", mean(host[0])),
        String.format("%.4f", mean(host[1])),
        String.format("%.4f", max(host[1])),
        String.format("%.4f", mean(host[2])),
        String.format("%.4f", max(host[2])),
        String.format("%.4f", mean(djlRounding)),
        String.format("%.4f", mean(hipRounding)));
    Assert.assertEquals(silent, List.of(), arm + " gradients that are exactly zero");
    for (double[][] against : List.of(djl, host)) {
      String name = against == djl ? " vs djl" : " vs reference";
      Assert.assertTrue(mean(against[0]) <= 1e-3, arm + name + " mean loss rel " + mean(against[0]));
      Assert.assertTrue(mean(against[1]) <= 1e-2, arm + name + " mean scores rel " + mean(against[1]));
      Assert.assertTrue(mean(against[2]) <= 2e-2, arm + name + " mean gradients rel " + mean(against[2]));
    }
    Assert.assertTrue(
        mean(hipRounding) <= 1.25 * mean(djlRounding),
        "the HIP path is further from FP32 (" + mean(hipRounding) + ") than djl (" + mean(djlRounding) + ")");
  }

  private static ThreePlayerSteps.Outcome run(
      PrismNetwork network,
      NDManager manager,
      HostBatch batch,
      TrainingKernels kernels,
      Precision precision) {
    return ThreePlayerSteps.run(network, manager, batch, kernels, precision, Context.Intermediates.CLOSE);
  }

  /** 種 {@code seed} での、損失・s・勾配の相対の差を書く。 */
  private static void put(
      double[][] errors, int seed, ThreePlayerSteps.Outcome reference, ThreePlayerSteps.Outcome other) {
    errors[0][seed] = Math.abs(other.loss() - reference.loss()) / Math.abs(reference.loss());
    errors[1][seed] = ThreePlayerSteps.relative(reference.scores(), other.scores());
    errors[2][seed] = ThreePlayerSteps.gradients(reference, other);
  }

  private static double mean(double[] values) {
    return Arrays.stream(values).average().orElse(0);
  }

  private static double max(double[] values) {
    return Arrays.stream(values).max().orElse(0);
  }
}
