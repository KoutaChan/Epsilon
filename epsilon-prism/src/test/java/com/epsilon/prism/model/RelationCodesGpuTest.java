package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationCodes;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.observe.Observation;
import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GPU の学習の関係の符号（{@link RelationCodes}、{@code t0_relation.hip}）が、eager の {@link RelationClass} と詰め物の位置を含む全ての
 * 語でビットで同じこと（rocm の組。GPU 1 枚）。乱数の属性の塊（S は 156 まで）、エンジンで打った半荘の塊、
 * {@code -Dprism.bench.rows=<rows-v2/t>} があれば本物の行の微小バッチ（歩の行を本番と同じく S の順に並べ、トークンの上限
 * {@link FitPlan#TOKEN_BUDGET} で詰めた塊。歩の最後の塊が S の最も大きい塊）を比べる。
 */
public class RelationCodesGpuTest {
  private static final Logger log = LoggerFactory.getLogger(RelationCodesGpuTest.class);

  /** 本物の行のディレクトリを渡す系の性質（FitStepGpuBench と同じ）。 */
  private static final String ROWS_PROPERTY = "prism.bench.rows";

  /** 本物の行の 1 歩の行の数と、比べる歩の数。 */
  private static final int STEP_ROWS = 4096;
  private static final int STEPS = 4;

  /** 乱数の塊とエンジンの塊の行の数。 */
  private static final int ROWS = 96;

  /** 乱数の塊の行の出来事の数（最も長い行は S = 39 + 117）。 */
  private static final int[] EVENTS = {0, 12, 35, 60, 90, 117};

  /** 行の O の語と、そのトークンの数 S（局・席・牌の 39 ＋ 公開の出来事）。 */
  private record Row(int tokens, long[] words) {}

  @Test(groups = "rocm")
  public void kernelMatchesTheEagerCodes() throws IOException {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Device gpu = Device.gpu(0);
    try (NDManager manager = NDManager.newBaseManager(gpu, "PyTorch");
        PtStream stream = ((PtEngine) manager.getEngine()).newStream(gpu);
        PtStreamScope ignored = stream.openScope()) {
      HipAttention kernel = HipAttention.on(HipLibrary.load(), stream);
      Batches.Batch random = Batches.build(randomRows(), 5);
      compare(manager, kernel, "random", random::inputs);
      List<long[]> engine = HipFixtures.mixedRows(ROWS, 900);
      compare(
          manager,
          kernel,
          "engine",
          scope -> NetworkInputs.of(scope, Uploads.on(scope), HipFixtures.encode(engine)));
      String directory = System.getProperty(ROWS_PROPERTY);
      if (directory != null) {
        for (List<long[]> piece : pieces(Path.of(directory))) {
          compare(
              manager,
              kernel,
              "rows",
              scope -> NetworkInputs.of(scope, Uploads.on(scope), HipFixtures.encode(piece)));
        }
      }
    }
  }

  /** 塊の符号を eager と自前の演算で作り、全ての語を照らす（違えば最初の違いの位置を出す）。 */
  private static void compare(
      NDManager manager, RelationCodes kernel, String name, Function<NDManager, NetworkInputs> batch) {
    try (NDManager scope = manager.newSubManager()) {
      NetworkInputs inputs = batch.apply(scope);
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
      }
      int size = index.tokens;
      log.info(
          "PRISM-T0 {} rows={} S={} words={} mismatched={}", name, inputs.rows(), size, expected.limit(), mismatched);
      if (first >= 0) {
        int pair = first / RelationCodes.WORDS;
        log.info(
            "PRISM-T0   first row={} i={} j={} word={} eager={} kernel={}",
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

  /** 符号の INT16 の語（ホストへ写す）。 */
  private static ShortBuffer words(NDArray codes) {
    return codes.toByteBuffer().order(ByteOrder.nativeOrder()).asShortBuffer();
  }

  /** 乱数の属性の行（手番・応答・鳴いた後を、出来事の数を変えて）。 */
  private static List<Batches.Row> randomRows() {
    List<Batches.Row> rows = new ArrayList<>();
    for (int r = 0; r < ROWS; r++) {
      int events = EVENTS[r % EVENTS.length];
      rows.add(
          switch (r % 4) {
            case 0, 1 -> Batches.turn(events);
            case 2 -> Batches.response(events);
            default -> Batches.postCall(events);
          });
    }
    return rows;
  }

  /** 本物の行の歩ごとに、行を S の順に並べてトークンの上限で詰めた塊（O の語の列）。 */
  private static List<List<long[]>> pieces(Path directory) throws IOException {
    List<List<long[]>> pieces = new ArrayList<>();
    Observation observation = new Observation();
    try (CohortStore store = new CohortStore()) {
      store.addCohort(directory);
      int cohort = store.cohorts().get(0);
      int[] rows = Fit.shuffle(store.select(cohort, cohort, Selection.LABELS), 5);
      for (int step = 0; step < STEPS && (step + 1) * STEP_ROWS <= rows.length; step++) {
        List<Row> sorted = new ArrayList<>(STEP_ROWS);
        for (int r = step * STEP_ROWS; r < (step + 1) * STEP_ROWS; r++) {
          long[] words = store.read(rows[r]).observation();
          sorted.add(new Row(BatchPlan.FIXED_TOKENS + observation.bind(words, 0).publicEvents(), words));
        }
        sorted.sort(Comparator.comparingInt(Row::tokens));
        List<long[]> piece = new ArrayList<>();
        for (Row row : sorted) {
          if (!piece.isEmpty() && (long) (piece.size() + 1) * row.tokens() > FitPlan.TOKEN_BUDGET) {
            pieces.add(piece);
            piece = new ArrayList<>();
          }
          piece.add(row.words());
        }
        pieces.add(piece);
      }
    }
    return pieces;
  }
}
