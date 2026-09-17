package com.epsilon.pico.ai.decision.input;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 実在する遷移だけを保存した推論入力が、固定長の学習入力と一致し、複製・転送・作業領域の再利用後も正しく参照できることを検証する。 */
public class DecisionCompactTransitionTest {
  private static final DecisionBucket BUCKET = new DecisionBucket(4, 16);
  private static final int PREFIX = 3;
  private static final int SUFFIX = 2;
  private static final short DIRTY_CATEGORY = -17;
  private static final float DIRTY_NUMERIC = -19.25f;

  @DataProvider
  public Object[][] bufferKinds() {
    return new Object[][] {{false}, {true}};
  }

  @DataProvider
  public Object[][] inferenceSourceKinds() {
    return new Object[][] {{false, false}, {false, true}, {true, false}, {true, true}};
  }

  @DataProvider
  public Object[][] singleTransitionCases() {
    return new Object[][] {
      {1, false}, {1, true}, {4, false}, {4, true}, {16, false}, {16, true}, {40, false}, {40, true}
    };
  }

  @Test(dataProvider = "singleTransitionCases")
  public void singleTransitionsKeepCanonicalPaddingAndInferenceInputs(int actions, boolean direct) {
    DecisionBucket bucket = new DecisionBucket(actions, 1);
    List<Scenario> source = scenarios();
    List<Scenario> rows = List.of(source.get(0), source.get(3), source.get(0), source.get(3));
    if (actions == 1) {
      rows =
          rows.stream()
              .map(row -> new Scenario(row.state, List.of(row.actions.getLast())))
              .toList();
    }
    DecisionHostBatch actual =
        inference(rows, new DecisionBatchBuilder.EncodingWorkspacePool(1), bucket);
    DecisionHostBatch expected = training(rows, bucket);
    for (int row = 0; row < rows.size(); row++) {
      for (int action = 0; action < actions; action++) {
        Assert.assertEquals(
            actual.actionTransitionCount(row, action),
            action < rows.get(row).actions.size() ? 1 : 0);
      }
    }
    assertCanonical(actual.sliceRows(0, 4), expected.sliceRows(0, 4), direct);
    assertCanonical(actual.sliceRows(1, 2), expected.sliceRows(1, 2), direct);
    assertCanonical(actual.copyRows(1, 2).sliceRows(0, 2), expected.sliceRows(1, 2), direct);
    int[] selection = {3, 0, 3};
    assertCanonical(
        actual.selectRows(selection, 3).sliceRows(0, 3),
        expected.selectRows(selection, 3).sliceRows(0, 3),
        direct);
    DecisionHostBatch.RowBatch actualRows =
        DecisionHostBatch.RowBatch.of(List.of(actual.sliceRows(2, 2), actual.sliceRows(0, 2)));
    DecisionHostBatch.RowBatch expectedRows =
        DecisionHostBatch.RowBatch.of(List.of(expected.sliceRows(2, 2), expected.sliceRows(0, 2)));
    var indices = actualRows.inferenceIndexLayout(true);
    var layout = actualRows.inferenceInputLayout(indices);
    Assert.assertEquals(indices, expectedRows.inferenceIndexLayout(true));
    short[] expectedCategories = new short[layout.categoricalElementCount()];
    float[] expectedNumerics = new float[layout.numericElementCount()];
    expectedRows.copyInferenceInputsTo(
        ShortBuffer.wrap(expectedCategories), FloatBuffer.wrap(expectedNumerics), layout);
    ShortBuffer categories = categoryBuffer(expectedCategories.length, direct);
    FloatBuffer numerics = numericBuffer(expectedNumerics.length, direct);
    for (int repeat = 0; repeat < 2; repeat++) {
      dirty(categories, numerics);
      actualRows.copyInferenceInputsTo(categories, numerics, layout);
      assertWritten(categories, numerics, expectedCategories, expectedNumerics);
      for (int element = 0; element < expectedNumerics.length; element++) {
        Assert.assertEquals(
            Float.floatToRawIntBits(numerics.get(PREFIX + element)),
            Float.floatToRawIntBits(expectedNumerics[element]));
      }
    }
  }

  @Test(dataProvider = "bufferKinds")
  public void inferenceMatchesDenseTrainingInReusedCanonicalBuffers(boolean direct) {
    List<Scenario> scenarios = scenarios();
    DecisionHostBatch inference =
        inference(scenarios, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    DecisionHostBatch dense = training(scenarios);
    Assert.assertTrue(inference.actionTransitionCount(1, 0) > 1);
    Assert.assertTrue(inference.actionTransitionCount(2, 0) > 1);
    Assert.assertEquals(inference.actionTransitionCount(0, 0), 1);
    assertCanonical(inference.sliceRows(0, 4), dense.sliceRows(0, 4), direct);
    assertCanonical(inference.sliceRows(1, 2), dense.sliceRows(1, 2), direct);
  }

  @Test(dataProvider = "bufferKinds")
  public void selectedCopiedAndRestoredRowsKeepCanonicalPadding(boolean direct) {
    List<Scenario> scenarios = scenarios();
    DecisionHostBatch inference =
        inference(scenarios, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    DecisionHostBatch dense = training(scenarios);
    assertCanonical(inference.copyRows(1, 2).sliceRows(0, 2), dense.sliceRows(1, 2), direct);
    int[] selection = {3, 1, 3};
    DecisionHostBatch selected = inference.selectRows(selection, selection.length);
    DecisionHostBatch expected = dense.selectRows(selection, selection.length);
    assertCanonical(selected.sliceRows(0, 3), expected.sliceRows(0, 3), direct);
    DecisionHostBatch restored =
        DecisionHostBatch.takeEncodedInferenceBatch(
            3, BUCKET, categories(selected.sliceRows(0, 3)), numerics(selected.sliceRows(0, 3)));
    assertCanonical(restored.sliceRows(0, 3), expected.sliceRows(0, 3), direct);
  }

  @Test(dataProvider = "inferenceSourceKinds")
  public void inferenceTransferMatchesDenseForMultipleSourceSlices(boolean direct, boolean mixed) {
    List<Scenario> scenarios = scenarios();
    DecisionHostBatch inference =
        inference(scenarios, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    DecisionHostBatch dense = training(scenarios);
    DecisionHostBatch restored =
        DecisionHostBatch.takeEncodedInferenceBatch(
            4, BUCKET, categories(dense.sliceRows(0, 4)), numerics(dense.sliceRows(0, 4)));
    DecisionHostBatch.RowBatch actual =
        DecisionHostBatch.RowBatch.of(
            mixed
                ? List.of(
                    inference.sliceRows(2, 1), restored.sliceRows(3, 1), inference.sliceRows(0, 2))
                : List.of(inference.sliceRows(2, 2), inference.sliceRows(0, 2)));
    DecisionHostBatch.RowBatch expected =
        DecisionHostBatch.RowBatch.of(List.of(dense.sliceRows(2, 2), dense.sliceRows(0, 2)));
    var indexLayout = actual.inferenceIndexLayout(true);
    var inputLayout = actual.inferenceInputLayout(indexLayout);
    Assert.assertEquals(indexLayout, expected.inferenceIndexLayout(true));
    int[] expectedIndices = new int[indexLayout.totalCount()];
    expected.copyInferenceIndicesTo(IntBuffer.wrap(expectedIndices), indexLayout);
    IntBuffer indices =
        direct
            ? ByteBuffer.allocateDirect((PREFIX + expectedIndices.length + SUFFIX) * Integer.BYTES)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer()
            : IntBuffer.allocate(PREFIX + expectedIndices.length + SUFFIX);
    for (int index = 0; index < indices.capacity(); index++) indices.put(index, -1);
    indices.position(PREFIX);
    actual.copyInferenceIndicesTo(indices, indexLayout);
    Assert.assertEquals(indices.position(), PREFIX + expectedIndices.length);
    int[] actualIndices = new int[expectedIndices.length];
    indices.get(PREFIX, actualIndices);
    Assert.assertEquals(actualIndices, expectedIndices);
    for (int index = 0; index < PREFIX; index++) Assert.assertEquals(indices.get(index), -1);
    for (int index = 0; index < SUFFIX; index++) {
      Assert.assertEquals(indices.get(PREFIX + expectedIndices.length + index), -1);
    }
    short[] expectedCategories = new short[inputLayout.categoricalElementCount()];
    float[] expectedNumerics = new float[inputLayout.numericElementCount()];
    expected.copyInferenceInputsTo(
        ShortBuffer.wrap(expectedCategories), FloatBuffer.wrap(expectedNumerics), inputLayout);
    ShortBuffer categories = categoryBuffer(expectedCategories.length, direct);
    FloatBuffer numerics = numericBuffer(expectedNumerics.length, direct);
    for (int repeat = 0; repeat < 2; repeat++) {
      dirty(categories, numerics);
      actual.copyInferenceInputsTo(categories, numerics, inputLayout);
      assertWritten(categories, numerics, expectedCategories, expectedNumerics);
    }
  }

  @Test
  public void retainedSlicesOwnEncodedTransitionsAfterWorkspaceAndEngineStateChange() {
    List<Scenario> scenarios = scenarios();
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    DecisionHostBatch inference = inference(scenarios, workspaces);
    DecisionHostBatch dense = training(scenarios);
    DecisionHostBatch.RowSlice retained = inference.sliceRows(1, 2);
    DecisionHostBatch independent = inference.copyRows(1, 2);
    for (Scenario scenario : scenarios) scenario.state.hand(1).clear();
    inference(List.of(scenarios().get(0)), workspaces);
    inference(List.of(scenarios().get(3)), workspaces);
    assertCanonical(retained, dense.sliceRows(1, 2), false);
    assertCanonical(independent.sliceRows(0, 2), dense.sliceRows(1, 2), false);

    DecisionHostBatch.RowBatch actual =
        DecisionHostBatch.RowBatch.of(
            List.of(retained, inference.sliceRows(0, 1), independent.sliceRows(0, 1)));
    DecisionHostBatch.RowBatch expected =
        DecisionHostBatch.RowBatch.of(
            List.of(dense.sliceRows(1, 2), dense.sliceRows(0, 1), dense.sliceRows(1, 1)));
    DecisionHostBatch.InferenceIndexLayout layout = actual.inferenceIndexLayout(true);
    Assert.assertEquals(layout, expected.inferenceIndexLayout(true));
    int[] actualIndices = new int[layout.totalCount()];
    int[] expectedIndices = new int[layout.totalCount()];
    actual.copyInferenceIndicesTo(IntBuffer.wrap(actualIndices), layout);
    expected.copyInferenceIndicesTo(IntBuffer.wrap(expectedIndices), layout);
    Assert.assertEquals(actualIndices, expectedIndices);
  }

  @Test
  public void stateOnlyRowsDoNotWriteActionTransitionIndices() {
    DecisionHostBatch batch = DecisionBatchBuilder.stateOnlyBatch(scenarios().getFirst().state, 1);
    DecisionHostBatch.RowSlice rows = batch.sliceRows(0, 1);
    DecisionHostBatch.RowSlice dense = batch.copyRows(0, 1).sliceRows(0, 1);
    var indexLayout = rows.inferenceIndexLayout(true);
    Assert.assertEquals(indexLayout, dense.inferenceIndexLayout(true));
    Assert.assertEquals(indexLayout.transitionCount(), 0);
    var inputLayout = rows.inferenceInputLayout(indexLayout);
    short[] categories = new short[inputLayout.categoricalElementCount()];
    float[] numerics = new float[inputLayout.numericElementCount()];
    short[] expectedCategories = new short[categories.length];
    float[] expectedNumerics = new float[numerics.length];
    dense.copyInferenceInputsTo(
        ShortBuffer.wrap(expectedCategories), FloatBuffer.wrap(expectedNumerics), inputLayout);
    rows.copyInferenceInputsTo(
        ShortBuffer.wrap(categories), FloatBuffer.wrap(numerics), inputLayout);
    Assert.assertEquals(categories, expectedCategories);
    Assert.assertEquals(numerics, expectedNumerics);
    DecisionHostBatch.RowBatch.of(rows)
        .copyInferenceInputsTo(
            ShortBuffer.wrap(categories), FloatBuffer.wrap(numerics), inputLayout);
    Assert.assertEquals(categories, expectedCategories);
    Assert.assertEquals(numerics, expectedNumerics);
    IntBuffer destination = IntBuffer.wrap(new int[] {-1, -1, -1});
    destination.position(1).limit(1);
    rows.copyTransitionPresentIndicesTo(0, destination);
    Assert.assertEquals(destination.position(), 1);
    Assert.assertEquals(destination.limit(), 1);
    Assert.assertEquals(destination.array(), new int[] {-1, -1, -1});
    Assert.assertEquals(
        rows.policyExecutionLayout(), DecisionHostBatch.PolicyExecutionLayout.empty());
  }

  @Test(timeOut = 10_000)
  public void parallelEncodingSessionsPreserveEveryCanonicalRow() throws Exception {
    List<Scenario> scenarios = scenarios();
    DecisionHostBatch expected = training(scenarios);
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(
            scenarios.size(), BUCKET, new DecisionBatchBuilder.EncodingWorkspacePool(2));
    CountDownLatch sessionsOpened = new CountDownLatch(2);
    try (var executor = Executors.newFixedThreadPool(2)) {
      List<Future<?>> tasks = new ArrayList<>();
      for (int from = 0; from < scenarios.size(); from += 2) {
        int firstRow = from;
        tasks.add(
            executor.submit(
                () -> {
                  try (var session = builder.openEncoding()) {
                    sessionsOpened.countDown();
                    Assert.assertTrue(sessionsOpened.await(5, TimeUnit.SECONDS));
                    for (int row = firstRow; row < firstRow + 2; row++) {
                      Scenario source = scenarios.get(row);
                      session.encodeObservedInferenceRow(
                          row, source.state.publicObservation(1), source.actions);
                    }
                  }
                  return null;
                }));
      }
      for (Future<?> task : tasks) task.get(5, TimeUnit.SECONDS);
    }
    DecisionHostBatch actual = builder.buildEncodedInferenceRows();
    assertCanonical(actual.sliceRows(0, 4), expected.sliceRows(0, 4), false);
    var actualRows = actual.sliceRows(0, 4);
    var expectedRows = expected.sliceRows(0, 4);
    var layout = actualRows.inferenceIndexLayout(true);
    Assert.assertEquals(layout, expectedRows.inferenceIndexLayout(true));
    int[] actualIndices = new int[layout.totalCount()];
    int[] expectedIndices = new int[layout.totalCount()];
    actualRows.copyInferenceIndicesTo(IntBuffer.wrap(actualIndices), layout);
    expectedRows.copyInferenceIndicesTo(IntBuffer.wrap(expectedIndices), layout);
    Assert.assertEquals(actualIndices, expectedIndices);
  }

  @Test
  public void oneEncodingSessionPreservesDistinctPendingEngineDecisions() {
    List<GameEngine> engines =
        List.of(new GameEngine(17), new GameEngine(29), new GameEngine(43), new GameEngine(71));
    List<EngineDecisionPoint> points = new ArrayList<>();
    for (GameEngine engine : engines) {
      var awaiting = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
      points.add(awaiting.decisions().getFirst());
    }
    DecisionBucket bucket = new DecisionBucket(40, 16);
    DecisionBatchBuilder expected =
        DecisionBatchBuilder.inference(
            engines.size(), bucket, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    DecisionBatchBuilder actual =
        DecisionBatchBuilder.inference(
            engines.size(), bucket, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    for (int row = 0; row < engines.size(); row++) {
      expected.addInferenceRow(
          engines.get(row), points.get(row), DecisionBoundaryContext.uniform());
    }
    try (var session = actual.openEncoding()) {
      for (int row = engines.size() - 1; row >= 0; row--) {
        session.encodeInferenceRow(
            row, engines.get(row), points.get(row), DecisionBoundaryContext.uniform());
      }
    }
    assertCanonical(
        actual.buildEncodedInferenceRows().sliceRows(0, 4),
        expected.build().sliceRows(0, 4),
        false);
  }

  @Test
  public void failedEncodingScopeClosesAndLaterScopesPreserveWrittenRows() {
    List<Scenario> scenarios = scenarios();
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(
            scenarios.size(), BUCKET, new DecisionBatchBuilder.EncodingWorkspacePool(1));
    var failedSession = builder.openEncoding();
    IllegalArgumentException failure = new IllegalArgumentException("stop this encoding range");
    Assert.assertSame(
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> {
              try (failedSession) {
                Scenario source = scenarios.getFirst();
                failedSession.encodeObservedInferenceRow(
                    0, source.state.publicObservation(1), source.actions);
                throw failure;
              }
            }),
        failure);
    failedSession.close();
    Scenario replacement = scenarios.get(1);
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            failedSession.encodeObservedInferenceRow(
                0, replacement.state.publicObservation(1), replacement.actions));
    try (var session = builder.openEncoding()) {
      for (int row = 1; row < scenarios.size(); row++) {
        Scenario source = scenarios.get(row);
        session.encodeObservedInferenceRow(row, source.state.publicObservation(1), source.actions);
      }
    }
    assertCanonical(
        builder.buildEncodedInferenceRows().sliceRows(0, 4),
        training(scenarios).sliceRows(0, 4),
        false);
  }

  private static void assertCanonical(
      DecisionHostBatch.RowSlice actual, DecisionHostBatch.RowSlice expected, boolean direct) {
    short[] expectedCategories = categories(expected);
    float[] expectedNumerics = numerics(expected);
    ShortBuffer categories = categoryBuffer(expectedCategories.length, direct);
    FloatBuffer numerics = numericBuffer(expectedNumerics.length, direct);
    for (int repeat = 0; repeat < 2; repeat++) {
      dirty(categories, numerics);
      actual.copyInputCategoriesTo(categories);
      actual.copyInputNumericsTo(numerics);
      assertWritten(categories, numerics, expectedCategories, expectedNumerics);
    }
  }

  private static short[] categories(DecisionHostBatch.RowSlice rows) {
    short[] values = new short[rows.inputCategoricalElementCount()];
    rows.copyInputCategoriesTo(ShortBuffer.wrap(values));
    return values;
  }

  private static float[] numerics(DecisionHostBatch.RowSlice rows) {
    float[] values = new float[rows.inputNumericElementCount()];
    rows.copyInputNumericsTo(FloatBuffer.wrap(values));
    return values;
  }

  private static ShortBuffer categoryBuffer(int elements, boolean direct) {
    int count = PREFIX + elements + SUFFIX;
    return direct
        ? ByteBuffer.allocateDirect(count * Short.BYTES)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()
        : ShortBuffer.allocate(count);
  }

  private static FloatBuffer numericBuffer(int elements, boolean direct) {
    int count = PREFIX + elements + SUFFIX;
    return direct
        ? ByteBuffer.allocateDirect(count * Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        : FloatBuffer.allocate(count);
  }

  private static void dirty(ShortBuffer categories, FloatBuffer numerics) {
    for (int index = 0; index < categories.capacity(); index++)
      categories.put(index, DIRTY_CATEGORY);
    for (int index = 0; index < numerics.capacity(); index++) numerics.put(index, DIRTY_NUMERIC);
    categories.position(PREFIX);
    numerics.position(PREFIX);
  }

  private static void assertWritten(
      ShortBuffer categories,
      FloatBuffer numerics,
      short[] expectedCategories,
      float[] expectedNumerics) {
    Assert.assertEquals(categories.position(), PREFIX + expectedCategories.length);
    Assert.assertEquals(numerics.position(), PREFIX + expectedNumerics.length);
    short[] actualCategories = new short[expectedCategories.length];
    float[] actualNumerics = new float[expectedNumerics.length];
    categories.get(PREFIX, actualCategories);
    numerics.get(PREFIX, actualNumerics);
    Assert.assertEquals(actualCategories, expectedCategories);
    Assert.assertEquals(actualNumerics, expectedNumerics);
    for (int index = 0; index < PREFIX; index++) {
      Assert.assertEquals(categories.get(index), DIRTY_CATEGORY);
      Assert.assertEquals(numerics.get(index), DIRTY_NUMERIC);
    }
    for (int index = 0; index < SUFFIX; index++) {
      Assert.assertEquals(
          categories.get(PREFIX + expectedCategories.length + index), DIRTY_CATEGORY);
      Assert.assertEquals(numerics.get(PREFIX + expectedNumerics.length + index), DIRTY_NUMERIC);
    }
  }

  private static DecisionHostBatch inference(
      List<Scenario> scenarios, DecisionBatchBuilder.EncodingWorkspacePool workspaces) {
    return inference(scenarios, workspaces, BUCKET);
  }

  private static DecisionHostBatch inference(
      List<Scenario> scenarios,
      DecisionBatchBuilder.EncodingWorkspacePool workspaces,
      DecisionBucket bucket) {
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(scenarios.size(), bucket, workspaces);
    for (Scenario scenario : scenarios) {
      builder.addDetachedInferenceRow(
          scenario.state,
          1,
          scenario.actions,
          scenario.state.publicState(),
          DecisionBoundaryContext.uniform());
    }
    return builder.build();
  }

  private static DecisionHostBatch training(List<Scenario> scenarios) {
    return training(scenarios, BUCKET);
  }

  private static DecisionHostBatch training(List<Scenario> scenarios, DecisionBucket bucket) {
    DecisionBatchBuilder builder = DecisionBatchBuilder.training(scenarios.size(), bucket);
    for (Scenario scenario : scenarios) {
      float[] policy = new float[scenario.actions.size()];
      Arrays.fill(policy, 1.0f / policy.length);
      builder.addDetachedTrainingRow(
          scenario.state,
          1,
          scenario.actions,
          scenario.state.publicState(),
          DecisionBoundaryContext.uniform(),
          0,
          policy,
          policy,
          0.0f,
          0.0f,
          1.0f,
          1.0f);
    }
    return builder.build();
  }

  private static List<Scenario> scenarios() {
    GameState discard = state("123455m123p123s55z");
    GameState chi = state("355566m123p123s5z");
    chi.commitDiscard(0, Tile.M4, false);
    GameState pon = state("345556m123p123s5z");
    pon.commitDiscard(0, Tile.M5, false);
    GameState ron = state("123m123p123s1112z");
    ron.commitDiscard(0, Tile.NAN, false);
    return List.of(
        new Scenario(
            discard,
            List.of(
                Action.dahai(Tile.M5), Action.dahai(Tile.M5, true), Action.riichiDahai(Tile.M5))),
        new Scenario(
            chi,
            List.of(
                Action.chiSequence(Tile.M3, Tile.M4, false),
                Action.chiSequence(Tile.M4, Tile.M4, true),
                Action.pass())),
        new Scenario(
            pon,
            List.of(
                Action.pon(Tile.M5),
                Action.pon(Tile.M5, true),
                Action.daiminkan(Tile.M5),
                Action.pass())),
        new Scenario(ron, List.of(Action.ronAgari(), Action.pass())));
  }

  private static GameState state(String notation) {
    GameState state = new GameState(17);
    state.startRound(0, 0, 0, 0);
    state.setCurrentPlayer(1);
    state.initializeWallForReconstruction(new int[] {Tile.M4}, 52);
    state.commitDiscard(1, Tile.TON, false);
    Hand hand = state.hand(1);
    hand.clear();
    int start = 0;
    for (int end = 0; end < notation.length(); end++) {
      int offset =
          switch (notation.charAt(end)) {
            case 'm' -> 0;
            case 'p' -> 9;
            case 's' -> 18;
            case 'z' -> 27;
            default -> -1;
          };
      if (offset < 0) continue;
      for (int digit = start; digit < end; digit++) hand.add(offset + notation.charAt(digit) - '1');
      start = end + 1;
    }
    if (hand.count(Tile.M5) > 0) {
      hand.removeNonAka(Tile.M5);
      hand.addPhysicalTile(Tile.M5 * 4);
    }
    return state;
  }

  private record Scenario(GameState state, List<Action> actions) {}
}
