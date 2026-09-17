package com.epsilon.nano.ai.decision.data;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.github.luben.zstd.Zstd;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 非同期保存した学習データを読み出せることと、永続化時の書き込み完了を検証する。 */
public class EpsilonDecisionTrajectoryPayloadStoreTest {

  @DataProvider
  public Object[][] transitions() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "transitions")
  public void asyncPayloadPreservesRowsAndRemainsReadableAfterDurableClose(
      boolean multipleTransitions) throws Exception {
    EpsilonDecisionTrajectoryPayload expected = payload(multipleTransitions);
    byte[] canonical = canonicalBytes(expected);
    Path directory = Files.createTempDirectory("decision-payload-");
    try {
      FilePayloadRef written;
      try (var store =
          EpsilonDecisionTrajectoryPayloadStore.asyncFileBacked(directory, 1, 1, true)) {
        var pending =
            store.write(expected.input(), expected.behaviorPolicy(), expected.rolloutPolicy());
        assertPayload(store.read(pending), expected);
        written = store.externalFileRef(pending);
        Assert.assertEquals(written.bucket(), expected.input().bucket());
      }
      Assert.assertTrue(Files.isRegularFile(Path.of(written.file())));
      assertPayload(
          EpsilonDecisionTrajectoryPayloadStore.externalFileReader().read(written), expected);

      byte[] stored = Files.readAllBytes(Path.of(written.file()));
      byte[] frame =
          Arrays.copyOfRange(
              stored,
              Math.toIntExact(written.offset()),
              Math.toIntExact(written.offset()) + written.length());
      Assert.assertEquals(Zstd.decompress(frame, canonical.length), canonical);

      byte[] legacyFrame = Zstd.compress(canonical, 0);
      Path legacyFile = directory.resolve("heap-array-frame.bin");
      Files.write(legacyFile, new byte[7]);
      Files.write(legacyFile, legacyFrame, StandardOpenOption.APPEND);
      FilePayloadRef legacy =
          new FilePayloadRef(
              legacyFile.toString(), 0, 7L, legacyFrame.length, expected.input().bucket());
      assertPayload(
          EpsilonDecisionTrajectoryPayloadStore.externalFileReader().read(legacy), expected);
      try (var session = EpsilonDecisionTrajectoryPayloadStore.openExternalFileSession()) {
        EpsilonDecisionTrajectoryPayload decoded =
            session.read(legacy, EpsilonDecisionTrajectoryPayloadCodec::read);
        assertPayload(decoded, expected);
      }
    } finally {
      try (var files = Files.list(directory)) {
        for (Path file : files.toList()) Files.deleteIfExists(file);
      }
      Files.delete(directory);
    }
  }

  private static EpsilonDecisionTrajectoryPayload payload(boolean multipleTransitions) {
    GameState state = state(multipleTransitions ? "355566m123p123s5z" : "123455m123p123s55z");
    if (multipleTransitions) state.commitDiscard(0, Tile.M4, false);
    List<Action> actions =
        multipleTransitions
            ? List.of(
                Action.chiSequence(Tile.M3, Tile.M4, false),
                Action.chiSequence(Tile.M4, Tile.M4, true),
                Action.pass())
            : List.of(
                Action.dahai(Tile.M5), Action.dahai(Tile.M5, true), Action.riichiDahai(Tile.M5));
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(2, new DecisionBucket(4, multipleTransitions ? 16 : 1));
    builder.addDetachedInferenceRow(
        state,
        1,
        List.of(actions.getLast()),
        state.publicState(),
        DecisionBoundaryContext.uniform());
    builder.addDetachedInferenceRow(
        state, 1, actions, state.publicState(), DecisionBoundaryContext.uniform());
    DecisionHostBatch batch = builder.build();
    if (multipleTransitions) Assert.assertTrue(batch.actionTransitionCount(1, 0) > 1);
    else Assert.assertEquals(batch.actionTransitionCount(1, 0), 1);
    return new EpsilonDecisionTrajectoryPayload(
        batch.sliceRows(1, 1),
        new float[] {0.125f, 0.375f, 0.5f},
        new float[] {0.25f, 0.25f, 0.5f});
  }

  private static byte[] canonicalBytes(EpsilonDecisionTrajectoryPayload payload) throws Exception {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream output = new DataOutputStream(bytes)) {
      EpsilonDecisionTrajectoryPayloadCodec.write(
          output, payload, new EpsilonDecisionBinaryArrayCodec.Scratch());
    }
    return bytes.toByteArray();
  }

  private static void assertPayload(
      EpsilonDecisionTrajectoryPayload actual, EpsilonDecisionTrajectoryPayload expected) {
    Assert.assertEquals(actual.input().bucket(), expected.input().bucket());
    Assert.assertEquals(categories(actual.input()), categories(expected.input()));
    Assert.assertEquals(numerics(actual.input()), numerics(expected.input()));
    Assert.assertEquals(actual.behaviorPolicy(), expected.behaviorPolicy());
    Assert.assertEquals(actual.rolloutPolicy(), expected.rolloutPolicy());
  }

  private static short[] categories(DecisionHostBatch.RowSlice row) {
    short[] values = new short[row.inputCategoricalElementCount()];
    row.copyInputCategoriesTo(ShortBuffer.wrap(values));
    return values;
  }

  private static float[] numerics(DecisionHostBatch.RowSlice row) {
    float[] values = new float[row.inputNumericElementCount()];
    row.copyInputNumericsTo(FloatBuffer.wrap(values));
    return values;
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
    hand.removeNonAka(Tile.M5);
    hand.addPhysicalTile(Tile.M5 * 4);
    return state;
  }
}
