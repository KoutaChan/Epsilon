package com.epsilon.major.ai.decision.input;

import com.epsilon.core.Action;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 複数の入力領域から推論バッチを作る際、各参照位置が正しく対応付けられることを検証する。 */
public class DecisionInferenceIndexTest {
  @DataProvider
  public Object[][] storageKinds() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "storageKinds")
  public void multipleSlicesWriteGlobalIndicesFromTheirSourceRows(boolean direct) {
    DecisionHostBatch first = batch();
    DecisionHostBatch second = batch();
    DecisionHostBatch.RowBatch rows =
        DecisionHostBatch.RowBatch.of(List.of(first.sliceRows(2, 1), second.sliceRows(1, 2)));
    assertIndices(
        rows,
        new int[] {
          0, 29, 58, 59, 87, 112, 113,
          116, 117, 118, 145, 170, 174, 203,
          232, 261, 290, 291, 319, 344, 345
        },
        new int[] {0, 16, 17, 32, 33, 34, 64, 80, 81, 82, 128, 144, 145, 160, 161, 162},
        direct);
  }

  @Test(dataProvider = "storageKinds")
  public void singleSliceStartsIndicesAtZeroDespiteNonzeroSourceOffset(boolean direct) {
    assertIndices(
        DecisionHostBatch.RowBatch.of(batch().sliceRows(1, 2)),
        new int[] {0, 1, 2, 29, 54, 58, 87, 116, 145, 174, 175, 203, 228, 229},
        new int[] {0, 16, 17, 18, 64, 80, 81, 96, 97, 98},
        direct);
  }

  private static void assertIndices(
      DecisionHostBatch.RowBatch rows, int[] players, int[] transitions, boolean direct) {
    var layout = rows.inferenceIndexLayout(false);
    Assert.assertEquals(layout.playerMemoryCount(), players.length);
    Assert.assertEquals(layout.transitionCount(), transitions.length);
    int start = 3;
    int count = start + layout.totalCount() + 2;
    IntBuffer destination =
        direct
            ? ByteBuffer.allocateDirect(count * Integer.BYTES)
                .order(ByteOrder.nativeOrder())
                .asIntBuffer()
            : IntBuffer.allocate(count);
    for (int index = 0; index < count; index++) destination.put(index, -1);
    destination.position(start);
    rows.copyInferenceIndicesTo(destination, layout);
    Assert.assertEquals(destination.position(), start + layout.totalCount());
    int[] expected = new int[count];
    java.util.Arrays.fill(expected, -1);
    System.arraycopy(players, 0, expected, start + layout.playerMemoryOffset(), players.length);
    System.arraycopy(
        transitions, 0, expected, start + layout.transitionOffset(), transitions.length);
    int[] actual = new int[count];
    destination.get(0, actual);
    Assert.assertEquals(actual, expected);
  }

  private static DecisionHostBatch batch() {
    DecisionHostBatch batch = new DecisionHostBatch(3, new DecisionBucket(4, 16), false);
    for (int row = 0; row < 3; row++) {
      DecisionInputWriter writer = batch.inputs().writer(row);
      if (row == 1) {
        writer.player(0, DecisionInputSchema.PlayerInt.RIVER_COUNT, 2);
        writer.player(1, DecisionInputSchema.PlayerInt.MELD_COUNT, 1);
      } else if (row == 2) {
        writer.player(2, DecisionInputSchema.PlayerInt.RIVER_COUNT, 1);
        writer.player(3, DecisionInputSchema.PlayerInt.MELD_COUNT, 2);
      }
      for (int action = 0; action <= row; action++) {
        writer.action(action, DecisionInputSchema.ActionInt.ID, action + 1);
        writer.action(
            action,
            DecisionInputSchema.ActionInt.GROUP,
            DecisionFeatureCodec.actionGroup(Action.Type.PASS.group()));
        writer.action(
            action,
            DecisionInputSchema.ActionInt.TYPE,
            DecisionFeatureCodec.actionType(Action.Type.PASS));
        int transitions = row == 1 && action == 1 ? 3 : action + 1;
        for (int transition = 0; transition < transitions; transition++) {
          writer.transition(action, transition, DecisionInputSchema.ActionTransitionInt.PRESENT, 1);
          writer.transition(
              action,
              transition,
              DecisionInputSchema.ActionTransitionInt.KIND,
              DecisionInputSchema.ActionTransitionKind.IDENTITY.ordinal() + 1);
        }
      }
      batch.commitActionRow(row);
    }
    return batch;
  }

  @Test
  public void winningActionsUseGlobalOffsetsAndExcludePao() {
    DecisionHostBatch first = batch();
    DecisionHostBatch second = batch();
    var a = first.inputs().writer(2);
    a.action(
        1,
        DecisionInputSchema.ActionInt.TYPE,
        DecisionFeatureCodec.actionType(Action.Type.RON_AGARI));
    a.actionWinFact(1, DecisionInputSchema.WinFact.VALID, 1);
    var b = second.inputs().writer(1);
    b.action(
        0,
        DecisionInputSchema.ActionInt.TYPE,
        DecisionFeatureCodec.actionType(Action.Type.TSUMO_AGARI));
    b.actionWinFact(0, DecisionInputSchema.WinFact.VALID, 1);
    b.action(
        1,
        DecisionInputSchema.ActionInt.TYPE,
        DecisionFeatureCodec.actionType(Action.Type.RON_AGARI));
    b.actionWinFact(1, DecisionInputSchema.WinFact.VALID, 1);
    b.actionWinFact(1, DecisionInputSchema.WinFact.REQUIRES_PAO_CORRECTION, 1);
    var rows =
        DecisionHostBatch.RowBatch.of(List.of(first.sliceRows(2, 1), second.sliceRows(1, 2)));
    var layout = rows.inferenceIndexLayout(false);
    Assert.assertEquals(layout.winningCount(), 2);
    IntBuffer indices = IntBuffer.allocate(layout.totalCount());
    rows.copyInferenceIndicesTo(indices, layout);
    Assert.assertEquals(indices.get(layout.winningOffset()), 1);
    Assert.assertEquals(indices.get(layout.winningOffset() + 1), 4);
  }

  @Test
  public void waitRowsUseCompactedTransitionOffsetsAcrossSlices() {
    DecisionHostBatch first = batch();
    DecisionHostBatch second = batch();
    first
        .inputs()
        .writer(2)
        .waitWinFact(
            1, 1, 12, DecisionInputSchema.WaitWinType.TSUMO, DecisionInputSchema.WinFact.VALID, 1);
    second
        .inputs()
        .writer(1)
        .waitWinFact(
            1, 2, 0, DecisionInputSchema.WaitWinType.RON, DecisionInputSchema.WinFact.VALID, 1);
    var rows =
        DecisionHostBatch.RowBatch.of(List.of(first.sliceRows(2, 1), second.sliceRows(1, 2)));
    var layout = rows.inferenceIndexLayout(false);
    Assert.assertEquals(layout.waitCount(), 2);
    IntBuffer indices = IntBuffer.allocate(layout.totalCount());
    rows.copyInferenceIndicesTo(indices, layout);
    Assert.assertEquals(
        indices.get(layout.waitOffset()), 2 * DecisionInputSchema.MAX_WAIT_TILE_TYPES + 12);
    Assert.assertEquals(
        indices.get(layout.waitOffset() + 1), 9 * DecisionInputSchema.MAX_WAIT_TILE_TYPES);
  }
}
