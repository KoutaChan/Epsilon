package com.epsilon.major.ai.decision.data;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.major.ai.decision.input.DecisionInputLayout;
import com.epsilon.major.ai.decision.input.DecisionTrainingSlabWriter;
import com.epsilon.major.ai.decision.input.DecisionTrainingTargetLayout;
import com.epsilon.major.ai.decision.input.DecisionTrainingTargets;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.BitSet;
import org.testng.Assert;

/** Arenaの実サンプルで外部payload、fragmentの読み飛ばし、直接転送までを確認する。 */
public final class BranchFragmentAssertions {
  private BranchFragmentAssertions() {}

  public static void verify(EpsilonDecisionCompletedGame game) throws Exception {
    var identity = EpsilonDecisionTrainingTargetIdentity.selectedPg(.9f, .1f);
    var bytes = new ByteArrayOutputStream();
    EpsilonDecisionFragmentCodec.write(bytes, new EpsilonDecisionFragment(1, 1, game, identity));
    var decoded = EpsilonDecisionFragmentCodec.read(new ByteArrayInputStream(bytes.toByteArray()));
    for (int i = 0; i < game.samples().size(); i++) {
      Assert.assertEquals(
          decoded.game().samples().get(i).branchTarget(), game.samples().get(i).branchTarget());
    }
    BitSet selected = new BitSet();
    selected.set(0);
    selected.set(game.samples().size() - 1);
    var descriptors =
        EpsilonDecisionFragmentCodec.readSelected(
            new ByteArrayInputStream(bytes.toByteArray()),
            selected,
            0,
            game.samples().size(),
            new EpsilonDecisionFragmentStore.ExpectedIdentity(1, 1, identity));
    Assert.assertEquals(descriptors.size(), 2);
    try (var session = EpsilonDecisionTrajectoryPayloadStore.openExternalFileSession()) {
      for (var descriptor : descriptors) {
        var target = descriptor.branchTarget();
        Assert.assertEquals(descriptor.materializeChecked().branchTarget(), target);
        var bucket = descriptor.bucket();
        var inputLayout = new DecisionInputLayout(1, bucket);
        var targetLayout = new DecisionTrainingTargetLayout(1, bucket);
        FloatBuffer numerics = FloatBuffer.allocate(targetLayout.numericElementCount());
        var destination =
            new DecisionTrainingSlabWriter(
                1,
                bucket,
                ShortBuffer.allocate(inputLayout.categoricalElementCount()),
                FloatBuffer.allocate(inputLayout.numericElementCount()),
                IntBuffer.allocate(1),
                numerics,
                IntBuffer.allocate(4096),
                IntBuffer.allocate(
                    bucket.legalActionCapacity() * bucket.actionTransitionCapacity()));
        session.readTrainingRow(descriptor, destination, 0, 1, 1);
        destination.seal();
        float[] expected = new float[DecisionBranchTarget.WIDTH];
        target.writeTo(expected, 0);
        for (int i = 0; i < expected.length; i++) {
          Assert.assertEquals(
              numerics.get(
                  targetLayout.rowNumericOffset(0) + DecisionTrainingTargets.BRANCH_OFFSET + i),
              expected[i]);
        }
      }
    }
  }
}
