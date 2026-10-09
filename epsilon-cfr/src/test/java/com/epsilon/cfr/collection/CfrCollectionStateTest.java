package com.epsilon.cfr.collection;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.core.Action;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundTransition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrCollectionStateTest {
  @Test
  public void consecutiveRootsOwnScoresAndGrpPrefixesAndCountRepeatedRounds() {
    RootState initial = RootState.initial();
    int[] scores = {28000, 24000, 24000, 24000};
    RootState repeat = initial.next(new RoundTransition.NextRound(0, 0, 1, 0), scores);
    RootState branch = initial.next(new RoundTransition.NextRound(1, 1, 0, 0), scores);
    Assert.assertEquals(initial.roundOrdinal(), 1);
    Assert.assertEquals(repeat.roundOrdinal(), 2);
    Assert.assertEquals(repeat.kyokuIndex(), initial.kyokuIndex());
    Assert.assertEquals(branch.roundOrdinal(), 2);
    Assert.assertEquals(branch.kyokuIndex(), 1);
    Assert.assertEquals(branch.dealer(), 1);
    Assert.assertEquals(initial.scores(), new int[] {25000, 25000, 25000, 25000});
    Assert.assertEquals(
        Arrays.copyOf(repeat.publicPrefix(), EpsilonGrpFeature.FEATURE_SIZE),
        initial.publicPrefix());
    Assert.assertEquals(
        Arrays.copyOfRange(
            repeat.publicPrefix(), EpsilonGrpFeature.FEATURE_SIZE, repeat.publicPrefix().length),
        EpsilonGrpFeature.fromProgress(0, 1, 0, scores));
    Assert.assertEquals(
        Arrays.copyOfRange(
            branch.publicPrefix(), EpsilonGrpFeature.FEATURE_SIZE, branch.publicPrefix().length),
        EpsilonGrpFeature.fromProgress(1, 0, 0, scores));
    scores[0] = 0;
    repeat.scores()[1] = 0;
    repeat.publicPrefix()[0] = -1;
    Assert.assertEquals(repeat.scores(), new int[] {28000, 24000, 24000, 24000});
    Assert.assertEquals(branch.scores(), repeat.scores());
    Assert.assertEquals(initial.roundOrdinal(), 1);
    Assert.assertEquals(
        Arrays.copyOf(repeat.publicPrefix(), EpsilonGrpFeature.FEATURE_SIZE),
        initial.publicPrefix());
  }

  @Test
  public void contextsOwnLegalActionsAndKeepRoundMetadataOutOfTheModelInput() {
    var engine = new GameEngine(731);
    var boundary = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var point = boundary.decisions().getFirst();
    var base =
        new DecisionContext(
            731,
            point.id(),
            point.player(),
            engine.observation(point.player()),
            point.legalActions());
    var legal = new ArrayList<>(base.legalActions());
    var first = new DecisionContext(1, 2, base.seat(), base.observation(), legal, 1);
    var repeat = new DecisionContext(3, 4, base.seat(), base.observation(), legal, 7);
    legal.clear();
    Assert.assertEquals(first.legalActions(), base.legalActions());
    Assert.assertSame(first.observation(), base.observation());
    Assert.expectThrows(UnsupportedOperationException.class, () -> first.legalActions().clear());
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(first), CfrObservationCodec.encodePayload(repeat));
    Assert.assertEquals(first.roundOrdinal(), 1);
    Assert.assertEquals(repeat.roundOrdinal(), 7);
    Assert.assertEquals(
        new DecisionContext(0, 0, 0, null, List.of(Action.pass())).roundOrdinal(), 0);
  }
}
