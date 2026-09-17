package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 行動候補を仮に反映した結果が元の局面を変更せず、作業領域を再利用しても全体を再計算した結果と一致することを検証する。 */
public class DecisionProjectionTest {
  @DataProvider
  public Object[][] scenarios() {
    return new Object[][] {
      {"discard", false},
      {"discard", true},
      {"chi", false},
      {"chi", true},
      {"pon", false},
      {"pon", true},
      {"calledAka", false},
      {"ankan", false},
      {"ankan", true},
      {"kakan", false},
      {"kakan", true},
      {"tsumo", false},
      {"ron", false},
      {"kyushu", false}
    };
  }

  @Test(dataProvider = "scenarios")
  public void metadataMatchesFullProjectionWithoutChangingSource(String name, boolean aka) {
    Scenario scenario = scenario(name, aka);
    var original = new ActionEffect.ProjectedHand();
    original.copyFrom(scenario.state.hand(1));
    var expectedHand = new ActionEffect.ProjectedHand();
    var scratch = new ActionEffect.ProjectedHand();
    var expected = new ActionEffect.ProjectionResult();
    var actual = new ActionEffect.ProjectionResult();
    List<Action> expectedDiscards = new ArrayList<>();
    List<Action> actualDiscards = new ArrayList<>();
    actualDiscards.add(Action.dahai(Tile.M1));
    for (Action action : scenario.actions) {
      ActionEffect.projectInto(scenario.state, 1, action, expectedHand, expected, expectedDiscards);
      ActionEffect.describeInto(scenario.state, 1, action, scratch, actual, actualDiscards);
      Assert.assertEquals(actual.nextStep(), expected.nextStep());
      Assert.assertEquals(actual.meldAkaSource(), expected.meldAkaSource());
      Assert.assertEquals(actualDiscards, expectedDiscards);
      assertHand(scenario.state.hand(1), original);
    }
  }

  @Test
  public void reusedDecisionBufferMatchesFullProjectionForEveryActionAndTransition() {
    EngineDecisionBuffer buffer = new EngineDecisionBuffer();
    EnumSet<Action.Type> types = EnumSet.noneOf(Action.Type.class);
    for (Object[] data : scenarios()) {
      Scenario scenario = scenario((String) data[0], (boolean) data[1]);
      var original = new ActionEffect.ProjectedHand();
      original.copyFrom(scenario.state.hand(1));
      buffer.bind(scenario.state, 1, scenario.actions, scenario.state.publicState());
      Assert.assertSame(buffer.legalActions(), scenario.actions);
      Assert.assertEquals(buffer.actionCount(), scenario.actions.size());
      for (int pass = 0; pass < 2; pass++) {
        for (int offset = 0; offset < scenario.actions.size(); offset++) {
          int index = pass == 0 ? offset : scenario.actions.size() - 1 - offset;
          Action action = scenario.actions.get(index);
          types.add(action.type());
          var root = new ActionEffect.ProjectedHand();
          var expected = new ActionEffect.ProjectedHand();
          var projected = new ActionEffect.ProjectionResult();
          List<Action> discards = new ArrayList<>();
          ActionEffect.projectInto(scenario.state, 1, action, root, projected, discards);
          Assert.assertSame(buffer.action(index), action);
          Assert.assertEquals(buffer.continuation(index), projected.nextStep());
          Assert.assertEquals(buffer.meldAkaSource(index), projected.meldAkaSource());
          int transitions = Math.max(1, discards.size());
          Assert.assertEquals(buffer.transitionCount(index), transitions);
          for (int position = 0; position < transitions; position++) {
            int transition = pass == 0 ? position : transitions - 1 - position;
            Action discard;
            if (projected.nextStep() == ActionEffect.NextStep.IMMEDIATE_DISCARD) {
              discard = discards.get(transition);
              ActionEffect.projectImmediateDiscardInto(root, discard, expected);
            } else {
              discard =
                  switch (action.type()) {
                    case DAHAI, RIICHI_DAHAI -> action;
                    default -> null;
                  };
              expected.copyFrom(root);
            }
            Assert.assertEquals(buffer.discardAction(index, transition), discard);
            HandView actual = buffer.handAfterTransition(index, transition);
            assertHand(actual, expected);
            Assert.assertSame(buffer.handAfterTransition(index, transition), actual);
            assertHand(scenario.state.hand(1), original);
          }
          if (action.type() == Action.Type.TSUMO_AGARI || action.type() == Action.Type.RON_AGARI) {
            WinSettlementProjection win =
                new WinSettlementProjection()
                    .load(scenario.state, 1, action, new HandScoreEvaluator());
            VisibleHandScoreBuffer actual = buffer.immediateWin(index).visibleScore();
            Assert.assertTrue(actual.available());
            Assert.assertEquals(actual.yakuBits(), win.visibleScore().yakuBits());
            Assert.assertEquals(actual.visibleHan(), win.visibleScore().visibleHan());
            Assert.assertEquals(actual.fu(), win.visibleScore().fu());
            Assert.assertEquals(actual.basePoints(), win.visibleScore().basePoints());
            for (int seat = 0; seat < 4; seat++) {
              Assert.assertEquals(
                  buffer.immediateWin(index).paymentFloor(seat), win.paymentFloor(seat));
              Assert.assertEquals(
                  buffer.immediateWin(index).projectedRank(seat), win.projectedRank(seat));
            }
          }
        }
      }
    }
    Assert.assertEquals(types, EnumSet.allOf(Action.Type.class));
  }

  private static Scenario scenario(String name, boolean aka) {
    GameState state = MahjongFixtures.state();
    state.setCurrentPlayer(1);
    state.initializeWallForReconstruction(new int[] {Tile.M4}, 52);
    state.commitDiscard(1, Tile.TON, false);
    List<Action> actions =
        switch (name) {
          case "discard" -> {
            MahjongFixtures.hand(state, 1, "123455m123p123s55z");
            yield List.of(
                Action.dahai(Tile.M5, aka),
                Action.riichiDahai(Tile.M5, aka),
                Action.dahai(Tile.P1));
          }
          case "chi" -> {
            MahjongFixtures.hand(state, 1, "355566m123p123s5z");
            state.commitDiscard(0, Tile.M4, false);
            yield List.of(
                Action.chiSequence(Tile.M3, Tile.M4, aka),
                Action.chiSequence(Tile.M4, Tile.M4, aka),
                Action.pass());
          }
          case "pon", "calledAka" -> {
            MahjongFixtures.hand(state, 1, "345556m123p123s5z");
            state.commitDiscard(0, Tile.M5, name.equals("calledAka"));
            yield List.of(Action.pon(Tile.M5, aka), Action.daiminkan(Tile.M5), Action.pass());
          }
          case "ankan" -> {
            MahjongFixtures.hand(state, 1, "5555m123p123s1112z");
            yield List.of(Action.ankan(Tile.M5), Action.dahai(Tile.M5, aka));
          }
          case "kakan" -> {
            MahjongFixtures.hand(state, 1, "5m123p123s1122z");
            state.addMeld(1, Meld.pon(Tile.M5, Meld.RelativeSource.KAMICHA), 0);
            yield List.of(Action.kakan(Tile.M5), Action.dahai(Tile.M5, aka));
          }
          case "tsumo" -> {
            MahjongFixtures.hand(state, 1, "123m123p123s11122z");
            state.recordDraw(1, Tile.NAN * 4, TurnEvent.DrawSource.WALL, false);
            yield List.of(Action.tsumoAgari(), Action.dahai(Tile.NAN));
          }
          case "ron" -> {
            MahjongFixtures.hand(state, 1, "123m123p123s1112z");
            state.commitDiscard(0, Tile.NAN, false);
            yield List.of(Action.ronAgari(), Action.pass());
          }
          case "kyushu" -> {
            MahjongFixtures.hand(state, 1, "119m19p19s1234567z");
            yield List.of(Action.kyushuKyuhai(), Action.dahai(Tile.M1));
          }
          default -> throw new AssertionError(name);
        };
    if (aka) {
      state.hand(1).removeNonAka(Tile.M5);
      state.hand(1).addPhysicalTile(Tile.M5 * 4);
    }
    return new Scenario(state, actions);
  }

  private static void assertHand(HandView actual, HandView expected) {
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      Assert.assertEquals(actual.count(tile), expected.count(tile));
      Assert.assertEquals(actual.hasAkaTile(tile), expected.hasAkaTile(tile));
    }
    Assert.assertEquals(actual.concealedTileCount(), expected.concealedTileCount());
    Assert.assertEquals(actual.concealedTileTypeMask(), expected.concealedTileTypeMask());
    Assert.assertEquals(actual.concealedPairTileTypeMask(), expected.concealedPairTileTypeMask());
    Assert.assertEquals(actual.concealedAkaMask(), expected.concealedAkaMask());
    Assert.assertEquals(actual.ownedAkaMask(), expected.ownedAkaMask());
    Assert.assertEquals(actual.canonicalMeldSignature(), expected.canonicalMeldSignature());
    Assert.assertEquals(actual.meldCount(), expected.meldCount());
    Assert.assertEquals(actual.kanCount(), expected.kanCount());
    Assert.assertEquals(actual.isMenzen(), expected.isMenzen());
    for (int meld = 0; meld < expected.meldCount(); meld++) {
      Assert.assertEquals(actual.meld(meld), expected.meld(meld));
      Assert.assertEquals(
          actual.meldCallAfterRiverIndex(meld), expected.meldCallAfterRiverIndex(meld));
      Assert.assertEquals(
          actual.meldKanAfterRiverIndex(meld), expected.meldKanAfterRiverIndex(meld));
    }
  }

  private record Scenario(GameState state, List<Action> actions) {}
}
