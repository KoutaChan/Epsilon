package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.ActionSection;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 席ごとの判断と、選んだ手が行われたかを記録する入れ物を確かめる。 */
public class DecisionsTest {

  private static final int PASS = Action.code(Action.Type.PASS);
  private static final int RON = Action.code(Action.Type.RON_AGARI);

  @Test
  public void slotsKeepTheirSeatKindIdAndLegalCodesInOrder() {
    Decisions decisions = new Decisions(Variant.TENHOU_4P);
    decisions.open(2, EngineDecisionKind.RESPONSE);
    decisions.add(RON);
    decisions.add(PASS);
    Assert.assertTrue(decisions.close(7));
    decisions.open(3, EngineDecisionKind.RESPONSE);
    decisions.add(PASS);
    Assert.assertTrue(decisions.close(8));
    Assert.assertEquals(decisions.size(), 2);
    Assert.assertEquals(decisions.id(0), 7);
    Assert.assertEquals(decisions.seat(0), 2);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.RESPONSE);
    Assert.assertEquals(decisions.legalCount(0), 2);
    Assert.assertEquals(decisions.legal(0, 0), RON);
    Assert.assertEquals(decisions.legal(0, 1), PASS);
    Assert.assertEquals(decisions.id(1), 8);
    Assert.assertEquals(decisions.legal(1, 0), PASS);
    Assert.assertEquals(decisions.slotOf(3), 1);
    Assert.assertEquals(decisions.slotOf(0), -1);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decisions.legal(1, 1));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decisions.id(2));
    decisions.clear();
    Assert.assertEquals(decisions.size(), 0);
    Assert.assertEquals(decisions.slotOf(2), -1);
  }

  /** 合法手が 1 つも無い席の判断は残らない。その判断の番号も使われず、次の判断が同じ番号を使える。 */
  @Test
  public void aSlotWithoutLegalActionsIsDropped() {
    Decisions decisions = new Decisions(Variant.TENHOU_3P);
    decisions.open(1, EngineDecisionKind.NUKI_RESPONSE);
    Assert.assertFalse(decisions.close(4));
    decisions.open(2, EngineDecisionKind.NUKI_RESPONSE);
    decisions.add(RON);
    Assert.assertTrue(decisions.close(4));
    Assert.assertEquals(decisions.size(), 1);
    Assert.assertEquals(decisions.seat(0), 2);
    Assert.assertEquals(decisions.id(0), 4);
  }

  @Test
  public void openingAndAddingFollowTheContract() {
    Decisions decisions = new Decisions(Variant.TENHOU_4P);
    Assert.assertThrows(IllegalStateException.class, () -> decisions.add(PASS));
    Assert.assertThrows(IllegalStateException.class, () -> decisions.close(1));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> decisions.open(0, EngineDecisionKind.NUKI_RESPONSE));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decisions.open(4, EngineDecisionKind.TURN));
    decisions.open(0, EngineDecisionKind.TURN);
    Assert.assertThrows(IllegalStateException.class, () -> decisions.open(1, EngineDecisionKind.TURN));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decisions.add(ActionSection.space(Variant.TENHOU_4P)));
    for (int kind = 0; kind < Decisions.MAX_LEGAL; kind++) {
      decisions.add(kind % Tiles.KINDS);
    }
    Assert.assertThrows(IllegalStateException.class, () -> decisions.add(PASS));
    Assert.assertTrue(decisions.close(1));
    Assert.assertThrows(IllegalStateException.class, () -> decisions.open(0, EngineDecisionKind.RESPONSE));
    decisions.open(1, EngineDecisionKind.RESPONSE);
    decisions.add(PASS);
    decisions.close(2);
    decisions.open(2, EngineDecisionKind.RESPONSE);
    decisions.add(PASS);
    decisions.close(3);
    Assert.assertThrows(IllegalStateException.class, () -> decisions.open(3, EngineDecisionKind.RESPONSE));
  }

  /** 和了の手は、点数のコピーを判断の中に持つ（元の点数を消しても残る）。和了でない手を和了の手として足すと例外になる。 */
  @Test
  public void aWinKeepsACopyOfItsScoreInTheSlot() {
    Scene scene = Scene.first().hand(0, "123m456p789s234s1z").draw(0, "1z").turn();
    WinScore score = new WinScore();
    score.copyFrom(scene.decisions.win(0));
    Assert.assertTrue(score.has(ScoringYaku.TENHOU));
    Decisions decisions = new Decisions(Variant.TENHOU_4P);
    decisions.open(0, EngineDecisionKind.TURN);
    decisions.addWin(Action.code(Action.Type.TSUMO_AGARI), score);
    score.clear();
    Assert.assertThrows(IllegalArgumentException.class, () -> decisions.addWin(PASS, score));
    decisions.close(1);
    Assert.assertTrue(decisions.win(0).has(ScoringYaku.TENHOU));
    Assert.assertEquals(decisions.legal(0, 0), Action.code(Action.Type.TSUMO_AGARI));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> decisions.win(1));
  }

  @Test
  public void resolutionsKeepTheirOrder() {
    Resolution resolution = new Resolution(Variant.TENHOU_4P);
    resolution.add(5, true, Resolution.Role.CAUSAL);
    resolution.add(6, false, Resolution.Role.PREEMPTED);
    resolution.add(7, true, Resolution.Role.FORCED);
    Assert.assertEquals(resolution.size(), 3);
    Assert.assertEquals(resolution.id(1), 6);
    Assert.assertFalse(resolution.executed(1));
    Assert.assertEquals(resolution.role(1), Resolution.Role.PREEMPTED);
    Assert.assertEquals(resolution.role(2), Resolution.Role.FORCED);
    Assert.assertThrows(IllegalStateException.class, () -> resolution.add(8, true, Resolution.Role.CAUSAL));
    resolution.clear();
    Assert.assertEquals(resolution.size(), 0);
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> resolution.role(0));
  }

  @Test
  public void copiesEqualTheOriginalsAndStayIndependent() {
    Decisions decisions = new Decisions(Variant.TENHOU_4P);
    decisions.open(1, EngineDecisionKind.CHANKAN);
    decisions.add(RON);
    decisions.add(PASS);
    decisions.close(11);
    Decisions copy = new Decisions(Variant.TENHOU_4P);
    copy.copyFrom(decisions);
    Fields.assertSameFields(decisions, copy);
    copy.clear();
    Assert.assertEquals(decisions.size(), 1);
    Assert.assertEquals(decisions.legal(0, 0), RON);

    Resolution resolution = new Resolution(Variant.TENHOU_3P);
    resolution.add(3, false, Resolution.Role.PREEMPTED);
    Resolution resolutionCopy = new Resolution(Variant.TENHOU_3P);
    resolutionCopy.copyFrom(resolution);
    Fields.assertSameFields(resolution, resolutionCopy);
    resolutionCopy.clear();
    Assert.assertEquals(resolution.size(), 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> copy.copyFrom(new Decisions(Variant.TENHOU_3P)));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> resolutionCopy.copyFrom(new Resolution(Variant.TENHOU_4P)));
  }
}
