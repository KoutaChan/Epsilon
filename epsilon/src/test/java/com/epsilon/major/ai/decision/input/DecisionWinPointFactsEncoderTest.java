package com.epsilon.major.ai.decision.input;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.scoring.YakuBits;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.engine.VisibleHandScoreBuffer;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 高符の待ち・和了入力を、点数と既存の符コードを保ったまま符号化する。 */
public class DecisionWinPointFactsEncoderTest {
  @Test
  public void threeKansProduce120FuWithoutChangingScoring() {
    Hand hand = new Hand();
    hand.addMeld(Meld.ankan(0)); // 一萬
    hand.addMeld(Meld.ankan(8)); // 九萬
    hand.addMeld(Meld.ankan(10)); // 二筒
    for (int tile : new int[] {21, 23, Tile.HAKU, Tile.HAKU}) hand.add(tile);
    GameState state = new GameState(17);
    state.startRound(0, 0, 0, 0);
    HandScoreBuffer score = new HandScoreBuffer();
    Assert.assertTrue(
        new HandScoreEvaluator()
            .scoreAfterAdding(
                hand,
                22, // 嵌五索
                WinMethod.RON,
                WinConditions.publicDecision(Tile.NAN, Tile.TON, RiichiState.NONE),
                state.doraState(),
                false,
                0,
                score));
    Assert.assertEquals(score.fu(), 120);
    Assert.assertEquals(score.han(), 4);
    assertFacts(score.yakuBits(), score.han(), score.fu(), 11);
    Assert.assertEquals(score.fu(), 120);
  }

  @Test
  public void highFuPreservesBasePointsIncludingAddedDora() {
    for (int fu : new int[] {120, 130, 140}) {
      for (int han = 3; han <= 20; han++) {
        assertFacts(ScoringYaku.RIICHI.bit(), han, fu, 11);
        for (int addedHan = 0; addedHan <= 10; addedHan++) {
          Assert.assertEquals(
              ScoreMath.normalBasePoints(han + addedHan, 110),
              ScoreMath.normalBasePoints(han + addedHan, fu));
        }
      }
    }
  }

  @Test
  public void existingCodesAndYakumanRemainUnchanged() {
    for (int code = 0; code < 12; code++) {
      for (int han = 0; han <= 20; han++) {
        assertFacts(ScoringYaku.RIICHI.bit(), han, ScoreMath.decodeFuCode(code), code);
      }
    }
    assertFacts(ScoringYaku.SUUANKOU.bit(), 0, 0, 0);
    assertFacts(0L, 0, 0, 0);
  }

  @Test
  public void doesNotSilentlyChangeLowHanPointsOrAcceptMalformedFu() {
    for (int han : new int[] {1, 2}) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> assertFacts(ScoringYaku.RIICHI.bit(), han, 120, 11));
    }
    Assert.expectThrows(
        IllegalArgumentException.class, () -> assertFacts(ScoringYaku.RIICHI.bit(), 4, 121, 11));
  }

  private static void assertFacts(long bits, int han, int fu, int expectedCode) {
    DecisionHostInputs inputs = new DecisionHostInputs(1, new DecisionBucket(1, 1));
    DecisionInputWriter writer = new DecisionInputWriter(inputs, 0);
    var score =
        new VisibleHandScoreBuffer(
            bits, han, fu, ScoreMath.basePoints(han, fu, YakuBits.yakumanCount(bits)));
    DecisionWinPointFactsEncoder.encodeAction(score, true, writer, 0);
    assertStored(inputs, DecisionInputLayout.Tensor.ACTION_WIN_FACTS, 0, bits, han, expectedCode);
    for (var type : DecisionInputSchema.WaitWinType.values()) {
      DecisionWinPointFactsEncoder.encodeWait(bits, han, fu, true, type, writer, 0, 0, 0);
      assertStored(
          inputs,
          DecisionInputLayout.Tensor.WAIT_WIN_FACTS,
          type.ordinal() * DecisionInputSchema.ACTION_WIN_FACT_STRIDE,
          bits,
          han,
          expectedCode);
    }
    Assert.assertEquals(score.fu(), fu);
  }

  private static void assertStored(
      DecisionHostInputs inputs,
      DecisionInputLayout.Tensor tensor,
      int offset,
      long bits,
      int han,
      int code) {
    int base = inputs.layout().region(tensor).rowOffset(0) + offset;
    short[] values = inputs.stateCategories();
    int[] expected = {bits != 0 ? 1 : 0, han, code, YakuBits.yakumanCount(bits), 1};
    for (int field = 0; field < expected.length; field++) {
      Assert.assertEquals((int) values[base + field], expected[field]);
    }
  }
}
