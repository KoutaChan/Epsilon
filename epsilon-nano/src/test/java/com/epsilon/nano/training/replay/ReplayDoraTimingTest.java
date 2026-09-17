package com.epsilon.nano.training.replay;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.format.TenhouXmlReader;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 槓後のチーや暗黙の見送りで、ドラ表示の時点が牌譜再生に正しく反映されることを検証する。 */
public class ReplayDoraTimingTest {
  @DataProvider
  public Object[][] kanDoraOrder() {
    return new Object[][] {
      {"<U57/><E57/><DORA hai=\"58\"/>", 1},
      {"<DORA hai=\"58\"/><U57/><E57/>", 2}
    };
  }

  @Test(dataProvider = "kanDoraOrder")
  public void chiAfterKanUsesRevealedDoraWithoutRewritingEarlierDiscard(
      String rinshanEvents, int discardDoraCount) {
    String xml =
        """
        <mjloggm ver="2.3">
        <INIT seed="3,0,0,0,0,85" ten="237,218,282,263" oya="3"
          hai0="4,8,24,32,40,41,48,89,90,100,116,120,128"
          hai1="0,5,6,20,44,49,64,92,93,108,112,113,124"
          hai2="1,21,25,36,53,56,72,76,84,104,114,121,125"
          hai3="2,9,12,33,37,45,60,80,81,82,117,126,129"/>
        <W34/><G117/><T50/><D116/><U22/><E0/><V54/><F114/>
        <N who="1" m="44137"/><E64/><V118/><F118/><W127/><G2/>
        <N who="0" m="23"/><D100/><U10/><E108/><V109/><F109/>
        <W7/><G60/><T119/><D119/><U68/><E68/><V61/><F121/><W94/><G94/>
        <N who="1" m="36458"/><E124/><N who="3" m="47658"/><G129/>
        <T17/><D120/><U132/><E132/><V18/><F125/><W133/><G133/><T73/><D73/>
        <U28/><E28/><N who="2" m="17455"/><F1/><N who="3" m="239"/><G37/>
        <T74/><D74/><U95/><N who="1" m="36466"/>
        """
            + rinshanEvents
            + "<N who=\"2\" m=\"34991\"/><F36/><RYUUKYOKU sc=\"237,0,218,0,282,0,263,0\""
            + " owari=\"237,0,218,0,282,0,263,0\"/></mjloggm>";
    List<EpsilonDecisionSample> samples =
        EpsilonReplaySampleCollector.collectDecisionSamples(TenhouXmlReader.readRecord(xml));
    EpsilonDecisionSample rinshanDiscard =
        samples.stream()
            .filter(sample -> sample.input().playerSeat(0) == 1)
            .filter(
                sample -> sample.chosenActionId() == Action.dahai(Tile.P6, false, true).toIndex())
            .toList()
            .getLast();
    EpsilonDecisionSample chi =
        samples.stream()
            .filter(sample -> sample.input().playerSeat(0) == 2)
            .filter(sample -> Action.fromIndex(sample.chosenActionId()).type() == Action.Type.CHI)
            .toList()
            .getLast();

    Assert.assertEquals(
        rinshanDiscard.input().roundCategory(0, DecisionInputSchema.RoundInt.DORA_INDICATOR_COUNT),
        discardDoraCount);
    Assert.assertEquals(
        chi.input().roundCategory(0, DecisionInputSchema.RoundInt.DORA_INDICATOR_COUNT), 2);
  }

  @Test
  public void implicitPassAfterDoraRevealIncludesTheNewIndicator() {
    int[][] hands = {
      {24, 26, 3, 5, 9, 13, 17, 21, 28, 32, 36, 56, 60},
      {0, 1, 2, 4, 8, 12, 40, 44, 48, 76, 80, 84, 124},
      {6, 10, 14, 18, 22, 29, 33, 37, 41, 45, 49, 53, 57},
      {7, 11, 15, 19, 23, 30, 34, 38, 42, 46, 50, 54, 58}
    };
    List<ReplayEvent> events =
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 1, Tile.HAKU, hands, new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(1, 25),
            new Dahai(1, 25, true),
            new Dora(Tile.P6),
            new Tsumo(2, 27),
            new ReplayEvent.Ryukyoku(new int[] {0, 0, 0, 0}),
            new ReplayEvent.EndKyoku(),
            new ReplayEvent.EndGame(new int[] {25000, 25000, 25000, 25000}));
    List<EpsilonDecisionSample> passes =
        EpsilonReplaySampleCollector.collectDecisionSamples(
                new ReplayRecord(
                    events,
                    new RecordMetadata(
                        RecordFormat.TENHOU,
                        List.of("", "", "", ""),
                        new int[] {25000, 25000, 25000, 25000},
                        "fixture",
                        null),
                    null,
                    RecordCompletion.COMPLETE))
            .stream()
            .filter(sample -> Action.fromIndex(sample.chosenActionId()).type() == Action.Type.PASS)
            .toList();

    Assert.assertFalse(passes.isEmpty());
    for (EpsilonDecisionSample pass : passes) {
      Assert.assertEquals(
          pass.input().roundCategory(0, DecisionInputSchema.RoundInt.DORA_INDICATOR_COUNT), 2);
    }
  }
}
