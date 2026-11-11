package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 変更の無いもしものテスト。牌譜の全ての判断（全ての局、全ての席）で、通常の解析と同じ O を方策に渡し、同じ候補と判断の時点の局面を
 * 返すことを確かめる。天鳳の XML から読んだ牌譜は、mjai に書いて読み直した同じ半荘（物理牌の割り当てが違う）とも比べる。
 */
public class ScenarioReplayTest {
  @Test
  public void unchangedPositionGivesThePolicyTheReviewInputAtEveryDecision() {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    PlayedRecords.fourPlayer()
        .forEach(
            (name, record) -> {
              records.put(name, record);
              records.put(name + " mjai", PlayedRecords.mjai(record));
            });
    records.put("chankan round", TestRounds.chankanRound());
    Set<EngineDecisionKind> kinds = EnumSet.noneOf(EngineDecisionKind.class);
    records.forEach((name, record) -> compareEveryDecision(name, record, kinds));
    Assert.assertEquals(
        kinds,
        EnumSet.of(
            EngineDecisionKind.TURN,
            EngineDecisionKind.RESPONSE,
            EngineDecisionKind.CHANKAN,
            EngineDecisionKind.POST_CALL_DAHAI));
  }

  private static void compareEveryDecision(
      String name, LoadedRecord record, Set<EngineDecisionKind> kinds) {
    var review = new RecordingPolicy();
    Format6.Document document = Format6.analyze(record, review);
    int decision = 0;
    for (Format6.Located located : document.decisions()) {
      Format6.Step step = located.step();
      String where =
          name
              + " round "
              + located.round()
              + " event "
              + step.eventIndex()
              + " seat "
              + step.actor();
      var scenario = new RecordingPolicy();
      ScenarioAnswer answer =
          Scenarios.evaluate(
              record,
              new ScenarioQuery(
                  located.round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
              scenario);
      Assert.assertEquals(scenario.inputs.size(), 1, where);
      Assert.assertEquals(
          scenario.input(0), review.input(decision), where + ": the fields of the O");
      Assert.assertEquals(scenario.inputs.getFirst(), review.inputs.get(decision), where);
      Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet(), where);
      Assert.assertNull(answer.table(), where);
      Assert.assertEquals(
          Scenarios.find(record, located.round(), step.eventIndex(), step.actor()).table(),
          step.table(),
          where);
      kinds.add(review.input(decision).kind());
      decision++;
    }
    Assert.assertEquals(decision, review.inputs.size(), name);
  }
}
