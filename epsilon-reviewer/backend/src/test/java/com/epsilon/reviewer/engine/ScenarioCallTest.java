package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 応答の判断で鳴いたとするもしもを確かめる。牌譜のとおりのチー・ポンなら、通常の解析の鳴いた後の打牌の判断と同じ入力を方策へ渡し、同じ候補
 * （選んだ手の印を除く）と局面を返す。応答でない判断・合法でない手・チーとポン以外の応答は断る。
 */
public class ScenarioCallTest {
  @Test
  public void recordedCallMatchesTheReviewDiscardAfterTheCall() {
    int compared = 0;
    for (String name : GoldenRecords.NAMES) {
      LoadedRecord record = GoldenRecords.read(name);
      compared += compareRecordedCalls(name, record);
      compared += compareRecordedCalls(name + " without walls", GoldenRecords.withoutWalls(record));
    }
    Assert.assertTrue(compared > 0, "The fixtures have no recorded chi or pon.");
  }

  @Test
  public void callIsOnlyAChiOrPonAtAResponse() {
    LoadedRecord record = GoldenRecords.read("kakan-a");
    List<Decision> decisions = decisions(record);
    Decision turn = first(decisions, EngineDecisionKind.TURN);
    Decision response = first(decisions, EngineDecisionKind.RESPONSE);
    int pass = Action.pass().toIndex();
    assertRefused(record, turn, turn.step().actionId(0), "call: the decision is not a response");
    assertRefused(record, response, pass, "call: only a chi or a pon");
    assertRefused(record, response, Integer.MAX_VALUE, "call: the call is not a legal response");
  }

  /** 牌譜で鳴いた応答の判断ごとに、同じ鳴きのもしもと、通常の解析の次の判断（鳴いた後の打牌）を比べる。 */
  private static int compareRecordedCalls(String name, LoadedRecord record) {
    List<Decision> decisions = decisions(record);
    int compared = 0;
    for (int i = 0; i < decisions.size(); i++) {
      Decision response = decisions.get(i);
      if (response.kind() != EngineDecisionKind.RESPONSE) continue;
      String recorded = response.step().chosenType();
      if (!recorded.equals("CHI") && !recorded.equals("PON")) continue;
      // 他家のロンが先に通った鳴きは成り立たないので、続く判断が鳴いた後の打牌のときだけ比べる。
      Decision discard = next(decisions, i);
      if (discard == null || discard.kind() != EngineDecisionKind.POST_CALL_DAHAI) continue;
      String where =
          name
              + " round "
              + response.round()
              + " event "
              + response.step().eventIndex()
              + " seat "
              + response.step().actor();
      var policy = new RecordingPolicy();
      int call = response.step().actionId(response.step().chosen());
      ScenarioAnswer answer = ScenarioEngine.evaluate(record, query(response, call), policy);
      Assert.assertEquals(policy.inputs.size(), 1, where);
      Assert.assertNull(PolicyInput.difference(discard.input(), policy.inputs.getFirst()), where);
      JsonObject expected = discard.step().candidateSet();
      expected.remove("chosen");
      Assert.assertEquals(Format6.candidateSet(answer), expected, where);
      Assert.assertEquals(answer.table(), discard.step().table(), where);
      compared++;
    }
    return compared;
  }

  /** 同じ局の同じ席の次の判断。 */
  private static Decision next(List<Decision> decisions, int index) {
    Decision from = decisions.get(index);
    for (int i = index + 1; i < decisions.size(); i++) {
      Decision later = decisions.get(i);
      if (later.round() != from.round()) return null;
      if (later.step().actor() == from.step().actor()) return later;
    }
    return null;
  }

  private static void assertRefused(
      LoadedRecord record, Decision decision, int call, String message) {
    ReviewerException refused =
        Assert.expectThrows(
            ReviewerException.class,
            () -> ScenarioEngine.evaluate(record, query(decision, call), new RecordingPolicy()));
    Assert.assertEquals(refused.code(), ErrorCode.INVALID_EDIT);
    Assert.assertTrue(refused.getMessage().startsWith(message), refused.getMessage());
  }

  private static ScenarioQuery query(Decision decision, int call) {
    return new ScenarioQuery(
        decision.round(),
        decision.step().eventIndex(),
        decision.step().actor(),
        TestEdits.NONE,
        call);
  }

  private static Decision first(List<Decision> decisions, EngineDecisionKind kind) {
    return decisions.stream().filter(d -> d.kind() == kind).findFirst().orElseThrow();
  }

  /** 通常の解析の判断を、方策が受け取った入力と組にして順に並べる。 */
  private static List<Decision> decisions(LoadedRecord record) {
    var policy = new RecordingPolicy();
    Format6.Document document = Format6.analyze(record, policy);
    List<Decision> decisions = new ArrayList<>();
    for (Format6.Located located : document.decisions())
      decisions.add(
          new Decision(located.round(), located.step(), policy.inputs.get(decisions.size())));
    Assert.assertEquals(decisions.size(), policy.inputs.size());
    return decisions;
  }

  private record Decision(int round, Format6.Step step, PolicyInput input) {
    EngineDecisionKind kind() {
      return input.kind();
    }
  }
}
