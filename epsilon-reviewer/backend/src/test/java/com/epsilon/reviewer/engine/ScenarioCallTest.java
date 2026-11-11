package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 応答の判断で鳴いたことにするもしものテスト。牌譜と同じチー・ポンをしたもしもは、通常の解析で鳴いた後に打牌する判断と同じ O を方策に
 * 渡し、同じ候補と局面を返す（選んだ手の印は比べない）。応答でない判断、合法でない手、チー・ポン以外の応答は断る。
 */
public class ScenarioCallTest {
  /** 立直の宣言牌にチーかポンができる応答の判断がある半荘の、乱数の種。 */
  private static final long RIICHI_TILE_SEED = 11;

  @Test
  public void recordedCallMatchesTheReviewDiscardAfterTheCall() {
    int compared = 0;
    for (Map.Entry<String, LoadedRecord> entry : PlayedRecords.fourPlayer().entrySet()) {
      compared += compareRecordedCalls(entry.getKey(), entry.getValue());
      compared += compareRecordedCalls(entry.getKey() + " mjai", PlayedRecords.mjai(entry.getValue()));
    }
    Assert.assertTrue(compared > 0, "The fixtures have no recorded chi or pon.");
  }

  @Test
  public void callIsOnlyAChiOrPonAtAResponse() {
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
    List<Decision> decisions = decisions(record);
    Decision turn = first(decisions, EngineDecisionKind.TURN);
    Decision response = first(decisions, EngineDecisionKind.RESPONSE);
    int pass = Action.code(Action.Type.PASS);
    assertRefused(record, turn, turn.step().actionId(0), "call: the decision is not a response");
    assertRefused(record, response, pass, "call: only a chi or a pon");
    assertRefused(record, response, Integer.MAX_VALUE, "call: the call is not a legal response");
  }

  /**
   * 立直の宣言牌を鳴くと、鳴いた時に立直が成立し、供託が 1 本増えて、宣言した席の点が 1000 点減る。鳴いた後の供託か点が O に書ける範囲
   * （供託は 100 本まで、点は -3,276,800 点から）を外れる変更は断る。範囲に収まる変更は、鳴いた後の値で評価する。
   */
  @Test
  public void callingARiichiTileRefusesEditsTheDepositPutsOutOfRange() {
    int maxKyotaku = 100;
    int minScore = -3_276_800;
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, RIICHI_TILE_SEED);
    for (Decision response : decisions(record)) {
      TableState shown = response.step().table();
      int declarer = shown.activeSeat();
      List<RiverTile> river = shown.players().get(declarer).river();
      int call = call(response.step());
      if (response.kind() != EngineDecisionKind.RESPONSE || call < 0 || !river.getLast().riichi())
        continue;
      var policy = new RecordingPolicy();
      ScenarioAnswer answer =
          Scenarios.evaluate(record, query(response, kyotaku(maxKyotaku - 1), call), policy);
      Assert.assertEquals(answer.table().kyotaku(), maxKyotaku);
      Assert.assertEquals(policy.input(0).kyotaku(), maxKyotaku);
      assertRefused(
          record, response, kyotaku(maxKyotaku), call, "call: the riichi deposit at the call");
      Scenarios.evaluate(record, query(response, kyotaku(maxKyotaku), null), new RecordingPolicy());
      List<Integer> scores =
          new ArrayList<>(shown.players().stream().map(PlayerState::score).toList());
      scores.set(declarer, minScore + 1000);
      policy = new RecordingPolicy();
      answer = Scenarios.evaluate(record, query(response, scores(scores), call), policy);
      Assert.assertEquals(answer.table().players().get(declarer).score(), minScore);
      Assert.assertEquals(policy.input(0).scores().get(declarer).intValue(), minScore);
      scores.set(declarer, minScore);
      assertRefused(
          record, response, scores(scores), call, "call: the riichi deposit at the call");
      Scenarios.evaluate(record, query(response, scores(scores), null), new RecordingPolicy());
      return;
    }
    Assert.fail("The fixture has no chi or pon offered on a riichi declaration tile.");
  }

  /** 牌譜で鳴いた応答の判断ごとに、同じ鳴きをしたもしもと、通常の解析での次の判断（鳴いた後の打牌）を比べる。 */
  private static int compareRecordedCalls(String name, LoadedRecord record) {
    List<Decision> decisions = decisions(record);
    int compared = 0;
    for (int i = 0; i < decisions.size(); i++) {
      Decision response = decisions.get(i);
      if (response.kind() != EngineDecisionKind.RESPONSE) continue;
      String recorded = response.step().chosenType();
      if (!recorded.equals("CHI") && !recorded.equals("PON")) continue;
      // 他家のロンが優先されると鳴きは成り立たないので、次の判断が鳴いた後の打牌のときだけ比べる。
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
      ScenarioAnswer answer = Scenarios.evaluate(record, query(response, call), policy);
      Assert.assertEquals(policy.inputs.size(), 1, where);
      Assert.assertEquals(policy.input(0), discard.input(), where);
      JsonObject expected = discard.step().candidateSet();
      expected.remove("chosen");
      Assert.assertEquals(Format6.candidateSet(answer), expected, where);
      Assert.assertEquals(answer.table(), discard.step().table(), where);
      compared++;
    }
    return compared;
  }

  /** 同じ局で同じ席が次に行う判断。無ければ null。 */
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
    assertRefused(record, decision, TestEdits.NONE, call, message);
  }

  private static void assertRefused(
      LoadedRecord record, Decision decision, ScenarioEdits edits, int call, String message) {
    ReviewerException refused =
        Assert.expectThrows(
            ReviewerException.class,
            () -> Scenarios.evaluate(record, query(decision, edits, call), new RecordingPolicy()));
    Assert.assertEquals(refused.code(), ErrorCode.INVALID_EDIT);
    Assert.assertTrue(refused.getMessage().startsWith(message), refused.getMessage());
  }

  private static ScenarioQuery query(Decision decision, int call) {
    return query(decision, TestEdits.NONE, call);
  }

  private static ScenarioQuery query(Decision decision, ScenarioEdits edits, Integer call) {
    return new ScenarioQuery(
        decision.round(), decision.step().eventIndex(), decision.step().actor(), edits, call);
  }

  private static ScenarioEdits kyotaku(int kyotaku) {
    return new ScenarioEdits(null, null, null, null, null, kyotaku);
  }

  private static ScenarioEdits scores(List<Integer> scores) {
    return new ScenarioEdits(null, null, null, List.copyOf(scores), null, null);
  }

  /** 候補にある最初のチーかポンの行動の番号。無ければ -1。 */
  private static int call(Format6.Step step) {
    for (int i = 0; i < step.candidates().size(); i++)
      if (step.type(i) == Action.Type.CHI || step.type(i) == Action.Type.PON)
        return step.actionId(i);
    return -1;
  }

  private static Decision first(List<Decision> decisions, EngineDecisionKind kind) {
    return decisions.stream().filter(d -> d.kind() == kind).findFirst().orElseThrow();
  }

  /** 通常の解析の判断を、方策が受け取った O のフィールドと組にして、順に並べる。 */
  private static List<Decision> decisions(LoadedRecord record) {
    var policy = new RecordingPolicy();
    Format6.Document document = Format6.analyze(record, policy);
    List<Decision> decisions = new ArrayList<>();
    for (Format6.Located located : document.decisions())
      decisions.add(
          new Decision(located.round(), located.step(), policy.input(decisions.size())));
    Assert.assertEquals(decisions.size(), policy.inputs.size());
    return decisions;
  }

  private record Decision(int round, Format6.Step step, ObservedInput input) {
    EngineDecisionKind kind() {
      return input.kind();
    }
  }
}
