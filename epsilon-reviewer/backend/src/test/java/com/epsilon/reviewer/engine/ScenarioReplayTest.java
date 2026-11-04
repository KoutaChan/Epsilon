package com.epsilon.reviewer.engine;

import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.replay.RecordCompletion;
import com.epsilon.replay.RecordFormat;
import com.epsilon.replay.RecordMetadata;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.ReplayRecord;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.engine.scenario.DecisionLocator;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import java.nio.charset.StandardCharsets;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 変更のないもしもが、牌譜の全ての局の全ての席の全ての判断で、通常の解析と同じ入力を方策へ渡し、同じ候補と判断の時点の局面になることを
 * 確かめる。天鳳の牌譜は記録の牌山のある再生と、牌山を外した再生（mjai・雀魂の道）の両方を比べる。
 */
public class ScenarioReplayTest {
  @Test
  public void unchangedPositionGivesThePolicyTheReviewInputAtEveryDecision() {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    for (String name : GoldenRecords.NAMES) {
      LoadedRecord record = GoldenRecords.read(name);
      records.put(name, record);
      records.put(name + " without walls", GoldenRecords.withoutWalls(record));
    }
    records.put(
        "river.mjai",
        RecordLoader.load(AnalysisTest.riverFixture().getBytes(StandardCharsets.UTF_8)));
    records.put("chankan round", chankanRound());
    Map<EngineDecisionKind, Integer> kinds = new EnumMap<>(EngineDecisionKind.class);
    records.forEach((name, record) -> compareEveryDecision(name, record, kinds));
    Assert.assertEquals(kinds.keySet(), EnumSet.allOf(EngineDecisionKind.class), kinds.toString());
  }

  /**
   * 天鳳の牌譜にない道を通す作った局（mjai の順）。席 1 が席 0 の 5p をポンし、赤の 5p を加槓する。席 2 は 34p の 25p 待ちで、初めの 5p は
   * 役がなく同巡の振聴になり、加槓は槍槓できるが明示して見送る（応答のない席 3 も見送りを明示する）。槓ドラは嶺上ツモの打牌の後に開く。最後の
   * 9m の打牌への席 0 のポンと席 3 のチーは、牌譜が終わるので未観測の応答になる。
   */
  static LoadedRecord chankanRound() {
    int[][] hands = {
      {55, 17, 18, 32, 33, 36, 37, 84, 85, 108, 109, 112, 116},
      {53, 54, 72, 76, 80, 92, 96, 100, 60, 64, 68, 120, 121},
      {0, 1, 2, 4, 8, 12, 20, 24, 28, 104, 105, 44, 48},
      {9, 13, 19, 21, 25, 29, 41, 56, 77, 81, 86, 124, 132}
    };
    List<ReplayEvent> events =
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 0, Tile.HATSU, hands, new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(0, 129),
            new Dahai(0, 55, false),
            new Pon(1, 0, 55, new int[] {53, 54}),
            new Dahai(1, 120, false),
            new Tsumo(2, 49),
            new Dahai(2, 49, true),
            new Tsumo(3, 130),
            new Dahai(3, 130, true),
            new Tsumo(0, 131),
            new Dahai(0, 131, true),
            new Tsumo(1, Tile.AKA_P5_ID),
            new Kakan(1, Tile.AKA_P5_ID),
            new None(2),
            new None(3),
            new Tsumo(1, 133),
            new Dora(Tile.S1),
            new Dahai(1, 133, true),
            new Tsumo(2, 34),
            new Dahai(2, 34, true));
    return synthetic(events);
  }

  /** 確かめの再生をしない作った牌譜（mjai、途中で終わる）。 */
  static LoadedRecord synthetic(List<ReplayEvent> events) {
    List<String> names = List.of("A", "B", "C", "D");
    return new LoadedRecord(
        new RecordSummary("mjai", names, null, 1),
        new ReplayRecord(
            events,
            new RecordMetadata(RecordFormat.MJAI, names, null, "unavailable", null),
            null,
            RecordCompletion.INCOMPLETE),
        LoadedRecord.roundStarts(events));
  }

  private static void compareEveryDecision(
      String name, LoadedRecord record, Map<EngineDecisionKind, Integer> kinds) {
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
          ScenarioEngine.evaluate(
              record,
              new ScenarioQuery(
                  located.round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
              scenario);
      PolicyInput expected = review.inputs.get(decision++);
      Assert.assertEquals(scenario.inputs.size(), 1, where);
      Assert.assertNull(PolicyInput.difference(expected, scenario.inputs.getFirst()), where);
      Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet(), where);
      Assert.assertNull(answer.table(), where);
      Assert.assertEquals(
          DecisionLocator.find(record, located.round(), step.eventIndex(), step.actor()).table(),
          step.table(),
          where);
      kinds.merge(expected.kind(), 1, Integer::sum);
    }
    Assert.assertEquals(decision, review.inputs.size(), name);
  }
}
