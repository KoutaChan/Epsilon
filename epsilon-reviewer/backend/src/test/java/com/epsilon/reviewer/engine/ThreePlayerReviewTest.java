package com.epsilon.reviewer.engine;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの牌譜の読み込み・解析・もしものテスト。牌譜は、エンジンで打った三人打ちの半荘と、それを mjai に書いて読み直した同じ半荘を
 * 使う。自己対局では起きにくい槍槓と抜いた北へのロンは、牌山を手で並べた局で足す。解析の文書は 3 席の局面と北抜きの手順（kita）を持つ。
 * 抜いた北は副露の後に並べる。北抜きの手と、抜いた北への応答も評価する。変更の無いもしもは、全ての判断で解析と同じ O と候補になる。
 * 四人打ちのモデルは三人打ちの牌譜を読めない。ルールベースの打ち手は三人打ちの牌譜を解析できる。
 */
public class ThreePlayerReviewTest {
  private static final Variant VARIANT = Variant.TENHOU_3P;

  @Test
  public void threePlayerTenhouAndMjaiRecordsLoadWithThreeSeats() {
    for (Map.Entry<String, LoadedRecord> entry : PlayedRecords.threePlayer().entrySet()) {
      String name = entry.getKey();
      LoadedRecord tenhou = entry.getValue();
      Assert.assertEquals(tenhou.game().variant(), VARIANT, name);
      Assert.assertEquals(tenhou.summary().source(), "tenhou", name);
      Assert.assertEquals(tenhou.summary().names().size(), 3, name);
      Assert.assertEquals(tenhou.summary().finalScores().size(), 3, name);
      LoadedRecord mjai = PlayedRecords.mjai(tenhou);
      Assert.assertEquals(mjai.game().variant(), VARIANT, name);
      Assert.assertEquals(mjai.summary().source(), "mjai", name);
      Assert.assertEquals(mjai.summary().finalScores(), tenhou.summary().finalScores(), name);
      Assert.assertEquals(mjai.summary().roundCount(), tenhou.summary().roundCount(), name);
    }
  }

  @Test
  public void theDocumentShowsThreeSeatsNukiAndTheirDecisions() {
    int kita = 0;
    int nukiCandidates = 0;
    int nukiResponses = 0;
    Map<String, LoadedRecord> records = new LinkedHashMap<>(PlayedRecords.threePlayer());
    records.putAll(PlayedRecords.threePlayerScenes());
    for (Map.Entry<String, LoadedRecord> entry : records.entrySet()) {
      String name = entry.getKey();
      Format6.Document document = Format6.analyze(entry.getValue(), new RecordingPolicy(VARIANT));
      for (Format6.Round round : document.rounds()) {
        Map<Integer, String> events = new HashMap<>();
        for (Format6.Step step : round.steps()) {
          Assert.assertEquals(step.table().players().size(), 3, name);
          if (!step.decision()) {
            events.put(step.eventIndex(), step.event());
            if (step.event().equals("kita")) {
              kita++;
              Assert.assertTrue(nukiShown(step.table().players().get(step.actor())), name);
            }
            continue;
          }
          nukiCandidates += step.offers(Action.Type.NUKI) ? 1 : 0;
          if ("kita".equals(events.get(step.cause()))) {
            nukiResponses++;
            Assert.assertTrue(step.offers(Action.Type.PASS), name);
          }
        }
      }
    }
    Assert.assertTrue(kita > 0, "no kita step");
    Assert.assertTrue(nukiCandidates > 0, "no nuki candidate");
    Assert.assertTrue(nukiResponses > 0, "no response to a nuki");
  }

  @Test
  public void unchangedScenarioGivesThePolicyTheReviewInputAtEveryDecision() {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    PlayedRecords.threePlayer()
        .forEach(
            (name, record) -> {
              records.put(name, record);
              records.put(name + " mjai", PlayedRecords.mjai(record));
            });
    records.putAll(PlayedRecords.threePlayerScenes());
    Set<EngineDecisionKind> kinds = EnumSet.noneOf(EngineDecisionKind.class);
    records.forEach((name, record) -> compareEveryDecision(name, record, kinds));
    Assert.assertEquals(kinds, EnumSet.allOf(EngineDecisionKind.class));
  }

  @Test
  public void aFourPlayerModelCannotReadAThreePlayerRecord() {
    LoadedRecord record = PlayedRecords.read(VARIANT, 0);
    ReviewerException analysis =
        Assert.expectThrows(
            ReviewerException.class, () -> Format6.analyze(record, new RecordingPolicy()));
    Assert.assertEquals(analysis.code(), ErrorCode.MODEL_UNAVAILABLE);
    Format6.Located first =
        Format6.analyze(record, new RecordingPolicy(VARIANT)).decisions().getFirst();
    ReviewerException scenario =
        Assert.expectThrows(
            ReviewerException.class,
            () ->
                Scenarios.evaluate(
                    record,
                    new ScenarioQuery(
                        first.round(),
                        first.step().eventIndex(),
                        first.step().actor(),
                        TestEdits.NONE,
                        null),
                    new RecordingPolicy()));
    Assert.assertEquals(scenario.code(), ErrorCode.MODEL_UNAVAILABLE);
  }

  /**
   * ルールベースの打ち手は、全ての判断で選んだ手の確率を 1 にする。北を抜けて、ツモ和了はできない手番では北を抜く。もしもでも同じ答えに
   * なる。
   */
  @Test
  public void theRuleBotAnalysesAThreePlayerRecord() throws IOException {
    LoadedRecord record = PlayedRecords.read(VARIANT, 0);
    try (Policy bot =
        ModelFamily.of(Series.RULE_BOT).open(null, VARIANT, PolicyOptions.defaults())) {
      ReviewModel model = model(bot);
      List<Format6.Located> decisions = Format6.analyze(record, model).decisions();
      Assert.assertFalse(decisions.isEmpty());
      int nuki = 0;
      for (Format6.Located located : decisions) {
        Format6.Step step = located.step();
        Assert.assertEquals(step.candidates().get(0).getAsJsonArray().get(1).getAsFloat(), 1f);
        if (step.offers(Action.Type.NUKI) && !step.offers(Action.Type.TSUMO_AGARI)) {
          nuki++;
          Assert.assertEquals(step.type(0), Action.Type.NUKI);
        }
      }
      Assert.assertTrue(nuki > 0, "no nuki decision");
      Format6.Step step = decisions.getFirst().step();
      ScenarioAnswer answer =
          Scenarios.evaluate(
              record,
              new ScenarioQuery(
                  decisions.getFirst().round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
              model);
      Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet());
    }
  }

  private static void compareEveryDecision(
      String name, LoadedRecord record, Set<EngineDecisionKind> kinds) {
    var review = new RecordingPolicy(VARIANT);
    Format6.Document document = Format6.analyze(record, review);
    int decision = 0;
    for (Format6.Located located : document.decisions()) {
      Format6.Step step = located.step();
      String where =
          name + " round " + located.round() + " event " + step.eventIndex() + " seat " + step.actor();
      var scenario = new RecordingPolicy(VARIANT);
      ScenarioAnswer answer =
          Scenarios.evaluate(
              record,
              new ScenarioQuery(located.round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
              scenario);
      Assert.assertEquals(scenario.inputs.size(), 1, where);
      Assert.assertEquals(scenario.inputs.getFirst(), review.inputs.get(decision), where);
      Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet(), where);
      Assert.assertEquals(
          Scenarios.find(record, located.round(), step.eventIndex(), step.actor()).table(),
          step.table(),
          where);
      kinds.add(review.input(decision).kind());
      decision++;
    }
    Assert.assertEquals(decision, review.inputs.size(), name);
  }

  private static boolean nukiShown(PlayerState player) {
    for (Meld meld : player.melds())
      if (meld.type().equals("NUKI") && meld.tiles().equals(List.of("N"))) return true;
    return false;
  }

  /** 方策の出力をそのまま評価に使うモデル（価値は推定しない）。 */
  private static ReviewModel model(Policy policy) {
    return new ReviewModel() {
      @Override
      public Evaluation evaluate(Observation observation) {
        return policy.evaluate(observation);
      }

      @Override
      public boolean valued() {
        return false;
      }

      @Override
      public double temperature() {
        return FixtureFamily.TEMPERATURE;
      }

      @Override
      public Variant variant() {
        return VARIANT;
      }
    };
  }
}
