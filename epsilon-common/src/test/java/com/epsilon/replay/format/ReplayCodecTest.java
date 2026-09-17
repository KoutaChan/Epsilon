package com.epsilon.replay.format;

import com.epsilon.client.tenhou.TenhouEvent;
import com.epsilon.client.tenhou.TenhouMessageParser;
import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.google.gson.*;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 生牌譜の所有権・物理 ID・記録された完了状態を検証する。 */
public class ReplayCodecTest {
  @Test
  public void offlineDrawRetainsThePhysicalTileWhileLiveInputHidesIt() {
    var live = (TenhouEvent.Draw) TenhouMessageParser.parse("<U52/>");
    var events = TenhouXmlReader.readRecord(xml("<U52/><E52/>")).events();
    var draw = (Tsumo) events.get(2);
    var discard = (Dahai) events.get(3);
    Assert.assertEquals(live.physicalTileId(), -1);
    Assert.assertEquals(draw.actor(), 1);
    Assert.assertEquals(draw.physicalTileId(), 52);
    Assert.assertEquals(discard.physicalTileId(), 52);
    Assert.assertTrue(discard.tsumogiri());
  }

  @Test
  public void mjaiDirectReadAllocatesDistinctNormalTilesAndPreservesCalledRedAndAddedDraw() {
    JsonObject initial = initialMjai();
    JsonArray hand = initial.getAsJsonArray("tehais").get(2).getAsJsonArray();
    hand.set(0, new JsonPrimitive("5p"));
    hand.set(1, new JsonPrimitive("5p"));
    String body =
        initial
            + "\n"
            + """
            {"type":"tsumo","actor":0,"pai":"5pr"}
            {"type":"dahai","actor":0,"pai":"5pr","tsumogiri":true}
            {"type":"pon","actor":2,"target":0,"pai":"5pr","consumed":["5p","5p"]}
            {"type":"tsumo","actor":2,"pai":"5p"}
            {"type":"kakan","actor":2,"pai":"5p"}
            """;
    var events = MjaiReader.readRecord(body).events();
    var pon = (Pon) events.get(4);
    var draw = (Tsumo) events.get(5);
    var kakan = (Kakan) events.get(6);
    Assert.assertEquals(pon.target(), 0);
    Assert.assertEquals(pon.calledPhysicalTileId(), Tile.AKA_P5_ID);
    Assert.assertNotEquals(pon.consumedPhysicalTileIds()[0], pon.consumedPhysicalTileIds()[1]);
    for (int id : pon.consumedPhysicalTileIds()) {
      Assert.assertEquals(Tile.typeOf(id), Tile.P5);
      Assert.assertFalse(Tile.isAka(id));
      Assert.assertNotEquals(draw.physicalTileId(), id);
    }
    Assert.assertEquals(kakan.addedPhysicalTileId(), draw.physicalTileId());
  }

  @DataProvider
  public Object[][] mjaiNineTerminalsReasons() {
    return new Object[][] {{"kyushu_kyuhai"}, {"kyushukyuhai"}, {"yao9"}};
  }

  @Test(dataProvider = "mjaiNineTerminalsReasons")
  public void mjaiNineTerminalsEmitsTheRecordedChoiceAtTheFinalDraw(String reason) {
    // 南4局の実際の手牌を使い、流局直前の判断が解析へ通知されることを確認する。
    String input =
        """
        {"type":"start_game"}
        {"type":"start_kyoku","bakaze":"S","kyoku":4,"honba":0,"kyotaku":0,"oya":3,"dora_marker":"3s","scores":[6000,34500,14600,44900],"tehais":[["2m","3m","3p","6p","7p","7p","7p","8p","4s","8s","S","F","F"],["5m","9m","1p","1p","5p","6p","6p","2s","5s","6s","N","N","P"],["1m","2m","4m","4m","7m","9m","9m","6s","7s","9s","W","P","F"],["7m","9m","4p","5p","9p","1s","9s","E","W","W","N","P","C"]]}
        {"type":"tsumo","actor":3,"pai":"7m"}
        {"type":"ryukyoku","reason":"%s","deltas":[0,0,0,0]}
        {"type":"end_kyoku"}
        {"type":"end_game"}
        """
            .formatted(reason);
    ReplayRecord record = MjaiReader.readRecord(input);
    var choices = new ArrayList<Action>();
    int[] scores =
        ReplayEngine.replay(
            record,
            (point, state, player, legal) -> {
              choices.add(legal.get(point.chosenSlot()));
              Assert.assertEquals(player, 3);
              Assert.assertEquals(point.choiceKind(), DecisionPoint.ChoiceKind.RECORDED_ACTION);
              Assert.assertEquals(point.eventIndex(), 3);
              Assert.assertEquals(point.causeEventIndex(), 2);
              Assert.assertEquals(state.hand(player).concealedTileCount(), 14);
              Assert.assertTrue(
                  legal.stream().anyMatch(action -> action.type() == Action.Type.DAHAI));
            });
    Assert.assertEquals(choices, List.of(Action.kyushuKyuhai()));
    Assert.assertEquals(scores, new int[] {6000, 34500, 14600, 44900});
  }

  @Test
  public void aHandDiscardCannotSilentlyConsumeItsOnlyMatchingDraw() {
    String record =
        initialMjai()
            + "\n"
            + """
            {"type":"tsumo","actor":0,"pai":"5pr"}
            {"type":"dahai","actor":0,"pai":"5pr","tsumogiri":false}
            """;
    Assert.expectThrows(IllegalArgumentException.class, () -> MjaiReader.readRecord(record));
  }

  @Test
  public void explicitStartGameIsNotDuplicatedAndSecondStartGameIsRejected() {
    JsonArray input = new JsonArray();
    input.add(
        JsonParser.parseString("{\"type\":\"start_game\",\"names\":[\"A\",\"B\",\"C\",\"D\"]}"));
    input.add(initialMjai());
    input.add(JsonParser.parseString("{\"type\":\"end_game\",\"scores\":null}"));
    ReplayRecord record = MjaiReader.readRecord(input.toString());
    Assert.assertEquals(record.events().stream().filter(StartGame.class::isInstance).count(), 1L);
    Assert.assertEquals(record.metadata().names(), List.of("A", "B", "C", "D"));
    Assert.assertEquals(record.completion(), RecordCompletion.COMPLETE);
    Assert.assertNull(record.metadata().finalScores());
    input.add(JsonParser.parseString("{\"type\":\"start_game\"}"));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> MjaiReader.readRecord(input.toString()));
  }

  @Test
  public void missingRequiredMjaiScoresFailWithAnEnglishFieldDiagnostic() {
    JsonObject initial = initialMjai();
    initial.remove("scores");
    var failure =
        Assert.expectThrows(
            IllegalArgumentException.class, () -> MjaiReader.readRecord(initial.toString()));
    Assert.assertTrue(failure.getMessage().contains("scores"));
    Assert.assertTrue(failure.getMessage().chars().allMatch(c -> c < 128));
  }

  @Test
  public void tenhouOnePassPreservesNamesRulesAndFinalPointsInsteadOfUma() {
    String record =
        xml(
            "<RYUUKYOKU type='yao9' sc='250,0,250,0,250,0,250,0' "
                + "owari='260,36.0,250,5.0,250,-15.0,240,-26.0'/>");
    record =
        record.replace(
            "<mjloggm>",
            "<mjloggm><GO type='169'/><UN n0='%E6%9D%B1' n1='A&amp;B' n2='C' n3='D'/>");
    ReplayRecord result = TenhouXmlReader.readRecord(record);
    Assert.assertEquals(result.metadata().names(), List.of("東", "A&B", "C", "D"));
    Assert.assertEquals(result.metadata().ruleDescription(), "Tenhou type=169");
    Assert.assertEquals(result.metadata().finalScores(), new int[] {26000, 25000, 25000, 24000});
    Assert.assertEquals(result.completion(), RecordCompletion.COMPLETE);
    Assert.assertEquals(((Ryukyoku) result.events().get(2)).reason(), "NINE_TERMINALS");
    Assert.assertEquals(
        ((EndGame) result.events().getLast()).scores(), result.metadata().finalScores());
  }

  @Test
  public void eofNeverInventsFinalScoresOrEndGameEvenAfterACompletedRound() {
    ReplayRecord record =
        TenhouXmlReader.readRecord(xml("<RYUUKYOKU sc='250,0,250,0,250,0,250,0'/>"));
    Assert.assertEquals(record.completion(), RecordCompletion.INCOMPLETE);
    Assert.assertNull(record.metadata().finalScores());
    Assert.assertTrue(record.events().getLast() instanceof EndKyoku);
    Assert.assertEquals(record.events().stream().filter(EndGame.class::isInstance).count(), 0L);
    ReplayRecord mjai = MjaiReader.readRecord(initialMjai().toString());
    Assert.assertEquals(mjai.completion(), RecordCompletion.INCOMPLETE);
    Assert.assertNull(mjai.metadata().finalScores());
  }

  @Test
  public void doubleRonKeepsBothDeltasBeforeTheSingleRoundBoundary() {
    var record =
        TenhouXmlReader.readRecord(
            xml(
                """
                <AGARI who="1" fromWho="0" sc="250,-20,250,20,250,0,250,0"/>
                <AGARI who="2" fromWho="0" sc="250,-30,250,0,250,30,250,0"/>
                """));
    var wins =
        record.events().stream().filter(Hora.class::isInstance).map(Hora.class::cast).toList();
    Assert.assertEquals(wins.get(0).deltas(), new int[] {-2000, 2000, 0, 0});
    Assert.assertEquals(wins.get(1).deltas(), new int[] {-3000, 0, 3000, 0});
    Assert.assertEquals(record.events().stream().filter(EndKyoku.class::isInstance).count(), 1L);
    Assert.assertTrue(record.events().getLast() instanceof EndKyoku);
  }

  @Test
  public void tenhouMeldsPreservePhysicalIdsAndAbsoluteSources() {
    var events =
        TenhouXmlReader.readRecord(xml("<N who='3' m='6154'/><N who='3' m='6162'/>")).events();
    var pon = (Pon) events.get(2);
    var kakan = (Kakan) events.get(3);
    Assert.assertEquals(pon.actor(), 3);
    Assert.assertEquals(pon.target(), 1);
    Assert.assertEquals(pon.calledPhysicalTileId(), 17);
    Assert.assertEquals(pon.consumedPhysicalTileIds(), new int[] {18, 19});
    Assert.assertEquals(kakan.addedPhysicalTileId(), 16);
  }

  @Test
  public void malformedXmlAndDtdAreRejectedAtTheRecordBoundary() {
    Assert.expectThrows(
        IllegalArgumentException.class, () -> TenhouXmlReader.readRecord("<mjloggm>"));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> TenhouXmlReader.readRecord("<!DOCTYPE mjloggm [<!ENTITY x 'x'>]>" + xml("")));
  }

  private static JsonObject initialMjai() {
    JsonObject initial =
        JsonParser.parseString(
                """
                {"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,
                "oya":0,"dora_marker":"C","scores":[25000,25000,25000,25000]}
                """)
            .getAsJsonObject();
    JsonArray hands = new JsonArray();
    for (int seat = 0; seat < 4; seat++) {
      JsonArray hand = new JsonArray();
      for (int i = 0; i < 13; i++) hand.add(TileNotation.formatPhysicalTile(seat * 13 + i));
      hands.add(hand);
    }
    initial.add("tehais", hands);
    return initial;
  }

  private static String xml(String body) {
    StringBuilder result =
        new StringBuilder("<mjloggm><INIT seed='0,0,0,0,0,132' ten='250,250,250,250' oya='0'");
    for (int seat = 0; seat < 4; seat++) {
      result.append(" hai").append(seat).append("='");
      for (int i = 0; i < 13; i++) {
        if (i > 0) result.append(',');
        result.append(seat * 13 + i);
      }
      result.append("'");
    }
    return result.append("/>").append(body).append("</mjloggm>").toString();
  }
}
