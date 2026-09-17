package com.epsilon.replay.format;

import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.WinDetails;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜に記録された採点と裏ドラを、再計算せずに保持する。 */
public class WinDetailsCodecTest {
  @Test
  public void tenhouRonPreservesFuPointsAndUraEvenWhenNoUraBonusWasAwarded() {
    var details =
        tenhouWin(
                """
                <AGARI who="0" fromWho="1" machi="0" ten="30,5800,0"
                  yaku="1,1,7,1,52,1,53,0" doraHai="79" doraHaiUra="46"/>
                """)
            .details();
    Assert.assertEquals(details.han(), Integer.valueOf(3));
    Assert.assertEquals(details.fu(), Integer.valueOf(30));
    Assert.assertEquals(details.points(), Integer.valueOf(5800));
    Assert.assertEquals(details.yakuman(), Integer.valueOf(0));
    Assert.assertEquals(details.uraIndicators(), List.of("3p"));
    Assert.assertEquals(
        details.yaku(),
        List.of(
            new WinDetails.Yaku("RIICHI", 1, 0),
            new WinDetails.Yaku("PINFU", 1, 0),
            new WinDetails.Yaku("DORA", 1, 0)));
  }

  @Test
  public void tenhouDoubleRonKeepsEachWinnersScoringSeparate() {
    var events =
        TenhouXmlReader.readRecord(
                """
                <mjloggm>\
                """
                    + initialXml()
                    + """
                      <AGARI who="1" fromWho="0" ten="40,7700,0" yaku="1,1,54,1,53,1"
                        doraHaiUra="3"/>
                      <AGARI who="2" fromWho="0" ten="30,2000,0" yaku="18,1,19,1"/>
                    </mjloggm>
                    """)
            .events();
    var first = ((ReplayEvent.Hora) events.get(2)).details();
    var second = ((ReplayEvent.Hora) events.get(3)).details();
    Assert.assertEquals(first.uraIndicators(), List.of("1m"));
    Assert.assertEquals(first.yaku().getLast(), new WinDetails.Yaku("URADORA", 1, 0));
    Assert.assertEquals(second.points(), Integer.valueOf(2000));
    Assert.assertNull(second.uraIndicators());
    Assert.assertEquals(
        second.yaku(),
        List.of(
            new WinDetails.Yaku("YAKUHAI_HAKU", 1, 0), new WinDetails.Yaku("YAKUHAI_HATSU", 1, 0)));
  }

  @Test
  public void tenhouYakumanVariantsRetainTheirNamesAndRecordedSingleMultipliers() {
    var details =
        tenhouWin(
                """
                <AGARI who="0" fromWho="1" ten="0,96000,5" yakuman="41,42"/>
                """)
            .details();
    Assert.assertEquals(details.yakuman(), Integer.valueOf(2));
    Assert.assertEquals(details.han(), Integer.valueOf(0));
    Assert.assertEquals(details.points(), Integer.valueOf(96000));
    Assert.assertEquals(
        details.yaku(),
        List.of(
            new WinDetails.Yaku("SUUANKOU_TANKI", 0, 1), new WinDetails.Yaku("TSUUIISOU", 0, 1)));
  }

  @Test
  public void tenhouRedIndicatorsAndUnknownYakuRetainSourceIdentity() {
    var details =
        tenhouWin(
                """
                <AGARI who="1" fromWho="2" ten="30,8000,1" yaku="999,5"
                  doraHaiUra="16,52,88,124"/>
                """)
            .details();
    Assert.assertEquals(details.uraIndicators(), List.of("5mr", "5pr", "5sr", "P"));
    Assert.assertEquals(details.yaku(), List.of(new WinDetails.Yaku("TENHOU_999", 5, 0)));
  }

  @Test
  public void mjaiStandardHoraRetainsItsRecordedScoresAndGenericWindYaku() {
    var details =
        mjaiWin(
                """
                {"type":"hora","actor":2,"target":3,"pai":"P","fu":40,"fan":4,
                 "hora_points":8000,"yakus":[["reach",1],["jikaze",1],["sangenpai",1],["uradora",1]],
                 "uradora_markers":["5pr"]}
                """)
            .details();
    Assert.assertEquals(details.han(), Integer.valueOf(4));
    Assert.assertEquals(details.fu(), Integer.valueOf(40));
    Assert.assertEquals(details.points(), Integer.valueOf(8000));
    Assert.assertEquals(details.yaku().get(1), new WinDetails.Yaku("YAKUHAI_SEAT", 1, 0));
    Assert.assertEquals(details.yaku().get(2), new WinDetails.Yaku("YAKUHAI_SANGEN", 1, 0));
    Assert.assertEquals(details.uraIndicators(), List.of("5pr"));
  }

  @Test
  public void mjaiYakumanUsesHundredFanUnitsWithoutDisplayingHundredsOfHan() {
    var details =
        mjaiWin(
                """
                {"type":"hora","actor":2,"target":3,"fan":200,"fu":0,"hora_points":64000,
                 "yakus":[["daisangen",100],["tsuiso",100]],"uradora_markers":[]}
                """)
            .details();
    Assert.assertEquals(details.han(), Integer.valueOf(0));
    Assert.assertEquals(details.yakuman(), Integer.valueOf(2));
    Assert.assertEquals(
        details.yaku(),
        List.of(new WinDetails.Yaku("DAISANGEN", 0, 1), new WinDetails.Yaku("TSUUIISOU", 0, 1)));
    Assert.assertEquals(details.uraIndicators(), List.of());
  }

  @Test
  public void explicitMjaiDetailsRetainDoubleYakumanAndNullFields() {
    var details =
        mjaiWin(
                """
                {"type":"hora","actor":2,"target":2,"details":{
                  "han":0,"fu":null,"points":64000,"yakuman":2,
                  "yaku":[{"code":"KOKUSHI_13","han":0,"yakuman":2}],"uraIndicators":null}}
                """)
            .details();
    Assert.assertNull(details.fu());
    Assert.assertNull(details.uraIndicators());
    Assert.assertEquals(details.yakuman(), Integer.valueOf(2));
    Assert.assertEquals(details.yaku(), List.of(new WinDetails.Yaku("KOKUSHI_13", 0, 2)));
  }

  @Test
  public void missingScoringRemainsUnknownAndEmptyIndicatorsRemainKnown() {
    Assert.assertNull(tenhouWin("<AGARI who=\"0\" fromWho=\"1\"/>").details());
    Assert.assertNull(mjaiWin("{\"type\":\"hora\",\"actor\":0,\"target\":1}").details());
    var details =
        mjaiWin(
                """
                {"type":"hora","actor":0,"target":1,"uradora_markers":[]}
                """)
            .details();
    Assert.assertNull(details.han());
    Assert.assertNull(details.yaku());
    Assert.assertNull(details.yakuman());
    Assert.assertEquals(details.uraIndicators(), List.of());
  }

  @Test
  public void mjaiYakuCountsSupplyTotalsWhenFanIsNotRecorded() {
    var details =
        mjaiWin(
                """
                {"type":"hora","actor":0,"target":1,"yakus":[["reach",1],["future_yaku",2]]}
                """)
            .details();
    Assert.assertEquals(details.han(), Integer.valueOf(3));
    Assert.assertEquals(details.yaku().getLast(), new WinDetails.Yaku("MJAI_future_yaku", 2, 0));
  }

  @Test(
      expectedExceptions = IllegalArgumentException.class,
      expectedExceptionsMessageRegExp = "Tenhou yaku must contain ID and han pairs\\.")
  public void malformedTenhouYakuPairsFailAtTheRecordBoundary() {
    tenhouWin("<AGARI who=\"0\" fromWho=\"1\" yaku=\"1,1,7\"/>");
  }

  private static ReplayEvent.Hora tenhouWin(String tag) {
    return (ReplayEvent.Hora)
        TenhouXmlReader.readRecord("<mjloggm>" + initialXml() + tag + "</mjloggm>").events().get(2);
  }

  private static ReplayEvent.Hora mjaiWin(String json) {
    return (ReplayEvent.Hora) MjaiReader.readRecord("[" + json + "]").events().get(1);
  }

  private static String initialXml() {
    StringBuilder xml =
        new StringBuilder("<INIT seed='0,0,0,0,0,132' ten='250,250,250,250' oya='0'");
    for (int seat = 0; seat < 4; seat++) {
      xml.append(" hai").append(seat).append("='");
      for (int i = 0; i < 13; i++) {
        if (i > 0) xml.append(',');
        xml.append(seat * 13 + i);
      }
      xml.append("'");
    }
    return xml.append("/>").toString();
  }
}
