package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.record.SceneGames;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 天鳳の XML を書いて読み直すテスト。天鳳が書く形の牌譜の読み、局の番号、牌のタグ、読まない牌譜も確かめる。 */
public class TenhouXmlTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /**
   * 天鳳が書く形の四人打ちの牌譜（中身は作り物の 2 局で、席 1 が飛んで終わる）。{@code TenhouXml.write} が書かない SHUFFLE・BYE・PROF の
   * タグと、UN の段位・レート・性別を持ち、役は天鳳の役の番号で書いてある。東 1 局は、親の席 0 が 2 巡目に立直して一発でツモる（平和、
   * ドラ 1、赤 1、裏ドラなし）。1 本場は、親が国士無双十三面で席 1 の第一打にロンする。
   */
  private static final String TENHOU_FORM =
      """
      <mjloggm ver="2.3">\
      <SHUFFLE seed="mt19937ar-sha512-n288-base64,AAAAAAAA" ref=""/>\
      <GO type="169" lobby="0"/>\
      <UN n0="%E6%9D%B1" n1="%E5%8D%97" n2="%E8%A5%BF" n3="%E5%8C%97" dan="16,12,10,0" \
      rate="2105.31,1987.40,1820.55,1500.00" sx="M,F,M,C"/>\
      <TAIKYOKU oya="0"/>\
      <INIT seed="0,0,0,2,4,3" ten="250,250,250,250" oya="0" \
      hai0="104,8,52,0,69,96,48,76,4,100,56,80,68" hai1="1,12,24,40,53,64,81,92,105,108,112,124,128" \
      hai2="5,17,28,44,57,70,72,85,97,109,117,125,132" hai3="9,20,32,36,49,60,77,89,101,113,121,129,133"/>\
      <T120/><D120/><U33/><E33/><V37/><F37/><BYE who="3"/><W65/><G65/><UN n3="%E5%8C%97"/>\
      <T116/><REACH who="0" step="1"/><D116/><REACH who="0" ten="240,250,250,250" step="2"/>\
      <U61/><E61/><V93/><F93/><W45/><G45/><T84/>\
      <AGARI ba="0,1" hai="0,4,8,48,52,56,68,69,76,80,84,96,100,104" machi="84" ten="20,18000,2" \
      yaku="1,1,2,1,0,1,7,1,52,1,54,1,53,0" doraHai="3" doraHaiUra="135" who="0" fromWho="0" \
      sc="240,190,250,-60,250,-60,250,-60"/>\
      <INIT seed="0,1,0,1,5,71" ten="430,190,190,190" oya="0" \
      hai0="132,0,104,36,120,68,108,72,128,32,116,112,124" hai1="4,8,12,20,24,40,44,48,56,76,80,84,92" \
      hai2="5,9,13,21,25,41,45,49,57,77,81,85,93" hai3="6,10,14,22,26,42,46,50,58,78,82,86,94"/>\
      <T17/><D17/><U2/><E2/>\
      <AGARI ba="1,0" hai="0,2,32,36,68,72,104,108,112,116,120,124,128,132" machi="2" ten="30,48000,5" \
      yakuman="48" doraHai="71" who="0" fromWho="1" sc="430,483,190,-483,190,0,190,0" \
      owari="913,101.3,-293,-79.3,190,-1.0,190,-21.0"/>\
      <PROF lobby="0" type="169" add="101.3,1,0,0,0,0,1,1,2,0,1"/>\
      </mjloggm>""";

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** 自己対局の記録（鳴き・槓・立直・途中流局を広く含む。三人打ちは北抜きも）を書いて読み直すと同じ半荘になり、再生も記録と合う。 */
  @Test(dataProvider = "variants")
  public void selfPlayGamesSurviveAWriteAndARead(Variant variant) {
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    for (int seed = 0; seed < 30; seed++) {
      RecordedGame game = RecordedGames.play(variant, seed, "self-" + seed);
      RecordedGame again = TenhouXml.read(TenhouXml.write(game), game.meta().id());
      Assert.assertEquals(again.variant(), variant, game.meta().id());
      Assert.assertEquals(again, game, game.meta().id());
      driver.replay(again, ws);
    }
  }

  /**
   * 天鳳が書く形の牌譜を読む。SHUFFLE・BYE・PROF のタグと、UN の段位・レート・性別は読み飛ばす。AGARI の役は天鳳の役の番号で読む
   * （門前清自摸和 0、立直 1、一発 2、平和 7、国士無双十三面 48、ドラ 52、裏ドラ 53、赤ドラ 54）。読んだ半荘は記録どおりに再生できる。
   */
  @Test
  public void aLogInTenhouFormReadsByTenhouYakuNumbers() {
    RecordedGame game = TenhouXml.read(TENHOU_FORM, "tenhou-form");
    Assert.assertEquals(game.variant(), VARIANT);
    Assert.assertEquals(game.meta().names(), List.of("東", "南", "西", "北"));
    Assert.assertEquals(game.finalScores(), new int[] {91_300, -29_300, 19_000, 19_000});
    Assert.assertEquals(game.rounds().size(), 2);
    long yaku =
        ScoringYaku.RIICHI.bit() | ScoringYaku.IPPATSU.bit() | ScoringYaku.MENZEN_TSUMO.bit() | ScoringYaku.PINFU.bit();
    RoundResult.Win tsumo = RoundResult.Win.of(0, yaku, 6, 20, 1, 1, 0, 0, RoundOutcome.NO_SEAT);
    Assert.assertEquals(
        game.rounds().get(0).result(),
        new RoundResult(
            EndKind.TSUMO,
            null,
            List.of(tsumo),
            RoundOutcome.NO_SEAT,
            0,
            0,
            new int[] {18_000, -6_000, -6_000, -6_000},
            new int[] {135}));
    RoundResult.Win ron = RoundResult.Win.of(0, ScoringYaku.KOKUSHI.bit(), 0, 0, 0, 0, 0, 1, RoundOutcome.NO_SEAT);
    Assert.assertEquals(
        game.rounds().get(1).result(),
        new RoundResult(EndKind.RON, null, List.of(ron), 1, 0, 0, new int[] {48_300, -48_300, 0, 0}, new int[0]));
    new ReplayDriver(VARIANT).replay(game, new RuleWorkspace(VARIANT));
  }

  /**
   * 打ち手の名前は、UN タグの n0〜n3 を URL エンコードとして復号し、席の順に並べたもの（三人打ちは空き席の n3 を落とす）。途中に出る
   * UN タグは、書いてある席の名前だけを替える。書いた名前は読み直しても同じ。
   */
  @Test
  public void namesComeFromTheUnTags() {
    for (Variant variant : Variant.values()) {
      List<String> names = List.of("A", "B", "C", "D").subList(0, variant.seats());
      RecordedGame game = TenhouXml.read(TenhouXml.write(named(variant, names)), "named");
      Assert.assertEquals(game.meta().source(), RecordedGame.Source.TENHOU, variant.toString());
      Assert.assertEquals(game.meta().names(), names, variant.toString());
    }
    String xml = TenhouXml.write(named(VARIANT, List.of("A", "B", "C", "D")));
    String renamed =
        xml.replace("n0=\"%41\"", "n0=\"%E9%BA%BB%E9%9B%80\"").replaceFirst("<INIT", "<UN n2=\"%E3%81%82\"/><INIT");
    Assert.assertNotEquals(renamed, xml);
    RecordedGame game = TenhouXml.read(renamed, "named");
    Assert.assertEquals(game.meta().names(), List.of("麻雀", "B", "あ", "D"));
    Assert.assertEquals(TenhouXml.read(TenhouXml.write(game), "named"), game);
  }

  /** AGARI の点が翻・符と合わない牌譜は読まない。 */
  @Test
  public void anAgariWhosePointsDisagreeIsRejected() {
    String xml = TenhouXml.write(RecordedGames.play(VARIANT, 0, "self-0"));
    Matcher points = Pattern.compile("<AGARI [^>]*ten=\"\\d+,(\\d+),").matcher(xml);
    Assert.assertTrue(points.find(), "no AGARI");
    String wrong =
        xml.substring(0, points.start(1)) + (Integer.parseInt(points.group(1)) + 100) + xml.substring(points.end(1));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouXml.read(wrong, "wrong-points"));
  }

  /**
   * 包のある和了の AGARI は、3 つ目の三元牌を鳴かせた席を paoWho に書く。その席が放銃したロンは包の無いロンで、paoWho を書かない。
   * どちらの半荘も、読み直すと同じ半荘になり、再生も記録と合う。
   */
  @Test
  public void paoWhoNamesTheSeatThatFedTheThirdDragon() {
    RecordedGame tsumo = SceneGames.paoTsumo();
    RecordedGame ron = SceneGames.paoFeederRon();
    Assert.assertEquals(tsumo.rounds().get(0).result().wins().get(0).pao(), 1);
    Assert.assertEquals(ron.rounds().get(0).result().wins().get(0).pao(), RoundOutcome.NO_SEAT);
    Assert.assertEquals(agari(tsumo).get("paoWho"), "1");
    Assert.assertNull(agari(ron).get("paoWho"));
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    for (RecordedGame game : List.of(tsumo, ron)) {
      RecordedGame again = TenhouXml.read(TenhouXml.write(game), game.meta().id());
      Assert.assertEquals(again, game, game.meta().id());
      driver.replay(again, ws);
    }
  }

  /** ダブロンの AGARI の ba は、天鳳と同じく、どの和了者にも本場を書き、供託は最初の和了者にだけ書く。 */
  @Test
  public void everyAgariOfADoubleRonCarriesTheHonba() {
    Assert.assertEquals(agariBa(SceneGames.doubleRon(1)), List.of("1,1", "1,0"));
    Assert.assertEquals(agariBa(SceneGames.threePlayerDoubleRon()), List.of("2,1", "2,0"));
  }

  /** 赤なし・喰いなし・東風の卓（GO の type）は読まない。 */
  @Test
  public void otherTenhouRulesAreRejected() {
    String xml = TenhouXml.write(RecordedGames.play(VARIANT, 0, "self-0"));
    int hanchan = TenhouRule.hanchan(VARIANT);
    String go = "<GO type=\"" + hanchan + "\"";
    Assert.assertTrue(xml.contains(go), xml.substring(0, Math.min(200, xml.length())));
    int[] others = {
      hanchan | TenhouRule.NO_RED.bit(),
      hanchan | TenhouRule.NO_OPEN_TANYAO.bit(),
      hanchan & ~TenhouRule.EAST_SOUTH.bit()
    };
    for (int type : others) {
      String other = xml.replace(go, "<GO type=\"" + type + "\"");
      Assert.assertThrows(IllegalArgumentException.class, () -> TenhouXml.read(other, "type-" + type));
    }
  }

  /** GO の type に三麻の bit がある卓（三鳳南喰赤 185、その速卓 249）は三人打ち、ほかは四人打ち。書いた type を読むと同じルールに戻る。 */
  @Test
  public void theThreePlayerBitNamesTheVariant() {
    Assert.assertEquals(TenhouXml.variant(169), Variant.TENHOU_4P);
    Assert.assertEquals(TenhouXml.variant(185), Variant.TENHOU_3P);
    Assert.assertEquals(TenhouXml.variant(249), Variant.TENHOU_3P);
    for (Variant variant : Variant.values()) {
      Assert.assertEquals(TenhouXml.variant(TenhouRule.hanchan(variant)), variant);
    }
    Assert.assertEquals(TenhouRule.hanchan(Variant.TENHOU_4P), 9);
    Assert.assertEquals(TenhouRule.hanchan(Variant.TENHOU_3P), 25);
  }

  /** 三人打ちの局の始まりでは、4 席分のフィールドから空き席を落とす。空き席に配牌か点のある INIT は読まない。 */
  @Test
  public void theEmptySeatOfAThreePlayerGameIsDropped() {
    RecordedGame played = RecordedGames.play(Variant.TENHOU_3P, 0, "self3-0");
    String xml = TenhouXml.write(played);
    RecordedGame game = TenhouXml.read(xml, "self3-0");
    Assert.assertEquals(game.rounds().get(0).start().scores(), new int[] {35_000, 35_000, 35_000});
    Assert.assertEquals(game.finalScores(), played.finalScores());
    String scored = xml.replaceFirst("ten=\"350,350,350,0\"", "ten=\"350,350,350,10\"");
    Assert.assertNotEquals(scored, xml);
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouXml.read(scored, "scored"));
    String dealt = xml.replaceFirst("hai3=\"\"", "hai3=\"1\"");
    Assert.assertNotEquals(dealt, xml);
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouXml.read(dealt, "dealt"));
  }

  /** 局の番号は、四人打ちなら天鳳の番号と同じ値になる。三人打ちでは、天鳳が場ごとに 4 つずつ取る番号を 3 局に詰める。 */
  @Test
  public void roundNumbersFollowTheWind() {
    for (int round = 0; round <= VARIANT.finalRound(); round++) {
      Assert.assertEquals(TenhouRoundIndex.tenhou(round, VARIANT), round);
      Assert.assertEquals(TenhouRoundIndex.round(round, VARIANT), round);
    }
    Variant sanma = Variant.TENHOU_3P;
    Assert.assertEquals(TenhouRoundIndex.round(4, sanma), 3);
    Assert.assertEquals(TenhouRoundIndex.tenhou(8, sanma), 10);
    for (int round = 0; round <= sanma.finalRound(); round++) {
      Assert.assertEquals(TenhouRoundIndex.round(TenhouRoundIndex.tenhou(round, sanma), sanma), round);
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouRoundIndex.round(3, sanma));
  }

  @Test
  public void tileTagsNameTheSeatAndTheTile() {
    Assert.assertEquals(
        TenhouTiles.TileTag.parse("T89", VARIANT), new TenhouTiles.TileTag(TenhouTiles.Move.DRAW, 0, 89));
    Assert.assertEquals(
        TenhouTiles.TileTag.parse("G37", VARIANT), new TenhouTiles.TileTag(TenhouTiles.Move.DISCARD, 3, 37));
    Assert.assertEquals(new TenhouTiles.TileTag(TenhouTiles.Move.DISCARD, 1, 73).name(), "E73");
    Assert.assertNull(TenhouTiles.TileTag.parse("INIT", VARIANT));
    Assert.assertNull(TenhouTiles.TileTag.parse("W12", Variant.TENHOU_3P));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouTiles.TileTag.parse("T136", VARIANT));
    Assert.assertEquals(TenhouTiles.tiles("", VARIANT.tiles()), new int[0]);
    Assert.assertEquals(TenhouTiles.tiles("4,0,135", VARIANT.tiles()), new int[] {4, 0, 135});
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouTiles.tiles("4,8", Variant.TENHOU_3P.tiles()));
  }

  /** 半荘を書いた XML の、最初の AGARI のタグ。 */
  private static TenhouTag agari(RecordedGame game) {
    return TenhouTag.parse(TenhouXml.write(game)).stream()
        .filter(tag -> tag.name().equals("AGARI"))
        .findFirst()
        .orElseThrow();
  }

  /** 半荘を書いた XML の、AGARI のタグの ba を書いた順に並べたもの。 */
  private static List<String> agariBa(RecordedGame game) {
    return TenhouTag.parse(TenhouXml.write(game)).stream()
        .filter(tag -> tag.name().equals("AGARI"))
        .map(tag -> tag.required("ba"))
        .toList();
  }

  /** 自己対局の半荘に、打ち手の名前を付けたもの。 */
  private static RecordedGame named(Variant variant, List<String> names) {
    RecordedGame game = RecordedGames.play(variant, 0, "named");
    return new RecordedGame(
        variant, game.rounds(), game.finalScores(), new RecordedGame.Meta("named", RecordedGame.Source.TENHOU, names));
  }
}
