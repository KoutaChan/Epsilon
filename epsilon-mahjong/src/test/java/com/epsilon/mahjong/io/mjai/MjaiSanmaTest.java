package com.epsilon.mahjong.io.mjai;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの mjai を書いて読むテスト。見本はエンジンで打った半荘（手をランダムに選ぶ自己対局と、牌山を手で並べた 1 局の場面）。書いて
 * 読むと同じ記録になり、読んだ記録は同じエンジンで記録どおりに再生できる。
 */
public class MjaiSanmaTest {

  private static final Variant VARIANT = Variant.TENHOU_3P;

  /** 自己対局の半荘の数。 */
  private static final int GAMES = 30;

  /** 1z・2z の双碰の聴牌。 */
  private static final String TENPAI = "123p456p789p1122z";

  /** 面子の形にならない不聴の手牌（北を持たない）。 */
  private static final String NOTEN_A = "19m258p147s369s67z";

  private static final String NOTEN_B = "19m369p258s147p57z";
  private static final String NOTEN_C = "19m369p258s369s57z";

  /** 北抜きのイベントの type。 */
  private static final String KITA = "kita";

  /** JSON で書いた北抜きのイベントの先頭。 */
  private static final String KITA_EVENT = "{\"type\":\"kita\"";

  /** hora の yakus に書く抜きドラの名前。 */
  private static final String NUKIDORA = "nukidora";

  /**
   * 自己対局の半荘を書いて読むと、局の始まり、イベントの数、局の結果（裏ドラ表示牌の物理 id を除く）、終局の点が元と同じになり、読み直し
   * ても書き直しても変わらない。読んだ半荘は記録どおりに再生できる。見本には北抜きと、抜きドラの付いた和了が含まれる。
   */
  @Test
  public void selfPlayGamesSurviveAWriteAndARead() {
    int nukis = 0;
    int nukiDoraWins = 0;
    for (RecordedGame game : selfPlay()) {
      String text = MjaiCodec.write(game);
      roundTrip(game, text);
      for (RecordedRound round : game.rounds()) {
        nukis += (int) round.events().stream().filter(RecordEvent.Nuki.class::isInstance).count();
      }
      for (JsonObject hora : ofType(text, "hora")) {
        nukiDoraWins += yakuHan(hora, NUKIDORA) > 0 ? 1 : 0;
      }
    }
    Assert.assertTrue(nukis > 0, "no nuki was played");
    Assert.assertTrue(nukiDoraWins > 0, "no win has nuki dora");
  }

  /**
   * 書いた牌譜では、席ごとのフィールド（names・scores・tehais・deltas・tenpais）が 3 つずつで、局は 1〜3、場風は東・南・西。北抜きは
   * {@code {"type":"kita","actor":席,"pai":"N"}} の形だけで書き（type が nukidora・nuki・babei の北抜きは書かない）、チーは無い。
   */
  @Test
  public void theWrittenRecordHasThreeSeats() {
    int seats = VARIANT.seats();
    for (RecordedGame game : selfPlay()) {
      for (JsonObject event : parse(MjaiCodec.write(game))) {
        String type = event.get("type").getAsString();
        switch (type) {
          case "start_game" -> Assert.assertEquals(event.getAsJsonArray("names").size(), seats);
          case "end_game" -> Assert.assertEquals(event.getAsJsonArray("scores").size(), seats);
          case "start_kyoku" -> {
            Assert.assertEquals(event.getAsJsonArray("scores").size(), seats);
            JsonArray hands = event.getAsJsonArray("tehais");
            Assert.assertEquals(hands.size(), seats);
            for (JsonElement hand : hands) {
              Assert.assertEquals(hand.getAsJsonArray().size(), 13);
            }
            int kyoku = event.get("kyoku").getAsInt();
            Assert.assertTrue(kyoku >= 1 && kyoku <= seats, event.toString());
            Assert.assertTrue(Set.of("E", "S", "W").contains(event.get("bakaze").getAsString()), event.toString());
          }
          case "hora" -> Assert.assertEquals(event.getAsJsonArray("deltas").size(), seats);
          case "ryukyoku" -> {
            Assert.assertEquals(event.getAsJsonArray("deltas").size(), seats);
            Assert.assertEquals(event.getAsJsonArray("tenpais").size(), seats);
          }
          case KITA -> {
            int actor = event.get("actor").getAsInt();
            Assert.assertTrue(actor >= 0 && actor < seats, event.toString());
            Assert.assertEquals(event.toString(), KITA_EVENT + ",\"actor\":" + actor + ",\"pai\":\"N\"}");
          }
          case "nukidora", "nuki", "babei" -> Assert.fail("a nuki is written under another name: " + event);
          case "chi" -> Assert.fail("a three-player record has a chi: " + event);
          default -> {}
        }
      }
    }
  }

  /** 四人打ちの牌譜には北抜きも抜きドラも書かない。 */
  @Test
  public void fourPlayerRecordsHaveNoNukidora() {
    for (int seed = 0; seed < 5; seed++) {
      String text = MjaiCodec.write(RecordedGames.play(Variant.TENHOU_4P, seed, "four-" + seed));
      Assert.assertFalse(text.contains(KITA_EVENT), "seed " + seed);
      Assert.assertFalse(text.contains(NUKIDORA), "seed " + seed);
    }
  }

  /**
   * 北を 2 回抜いた親が嶺上開花でツモ和了する。hora の yakus には、抜いた北の 2 翻を nukidora に、表ドラの 1 翻を dora に書く。この
   * 2 つの翻の和は、記録のドラの翻（抜きドラを含む）と同じになる。
   */
  @Test
  public void nukiDoraIsWrittenApartFromTheDora() {
    RecordedGame game =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "123p456p789p1s9s44z")
                .hand(1, NOTEN_A)
                .hand(2, NOTEN_C)
                .indicator("8p")
                .draws("9s")
                .rinshan("2s3s"),
            "two-nukis",
            "0:NUKI",
            "0:NUKI",
            "0:TSUMO_AGARI");
    String text = MjaiCodec.write(game);
    roundTrip(game, text);
    RoundResult.Win win = game.rounds().get(0).result().wins().get(0);
    JsonObject hora = ofType(text, "hora").get(0);
    Assert.assertEquals(yakuHan(hora, NUKIDORA), 2, hora.toString());
    Assert.assertEquals(yakuHan(hora, "dora"), 1, hora.toString());
    Assert.assertEquals(yakuHan(hora, "dora") + yakuHan(hora, NUKIDORA), win.dora(), hora.toString());
  }

  /** 北を抜いた後に国士無双をツモ和了しても、役満にはドラの翻が付かないので nukidora を書かない。 */
  @Test
  public void aYakumanAfterANukiHasNoNukidora() {
    RecordedGame game =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "19m19p19s123z44z56z")
                .hand(1, NOTEN_A)
                .hand(2, NOTEN_C)
                .draws("4z")
                .rinshan("7z"),
            "kokushi-after-nuki",
            "0:NUKI",
            "0:TSUMO_AGARI");
    String text = MjaiCodec.write(game);
    roundTrip(game, text);
    JsonObject hora = ofType(text, "hora").get(0);
    Assert.assertEquals(yakuHan(hora, "kokushimuso"), 100, hora.toString());
    Assert.assertFalse(hora.toString().contains(NUKIDORA), hora.toString());
  }

  /**
   * pai の無い北抜きも同じ半荘に読む。北抜きの type として読むのは kita だけで、nukidora・nuki・babei の北抜きは読まない。yakus の
   * 抜きドラの名前も nukidora だけを読み、nuki・kita・babei は読まない。
   */
  @Test
  public void onlyKitaAndNukidoraAreRead() {
    int renamed = 0;
    for (RecordedGame game : selfPlay()) {
      String text = MjaiCodec.write(game);
      if (!text.contains(KITA_EVENT)) {
        continue;
      }
      RecordedGame read = MjaiCodec.read(text, "names");
      String withoutPai = withoutKitaPai(text);
      Assert.assertNotEquals(withoutPai, text);
      Assert.assertEquals(MjaiCodec.read(withoutPai, "names"), read);
      for (String removed : List.of("nukidora", "nuki", "babei")) {
        String type = text.replace(KITA_EVENT, "{\"type\":\"" + removed + "\"");
        Assert.assertNotEquals(type, text);
        Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(type, removed));
      }
      for (String removed : List.of("nuki", "kita", "babei")) {
        String yaku = text.replace("[\"nukidora\",", "[\"" + removed + "\",");
        if (!yaku.equals(text)) {
          renamed++;
          Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(yaku, removed));
        }
      }
    }
    Assert.assertTrue(renamed > 0, "no hora has a nukidora yaku");
  }

  /**
   * RiichiEnv の書き方の三人打ちの記録を読む。この書き方では、start_game は type だけを持ち、hora は内訳と pai を持たずに tsumo と
   * ura_markers を持つ。北抜きには pai が無く、ryukyoku の reason は exhaustive_draw・kyushu_kyuhai、大明槓の type は kan になる。読むと、
   * 内訳だけを落とした元の牌譜と同じ半荘になり（打ち手の名前は空）、記録どおりに再生できる。
   */
  @Test
  public void riichiEnvStyleThreePlayerRecordsReplay() {
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    int kitas = 0;
    int exhaustive = 0;
    for (RecordedGame original : selfPlay()) {
      String id = original.meta().id();
      String text = MjaiCodec.write(original);
      String riichiEnv = riichiEnvStyle(text);
      kitas += (int) ofType(riichiEnv, KITA).stream().filter(kita -> !kita.has("pai")).count();
      exhaustive += riichiEnv.contains("\"reason\":\"exhaustive_draw\"") ? 1 : 0;
      RecordedGame game = MjaiCodec.read(riichiEnv, id);
      Assert.assertEquals(game, MjaiCodec.read(withoutBreakdown(text), id), id);
      Assert.assertEquals(game.variant(), VARIANT, id);
      Assert.assertEquals(game.meta().names(), List.of("", "", ""), id);
      Assert.assertEquals(game.finalScores(), original.finalScores(), id);
      driver.replay(game, ws);
    }
    Assert.assertTrue(kitas > 0, "no kita without a pai was read");
    Assert.assertTrue(exhaustive > 0, "no exhaustive_draw was read");

    RecordedGame nine =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "19m19p19s1234567z")
                .hand(1, NOTEN_A)
                .hand(2, "19m369p258s147s67z")
                .draws("5p")
                .rinshan("5s"),
            "nine-terminals",
            "0:KYUSHU_KYUHAI");
    RecordedGame kan =
        RecordedGames.scene(
            Scenes.first(VARIANT).hand(0, "19m258p147s369s45z").hand(1, "444z123p456p78s19m").hand(2, NOTEN_C).draws("6z"),
            "north-daiminkan",
            "0:DAHAI(N)",
            "1:DAIMINKAN(N)");
    String nineText = riichiEnvStyle(MjaiCodec.write(nine));
    String kanText = riichiEnvStyle(MjaiCodec.write(kan));
    Assert.assertTrue(nineText.contains("\"reason\":\"kyushu_kyuhai\""), nineText);
    Assert.assertTrue(kanText.contains("{\"type\":\"kan\""), kanText);
    for (RecordedGame game : List.of(nine, kan)) {
      String id = game.meta().id();
      String text = MjaiCodec.write(game);
      Assert.assertEquals(MjaiCodec.read(riichiEnvStyle(text), id), MjaiCodec.read(withoutBreakdown(text), id), id);
    }
    Assert.assertEquals(
        MjaiCodec.read(nineText, "nine").rounds().get(0).result().abortive(), AbortiveDraw.NINE_TERMINALS);
  }

  /** names の無い start_game では、四人打ちか三人打ちかを scores の数で決める。names の数が scores の数と合わない牌譜は読まない。 */
  @Test
  public void theVariantComesFromTheScoresWithoutNames() {
    String four = MjaiCodec.write(RecordedGames.play(Variant.TENHOU_4P, 0, "four"));
    RecordedGame read = MjaiCodec.read(riichiEnvStyle(four), "four");
    Assert.assertEquals(read.variant(), Variant.TENHOU_4P);
    Assert.assertEquals(read.meta().names(), List.of("", "", "", ""));
    Assert.assertEquals(read, MjaiCodec.read(withoutBreakdown(four), "four"));

    String three = MjaiCodec.write(selfPlay().get(0));
    String fourNames = three.replaceFirst("\"names\":\\[[^\\]]*\\]", "\"names\":[\"\",\"\",\"\",\"\"]");
    Assert.assertNotEquals(fourNames, three);
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(fourNames, "disagree"));
  }

  /** 四人打ちの牌譜の北抜き、北でない牌の北抜き、北を持たない席の北抜き、三人打ちの牌譜のチーは読まない。 */
  @Test
  public void inconsistentThreePlayerRecordsAreRejected() {
    String four = MjaiCodec.write(RecordedGames.play(Variant.TENHOU_4P, 0, "four"));
    int tsumo = four.indexOf("{\"type\":\"tsumo\"");
    int actor = JsonParser.parseString(lineAt(four, tsumo)).getAsJsonObject().get("actor").getAsInt();
    String fourNuki = insertAfterLine(four, tsumo, KITA_EVENT + ",\"actor\":" + actor + ",\"pai\":\"N\"}");
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(fourNuki, "four-nuki"));

    String three = textWithNuki();
    String west = three.replaceFirst("(\"type\":\"kita\",\"actor\":\\d,\"pai\":)\"N\"", "$1\"W\"");
    Assert.assertNotEquals(west, three);
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(west, "west"));

    int kyoku = three.indexOf("{\"type\":\"start_kyoku\"");
    JsonArray hands = JsonParser.parseString(lineAt(three, kyoku)).getAsJsonObject().getAsJsonArray("tehais");
    int seat = 0;
    while (hands.get(seat).getAsJsonArray().contains(JsonParser.parseString("\"N\""))) {
      seat++;
    }
    String noNorth = insertAfterLine(three, kyoku, KITA_EVENT + ",\"actor\":" + seat + ",\"pai\":\"N\"}");
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(noNorth, "no-north"));

    int dahai = three.indexOf("{\"type\":\"dahai\"");
    JsonObject discard = JsonParser.parseString(lineAt(three, dahai)).getAsJsonObject();
    int discarder = discard.get("actor").getAsInt();
    JsonObject chi = new JsonObject();
    chi.addProperty("type", "chi");
    chi.addProperty("actor", (discarder + 1) % VARIANT.seats());
    chi.addProperty("target", discarder);
    chi.add("pai", discard.get("pai"));
    JsonArray consumed = new JsonArray();
    consumed.add("2p");
    consumed.add("3p");
    chi.add("consumed", consumed);
    String withChi = insertAfterLine(three, dahai, chi.toString());
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(withChi, "chi"));
  }

  /** 抜いた北へのロンでは、hora の pai は北、target は北を抜いた席になる。読むと、北を抜いた席が放銃したロンになる。 */
  @Test
  public void aRonOnANukiNamesTheNukiSeat() {
    RecordedGame game =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "19m258p369s147s5z6z")
                .hand(1, "123p456p789s11z44z")
                .hand(2, "19m369p258s147s3z7z")
                .draws("9p2z8p4z")
                .rinshan("5s"),
            "ron-on-nuki",
            "1:RIICHI_TSUMOGIRI(S)",
            "0:NUKI",
            "1:RON_AGARI");
    String text = MjaiCodec.write(game);
    RecordedGame read = roundTrip(game, text);
    JsonObject hora = ofType(text, "hora").get(0);
    Assert.assertEquals(hora.get("actor").getAsInt(), 1);
    Assert.assertEquals(hora.get("target").getAsInt(), 0);
    Assert.assertEquals(hora.get("pai").getAsString(), "N");
    RoundResult result = read.rounds().get(0).result();
    Assert.assertEquals(result.kind(), EndKind.RON);
    Assert.assertEquals(result.loser(), 0);
  }

  /**
   * 大明槓の後の嶺上の手番で北を抜くと、保留の槓ドラを補充のツモの前にめくる。書いた並びは RiichiEnv と同じ kita → dora → tsumo で、
   * 読むと北抜き → 表示牌 → ツモの並びになる。dora を補充のツモの後に入れ替えた牌譜は読まない。
   */
  @Test
  public void aNukiAfterAKanWritesThePendingDoraBeforeItsReplacement() {
    RecordedGame game =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "9s19m258p147s36s5z6z")
                .hand(1, "999s4z123p456p78p1m")
                .hand(2, "19m689p258s147s37z")
                .draws("2z")
                .rinshan("5s1z"),
            "kan-then-nuki",
            "0:DAHAI(9s)",
            "1:DAIMINKAN(9s)",
            "1:NUKI");
    String text = MjaiCodec.write(game);
    RecordedGame read = roundTrip(game, text);
    List<String> lines = List.of(text.split("\n"));
    int nuki = indexOfType(lines, KITA);
    Assert.assertEquals(type(lines.get(nuki + 1)), "dora");
    Assert.assertEquals(type(lines.get(nuki + 2)), "tsumo");
    List<RecordEvent> events = read.rounds().get(0).events();
    int at = indexOf(events, RecordEvent.Nuki.class);
    Assert.assertTrue(events.get(at + 1) instanceof RecordEvent.DoraReveal, events.toString());
    Assert.assertTrue(events.get(at + 2) instanceof RecordEvent.Draw, events.toString());

    List<String> late = new ArrayList<>(lines);
    late.set(nuki + 1, lines.get(nuki + 2));
    late.set(nuki + 2, lines.get(nuki + 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(String.join("\n", late), "late"));
  }

  /** 北のポン・大明槓・加槓・暗槓はふつうの面子で、どれも書いて読むと同じ記録になり、再生できる。 */
  @Test
  public void northMeldsAreOrdinaryMelds() {
    RecordedGame open =
        RecordedGames.scene(
            Scenes.first(VARIANT).hand(0, "19m258p147s369s45z").hand(1, "444z123p456p78s19m").hand(2, NOTEN_C).draws("6z"),
            "north-daiminkan",
            "0:DAHAI(N)",
            "1:DAIMINKAN(N)");
    RecordedGame added =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "19m258p147s369s45z")
                .hand(1, "44z123p456p789s19m")
                .hand(2, NOTEN_C)
                .draws("6z2z7p4z"),
            "north-kakan",
            "0:DAHAI(N)",
            "1:PON(N)",
            "1:DAHAI(9m)",
            "1:KAKAN(N)");
    RecordedGame closed =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "4444z123p456p789s")
                .hand(1, NOTEN_A)
                .hand(2, NOTEN_C)
                .indicator("3z")
                .indicator(1, "1z")
                .draws("1m")
                .rinshan("1m"),
            "north-ankan",
            "0:ANKAN(N)",
            "0:TSUMO_AGARI");
    for (RecordedGame game : List.of(open, added, closed)) {
      roundTrip(game, MjaiCodec.write(game));
    }
    Assert.assertTrue(hasMeld(open, Meld.Type.DAIMINKAN));
    Assert.assertTrue(hasMeld(added, Meld.Type.PON));
    Assert.assertTrue(added.rounds().get(0).events().contains(new RecordEvent.AddedKan(1, Tiles.PEI)));
    Assert.assertTrue(closed.rounds().get(0).events().contains(new RecordEvent.ClosedKan(0, Tiles.PEI)));
    JsonObject hora = ofType(MjaiCodec.write(closed), "hora").get(0);
    Assert.assertEquals(yakuHan(hora, "dora"), 4, hora.toString());
    Assert.assertEquals(yakuHan(hora, NUKIDORA), 0, hora.toString());
  }

  /**
   * 三人打ちの流局（1 人聴牌の荒牌、点の移動から席を求める流し満貫（ツモ損）、九種九牌、四槓散了）は、書いて読んでも同じ結果になり、
   * 再生できる。
   */
  @Test
  public void threePlayerDrawsSurviveAWriteAndARead() {
    RecordedGame exhaustive =
        RecordedGames.scene(
            Scenes.first(VARIANT).hand(0, TENPAI).hand(1, NOTEN_A).hand(2, NOTEN_B).drawAt(0, "4s4s4s"), "tenpai");
    RoundResult tenpai = roundTrip(exhaustive, MjaiCodec.write(exhaustive)).rounds().get(0).result();
    Assert.assertEquals(tenpai.kind(), EndKind.EXHAUSTIVE);
    Assert.assertEquals(tenpai.tenpaiMask(), 0b001);
    Assert.assertEquals(tenpai.deltas(), new int[] {2_000, -1_000, -1_000});

    RecordedGame childNagashi = nagashi(1);
    RoundResult child = roundTrip(childNagashi, MjaiCodec.write(childNagashi)).rounds().get(0).result();
    Assert.assertEquals(child.nagashiMask(), 0b010);
    Assert.assertEquals(child.deltas(), new int[] {-4_000, 6_000, -2_000});
    RecordedGame dealerNagashi = nagashi(0);
    RoundResult dealer = roundTrip(dealerNagashi, MjaiCodec.write(dealerNagashi)).rounds().get(0).result();
    Assert.assertEquals(dealer.nagashiMask(), 0b001);
    Assert.assertEquals(dealer.deltas(), new int[] {8_000, -4_000, -4_000});

    RecordedGame nine =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "19m19p19s1234567z")
                .hand(1, NOTEN_A)
                .hand(2, "19m369p258s147s67z")
                .draws("5p")
                .rinshan("5s"),
            "nine-terminals",
            "0:KYUSHU_KYUHAI");
    Assert.assertEquals(
        roundTrip(nine, MjaiCodec.write(nine)).rounds().get(0).result().abortive(), AbortiveDraw.NINE_TERMINALS);

    RecordedGame kans =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "1111p2222p58s1s67z")
                .hand(1, "3333p4444p137s56z")
                .hand(2, "19m689p258s147s57z")
                .draws("9s9p")
                .rinshan("2z3z2s3s"),
            "four-kans",
            "0:ANKAN(1p)",
            "0:ANKAN(2p)",
            "0:DAHAI(9s)",
            "1:ANKAN(3p)",
            "1:ANKAN(4p)",
            "1:DAHAI(9p)");
    Assert.assertEquals(
        roundTrip(kans, MjaiCodec.write(kans)).rounds().get(0).result().abortive(), AbortiveDraw.FOUR_KANS);
  }

  /** hora は内訳のフィールドを全て持つか、どれも持たないかのどちらか。一部だけを持つ hora は読まない。 */
  @Test
  public void aHoraWithPartOfItsBreakdownIsRejected() {
    List<JsonObject> events = parse(withoutBreakdown(MjaiCodec.write(selfPlay().get(0))));
    JsonObject hora =
        events.stream().filter(event -> event.get("type").getAsString().equals("hora")).findFirst().orElseThrow();
    hora.addProperty("fu", 30);
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(lines(events), "part"));
  }

  /**
   * 書いた牌譜を読み、書き直すと同じ文字列に、読み直すと同じ半荘になることを確かめる。局の始まり、イベントの数、局の結果（裏ドラ表示牌の
   * 物理 id を除く）、終局の点も元と同じことを確かめる。読んだ局は 1 局ずつ記録どおりに再生でき、エンジンの局末は元の結果と合う。
   *
   * @param game 元の半荘
   * @param text 元の半荘を書いた牌譜
   * @return 読んだ半荘
   */
  private static RecordedGame roundTrip(RecordedGame game, String text) {
    String id = game.meta().id();
    RecordedGame read = MjaiCodec.read(text, id);
    Assert.assertEquals(MjaiCodec.write(read), text, id);
    Assert.assertEquals(MjaiCodec.read(text, id), read, id);
    Assert.assertEquals(read.variant(), VARIANT, id);
    Assert.assertEquals(read.finalScores(), game.finalScores(), id);
    Assert.assertEquals(read.rounds().size(), game.rounds().size(), id);
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    for (int index = 0; index < game.rounds().size(); index++) {
      RecordedRound before = game.rounds().get(index);
      RecordedRound after = read.rounds().get(index);
      RoundResult result = after.result();
      String where = id + " " + before.start();
      Assert.assertEquals(after.start(), before.start(), where);
      Assert.assertEquals(after.events().size(), before.events().size(), where);
      Assert.assertEquals(result.kind(), before.result().kind(), where);
      Assert.assertEquals(result.abortive(), before.result().abortive(), where);
      Assert.assertEquals(result.wins(), before.result().wins(), where);
      Assert.assertEquals(result.loser(), before.result().loser(), where);
      Assert.assertEquals(result.tenpaiMask(), before.result().tenpaiMask(), where);
      Assert.assertEquals(result.nagashiMask(), before.result().nagashiMask(), where);
      Assert.assertEquals(result.deltas(), before.result().deltas(), where);
      Assert.assertEquals(result.uraCount(), before.result().uraCount(), where);
      Boundary boundary = driver.start(after, ws);
      while (boundary == Boundary.DECISIONS) {
        boundary = driver.step(ws);
      }
      Assert.assertTrue(before.result().matches(driver.engine().roundEnd(), before.riichiMask()), where);
    }
    if (game.rounds().size() > 1) {
      driver.replay(read, ws);
    }
    return read;
  }

  private static List<RecordedGame> selfPlay() {
    List<RecordedGame> games = new ArrayList<>();
    for (int seed = 0; seed < GAMES; seed++) {
      games.add(RecordedGames.play(VARIANT, seed, "sanma-" + seed));
    }
    return games;
  }

  /** 北抜きを含む自己対局の牌譜のうち、最初のもの。 */
  private static String textWithNuki() {
    for (RecordedGame game : selfPlay()) {
      String text = MjaiCodec.write(game);
      if (text.contains(KITA_EVENT)) {
        return text;
      }
    }
    throw new AssertionError("no self-play game takes out a north");
  }

  /** 北抜きの pai を落とした牌譜。 */
  private static String withoutKitaPai(String text) {
    return text.replaceAll("(\"type\":\"kita\",\"actor\":\\d),\"pai\":\"N\"", "$1");
  }

  /** hora の内訳（yakus・fan・fu・hora_points・pao）を落とした牌譜。 */
  private static String withoutBreakdown(String text) {
    List<JsonObject> events = parse(text);
    for (JsonObject event : events) {
      if (event.get("type").getAsString().equals("hora")) {
        for (String key : List.of("yakus", "fan", "fu", "hora_points", "pao")) {
          event.remove(key);
        }
      }
    }
    return lines(events);
  }

  /**
   * 牌譜を RiichiEnv の書き方に直す。start_game は type だけにする。hora は内訳と pai を落として tsumo を足し、裏ドラ表示牌を
   * ura_markers に書く。北抜きの pai を落とし、大明槓の type を kan に、ryukyoku の reason の fanpai・kyushukyuhai を
   * exhaustive_draw・kyushu_kyuhai に書き換える。
   */
  private static String riichiEnvStyle(String text) {
    List<JsonObject> events = parse(text);
    for (JsonObject event : events) {
      switch (event.get("type").getAsString()) {
        case "start_game" -> event.remove("names");
        case "hora" -> {
          for (String key : List.of("yakus", "fan", "fu", "hora_points", "pao", "pai")) {
            event.remove(key);
          }
          event.addProperty("tsumo", event.get("actor").getAsInt() == event.get("target").getAsInt());
          JsonElement ura = event.remove("uradora_markers");
          if (ura != null) {
            event.add("ura_markers", ura);
          }
        }
        case KITA -> event.remove("pai");
        case "daiminkan" -> event.addProperty("type", "kan");
        case "ryukyoku" -> {
          String reason = event.has("reason") ? event.get("reason").getAsString() : "";
          if (reason.equals("fanpai")) {
            event.addProperty("reason", "exhaustive_draw");
          } else if (reason.equals("kyushukyuhai")) {
            event.addProperty("reason", "kyushu_kyuhai");
          }
        }
        default -> {}
      }
    }
    return lines(events);
  }

  /**
   * 流し満貫で終わる局を打つ。1 本場で、1 つの席のツモを全て么九牌にし、ほかの席の最初のツモを么九牌でない牌にして、全員がツモ切りで
   * 荒牌まで打つ。山のツモの k 番目は席 k mod 3 が引く。
   *
   * @param seat ツモが全て么九牌の席
   * @return その 1 局だけの半荘
   */
  private static RecordedGame nagashi(int seat) {
    String[] terminals = "1m9m1p9p1s9s1z2z3z5z6z7z1m9m1p9p1s9s1z".split("(?<=[mpsz])");
    Scenes scene = Scenes.at(VARIANT, 0, 1, 0, 35_000, 35_000, 35_000);
    int seats = VARIANT.seats();
    for (int other = 0; other < seats; other++) {
      if (other != seat) {
        scene.drawAt(other, "5s");
      }
    }
    for (int k = seat, index = 0; k < VARIANT.liveWall(); k += seats, index++) {
      scene.drawAt(k, terminals[index]);
    }
    return RecordedGames.scene(scene, "nagashi-" + seat);
  }

  private static boolean hasMeld(RecordedGame game, Meld.Type type) {
    return game.rounds().get(0).events().stream()
        .anyMatch(
            event ->
                event instanceof RecordEvent.Call call
                    && Meld.of(call.meld()).type() == type
                    && Meld.of(call.meld()).base() == Tiles.PEI);
  }

  /**
   * hora の yakus にある役の翻を返す。
   *
   * @param hora 和了のイベント
   * @param name 役の名前
   * @return 翻。yakus に無ければ 0
   */
  private static int yakuHan(JsonObject hora, String name) {
    for (JsonElement pair : hora.getAsJsonArray("yakus")) {
      if (pair.getAsJsonArray().get(0).getAsString().equals(name)) {
        return pair.getAsJsonArray().get(1).getAsInt();
      }
    }
    return 0;
  }

  private static List<JsonObject> parse(String text) {
    List<JsonObject> events = new ArrayList<>();
    for (String line : text.split("\n")) {
      if (!line.isBlank()) {
        events.add(JsonParser.parseString(line).getAsJsonObject());
      }
    }
    return events;
  }

  /** 1 行に 1 イベントずつ書いた文字列を返す。 */
  private static String lines(List<JsonObject> events) {
    StringJoiner text = new StringJoiner("\n", "", "\n");
    events.forEach(event -> text.add(event.toString()));
    return text.toString();
  }

  private static List<JsonObject> ofType(String text, String type) {
    return parse(text).stream().filter(event -> event.get("type").getAsString().equals(type)).toList();
  }

  private static String type(String line) {
    return JsonParser.parseString(line).getAsJsonObject().get("type").getAsString();
  }

  private static int indexOfType(List<String> lines, String type) {
    for (int index = 0; index < lines.size(); index++) {
      if (type(lines.get(index)).equals(type)) {
        return index;
      }
    }
    throw new AssertionError("no " + type + " is written");
  }

  private static int indexOf(List<RecordEvent> events, Class<? extends RecordEvent> kind) {
    for (int index = 0; index < events.size(); index++) {
      if (kind.isInstance(events.get(index))) {
        return index;
      }
    }
    throw new AssertionError("no " + kind.getSimpleName() + " in " + events);
  }

  /**
   * 文字の位置から行末までを返す（改行は含まない）。
   *
   * @param at 文字の位置（ふつうは行頭）
   */
  private static String lineAt(String text, int at) {
    return text.substring(at, text.indexOf('\n', at));
  }

  /**
   * 文字の位置を含む行の後に 1 行を挟んだ文字列を返す。
   *
   * @param at 文字の位置
   * @param line 挟む行（改行は付けない）
   */
  private static String insertAfterLine(String text, int at, String line) {
    int end = text.indexOf('\n', at) + 1;
    return text.substring(0, end) + line + "\n" + text.substring(end);
  }
}
