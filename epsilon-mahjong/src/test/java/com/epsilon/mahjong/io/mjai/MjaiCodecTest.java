package com.epsilon.mahjong.io.mjai;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Variant;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import org.testng.Assert;
import org.testng.annotations.Test;

/** mjai に書いて読み直すテスト。読んだ半荘は、同じエンジンで記録どおりに再生できる。 */
public class MjaiCodecTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** 席ごとの、面子の形にならない不聴の配牌。 */
  private static final String[] NOTEN = {
    "147m258p369s1234z", "147p258s369s1234z", "258p147s369m1234z", "369p258s147m1234z"
  };

  /**
   * 自己対局の記録を mjai に書いて読むと、局の始まり、局の結果（裏ドラ表示牌の物理 id を除く）、点の移動、終局の点が元と同じになり、
   * 再生も記録と合う。読んだ半荘をもう一度書くと同じ文字列になり、それを読み直すと同じ半荘になる。
   */
  @Test
  public void gamesSurviveAWriteAndARead() {
    List<RecordedGame> games = new ArrayList<>();
    for (int seed = 0; seed < 30; seed++) {
      games.add(RecordedGames.play(VARIANT, seed, "self-" + seed));
    }
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    for (RecordedGame game : games) {
      String id = game.meta().id();
      String text = MjaiCodec.write(game);
      RecordedGame read = MjaiCodec.read(text, id);
      Assert.assertEquals(MjaiCodec.write(read), text, id);
      Assert.assertEquals(MjaiCodec.read(text, id), read, id);
      Assert.assertEquals(read.finalScores(), game.finalScores(), id);
      Assert.assertEquals(read.rounds().size(), game.rounds().size(), id);
      for (int index = 0; index < game.rounds().size(); index++) {
        RecordedRound before = game.rounds().get(index);
        RecordedRound after = read.rounds().get(index);
        RoundResult result = after.result();
        Assert.assertEquals(after.start(), before.start(), id);
        Assert.assertEquals(after.events().size(), before.events().size(), id);
        Assert.assertEquals(result.kind(), before.result().kind(), id);
        Assert.assertEquals(result.abortive(), before.result().abortive(), id);
        Assert.assertEquals(result.wins(), before.result().wins(), id);
        Assert.assertEquals(result.loser(), before.result().loser(), id);
        Assert.assertEquals(result.tenpaiMask(), before.result().tenpaiMask(), id);
        Assert.assertEquals(result.nagashiMask(), before.result().nagashiMask(), id);
        Assert.assertEquals(result.deltas(), before.result().deltas(), id);
        Assert.assertEquals(result.uraCount(), before.result().uraCount(), id);
      }
      driver.replay(read, ws);
    }
  }

  /**
   * 打ち手の名前は start_game の names から読む（null は空の文字列にする）。書くときは、半荘の打ち手の名前を names に書く。names の
   * 無い start_game は名前の無い打ち手として読み、人数は最初の局の点の数で決める。
   */
  @Test
  public void namesComeFromTheStartGame() {
    RecordedGame played = RecordedGames.play(VARIANT, 0, "self-0");
    RecordedGame game =
        new RecordedGame(
            VARIANT,
            played.rounds(),
            played.finalScores(),
            new RecordedGame.Meta("self-0", RecordedGame.Source.TENHOU, List.of("A", "B", "C", "D")));
    String text = MjaiCodec.write(game);
    Assert.assertEquals(
        MjaiCodec.read(text, "named").meta(),
        new RecordedGame.Meta("named", RecordedGame.Source.MJAI, List.of("A", "B", "C", "D")));
    String unnamed = text.replace("\"names\":[\"A\",\"B\",\"C\",\"D\"]", "\"names\":[\"A\",null,\"C\",\"D\"]");
    Assert.assertNotEquals(unnamed, text);
    Assert.assertEquals(MjaiCodec.read(unnamed, "unnamed").meta().names(), List.of("A", "", "C", "D"));
    String bare = text.replace(",\"names\":[\"A\",\"B\",\"C\",\"D\"]", "").replace("\"names\":[\"A\",\"B\",\"C\",\"D\"],", "");
    Assert.assertNotEquals(bare, text);
    RecordedGame read = MjaiCodec.read(bare, "bare");
    Assert.assertEquals(read.variant(), VARIANT);
    Assert.assertEquals(read.meta().names(), List.of("", "", "", ""), "a start_game without names is unnamed");
  }

  /** hora の点が翻・符と合わない牌譜は読まない。 */
  @Test
  public void aHoraWhosePointsDisagreeIsRejected() {
    RecordedGame game = RecordedGames.play(VARIANT, 0, "self-0");
    String text = MjaiCodec.write(game);
    int at = text.indexOf("\"hora_points\":");
    int end = text.indexOf(',', at);
    String wrong = text.substring(0, at) + "\"hora_points\":100" + text.substring(end);
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(wrong, game.meta().id()));
  }

  /** 最後の打牌と違う牌を鳴いた記録は読まない。 */
  @Test
  public void aCallOfAnotherTileIsRejected() {
    RecordedGame game = RecordedGames.play(VARIANT, 0, "self-0");
    String text = MjaiCodec.write(game);
    int at = text.indexOf("{\"type\":\"pon\"");
    int pai = text.indexOf("\"pai\":\"", at) + "\"pai\":\"".length();
    String other = text.charAt(pai) == '1' ? "2" : "1";
    String wrong = text.substring(0, pai) + other + text.substring(pai + 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> MjaiCodec.read(wrong, game.meta().id()));
  }

  /**
   * 内訳の無いダブロンで、立直していない 1 人目の裏ドラ表示牌のフィールドが空でも、立直した 2 人目の裏ドラ表示牌を読む。内訳を書かない
   * プログラムには、立直していない和了者にも空のフィールドを書くものがある。2 人目は裏ドラが 1 つ乗り、読んだ局は記録どおりに再生できる。
   */
  @Test
  public void anEmptyUraListOfTheFirstWinnerDoesNotHideTheSecondWinnersUra() {
    Scenes scene =
        Scenes.at(VARIANT, 1, 2, 0, 25_000, 25_000, 25_000, 25_000)
            .hand(1, "5p147m147s369m123z")
            .hand(2, "123m456m789m46p11z")
            .hand(3, "234s567s678m34p55s")
            .hand(0, "19p258m1479s4567z")
            .draws("9s9m9p9s3z")
            .indicator("7z")
            .ura(0, "1s");
    RecordedGame game =
        RecordedGames.scene(scene, "empty-ura", "3:RIICHI_TSUMOGIRI(9p)", "1:DAHAI(5p)", "2:RON_AGARI", "3:RON_AGARI");
    RoundResult played = game.rounds().get(0).result();
    Assert.assertEquals(played.wins().size(), 2);
    Assert.assertEquals(played.wins().get(0).seat(), 2);
    Assert.assertEquals(played.wins().get(0).ura(), 0);
    Assert.assertEquals(played.wins().get(1).seat(), 3);
    Assert.assertEquals(played.wins().get(1).ura(), 1);
    StringJoiner text = new StringJoiner("\n", "", "\n");
    for (String line : MjaiCodec.write(game).split("\n")) {
      if (line.isBlank()) {
        continue;
      }
      JsonObject event = JsonParser.parseString(line).getAsJsonObject();
      if (event.get("type").getAsString().equals("hora")) {
        for (String key : List.of("yakus", "fan", "fu", "hora_points", "pao", "pai")) {
          event.remove(key);
        }
        JsonElement ura = event.remove("uradora_markers");
        event.add("ura_markers", ura != null ? ura : new JsonArray());
      }
      text.add(event.toString());
    }
    Assert.assertTrue(text.toString().contains("\"ura_markers\":[]"));
    RecordedGame read = MjaiCodec.read(text.toString(), "empty-ura");
    RoundResult result = read.rounds().get(0).result();
    Assert.assertEquals(result.uraCount(), 1);
    Assert.assertEquals(result.deltas(), played.deltas());
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    Boundary boundary = driver.start(read.rounds().get(0), ws);
    while (boundary == Boundary.DECISIONS) {
      boundary = driver.step(ws);
    }
    Assert.assertTrue(result.sameDeltas(driver.engine().roundEnd()));
    Assert.assertTrue(played.matches(driver.engine().roundEnd(), game.rounds().get(0).riichiMask()));
  }

  /**
   * 和了の内訳も、流局の reason・tenpais も、end_game の scores も書かないプログラムの記録を読む。見本は、自己対局の半荘と、九種九牌の
   * 局と、供託が 1 本残ったまま荒牌で終わるオーラスの局。hora は点を持たない和了に、ryukyoku は終わり方を持たない流局になる。終局の点は
   * 最後の局から求め、残った供託はトップが取る。
   */
  @Test
  public void recordsWithoutDetailsOrFinalScoresReplay() {
    int draws = 0;
    for (int seed = 0; seed < 30; seed++) {
      RecordedGame game = withoutDetails(RecordedGames.play(VARIANT, seed, "self-" + seed));
      draws += (int) game.rounds().stream().filter(round -> round.result().kind() == null).count();
      new ReplayDriver(VARIANT).replay(game, new RuleWorkspace(VARIANT));
    }
    Assert.assertTrue(draws > 0, "no draw was read");

    RecordedGame nine = nineTerminals();
    Assert.assertEquals(nine.rounds().get(0).result().abortive(), AbortiveDraw.NINE_TERMINALS);
    withoutDetails(nine);

    RecordedGame last = depositLeft();
    Assert.assertEquals(last.rounds().get(0).result().kind(), EndKind.EXHAUSTIVE);
    Assert.assertEquals(last.rounds().get(0).result().deltas(), new int[4]);
    Assert.assertEquals(last.finalScores(), new int[] {41_000, 20_000, 20_000, 19_000});
    withoutDetails(last);
  }

  /**
   * ryukyoku の reason は、別名（RiichiEnv の名前と天鳳の名前）でも同じ終わり方に読む。type が none のイベントは、牌譜のどこにあっても
   * 読み飛ばす。
   */
  @Test
  public void reasonAliasesAndNoneEventsReadTheSameGame() {
    String nine = MjaiCodec.write(nineTerminals());
    String written = reason("kyushukyuhai");
    Assert.assertTrue(nine.contains(written), nine);
    Map<AbortiveDraw, List<String>> names = new EnumMap<>(AbortiveDraw.class);
    names.put(AbortiveDraw.NINE_TERMINALS, List.of("kyushukyuhai", "kyushu_kyuhai", "yao9"));
    names.put(AbortiveDraw.FOUR_WINDS, List.of("sufonrenta", "kaze4"));
    names.put(AbortiveDraw.FOUR_RIICHI, List.of("suchareach", "reach4"));
    names.put(AbortiveDraw.FOUR_KANS, List.of("sukaikan", "kan4"));
    names.put(AbortiveDraw.TRIPLE_RON, List.of("sanchaho", "ron3"));
    Assert.assertEquals(names.keySet(), EnumSet.allOf(AbortiveDraw.class));
    names.forEach(
        (abortive, reasons) -> {
          RecordedGame named = MjaiCodec.read(nine.replace(written, reason(reasons.get(0))), "named");
          Assert.assertEquals(named.rounds().get(0).result().abortive(), abortive);
          for (String alias : reasons) {
            Assert.assertEquals(MjaiCodec.read(nine.replace(written, reason(alias)), "named"), named, alias);
          }
        });

    String draw = MjaiCodec.write(depositLeft());
    String exhaustive = draw.replace(reason("fanpai"), reason("exhaustive_draw"));
    Assert.assertNotEquals(exhaustive, draw);
    Assert.assertEquals(MjaiCodec.read(exhaustive, "draw"), MjaiCodec.read(draw, "draw"));

    String text = MjaiCodec.write(RecordedGames.play(VARIANT, 0, "self-0"));
    String none = "{\"type\":\"none\"}\n";
    Assert.assertEquals(MjaiCodec.read(none + text.replace("\n", "\n" + none), "none"), MjaiCodec.read(text, "none"));
  }

  /**
   * 半荘を mjai に書き、和了の内訳・流局の reason と tenpais・end_game の scores を落として読み、元の半荘と比べる。配列でも 1 行
   * 1 イベントでも同じ半荘に読め、終局の点、和了者、放銃者、点の移動は元と同じになる。読んだ局は記録どおりに再生でき、エンジンの局末は
   * 元の結果の内訳と合う。内訳が無いので、天鳳の XML には書けない。
   *
   * @param original 内訳を持つ半荘
   * @return 内訳を落として読んだ半荘
   */
  private static RecordedGame withoutDetails(RecordedGame original) {
    String id = original.meta().id();
    List<JsonObject> events = new ArrayList<>();
    for (String line : MjaiCodec.write(original).split("\n")) {
      JsonObject event = JsonParser.parseString(line).getAsJsonObject();
      switch (event.get("type").getAsString()) {
        case "hora" -> {
          for (String key : List.of("yakus", "fan", "fu", "hora_points", "pao", "pai")) {
            event.remove(key);
          }
          JsonElement ura = event.remove("uradora_markers");
          if (ura != null) {
            event.add("ura_markers", ura);
          }
        }
        case "ryukyoku" -> {
          event.remove("reason");
          event.remove("tenpais");
        }
        case "end_game" -> event.remove("scores");
        default -> {}
      }
      events.add(event);
    }
    JsonArray array = new JsonArray();
    StringJoiner lines = new StringJoiner("\n", "", "\n");
    for (JsonObject event : events) {
      array.add(event);
      lines.add(event.toString());
    }
    RecordedGame game = MjaiCodec.read(lines.toString(), id);
    Assert.assertEquals(MjaiCodec.read(array.toString(), id), game, id);
    Assert.assertEquals(game.finalScores(), original.finalScores(), id);
    Assert.assertEquals(game.rounds().size(), original.rounds().size(), id);
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    for (int index = 0; index < game.rounds().size(); index++) {
      RecordedRound before = original.rounds().get(index);
      RecordedRound round = game.rounds().get(index);
      RoundResult recorded = before.result();
      RoundResult read = round.result();
      String where = id + " " + round.start();
      Assert.assertFalse(read.detailed(), where);
      Assert.assertEquals(read.loser(), recorded.loser(), where);
      Assert.assertEquals(read.deltas(), recorded.deltas(), where);
      if (recorded.win()) {
        Assert.assertEquals(read.kind(), recorded.kind(), where);
        Assert.assertEquals(
            read.wins(), recorded.wins().stream().map(win -> RoundResult.Win.unscored(win.seat())).toList(), where);
        Assert.assertEquals(read.uraCount(), recorded.uraCount(), where);
      } else {
        Assert.assertEquals(read, RoundResult.drawWithoutDetails(recorded.deltas()), where);
      }
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) {
        boundary = driver.step(ws);
      }
      Assert.assertTrue(recorded.matches(driver.engine().roundEnd(), before.riichiMask()), where);
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouXml.write(game));
    return game;
  }

  /** 親の席 0 が第一ツモで九種九牌を選んで流れる 1 局。 */
  private static RecordedGame nineTerminals() {
    return RecordedGames.scene(
        Scenes.first(VARIANT)
            .hand(0, "19m19p19s1234567z")
            .hand(1, NOTEN[1])
            .hand(2, NOTEN[2])
            .hand(3, NOTEN[3])
            .draws("5p"),
        "nine-terminals",
        "0:KYUSHU_KYUHAI");
  }

  /** 供託が 1 本残ったオーラスで、全員が不聴のままツモ切って荒牌になる 1 局。親が不聴でトップが返し点を超えているので、ここで終局する。 */
  private static RecordedGame depositLeft() {
    return RecordedGames.scene(
        Scenes.at(VARIANT, VARIANT.finalRound(), 0, 1, 40_000, 20_000, 20_000, 19_000)
            .hand(0, NOTEN[0])
            .hand(1, NOTEN[1])
            .hand(2, NOTEN[2])
            .hand(3, NOTEN[3]),
        "deposit-left");
  }

  /** ryukyoku の reason のフィールド。 */
  private static String reason(String name) {
    return "\"reason\":\"" + name + "\"";
  }
}
