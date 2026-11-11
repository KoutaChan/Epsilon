package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.io.tenhou.TenhouMeldCode;
import com.epsilon.mahjong.io.tenhou.TenhouTiles;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.SceneGames;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 天鳳のライブのメッセージの記録をセッションに通す。自己対局の半荘と牌山を手で並べた局（包、抜いた北へのロン）から作った各席の記録
 * では、セッションが自分の判断のたびにエンジンと同じ合法手を Decider に見せ、記録の手を天鳳の形式で返し、全ての局の結果が卓の精算と
 * 合うことを確かめる。手で書いた記録では、t の bit が合法手を絞ること、返す物理牌、立直・槓・鳴きの流れ、NEXTREADY と終局を確かめる。
 * 三人打ちでも同じことを確かめ、北抜きの返事と補充のツモも確かめる。
 */
public class TenhouSessionTest {

  private static final Variant FOUR = Variant.TENHOU_4P;
  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String INIT_SEED = "0,0,0,0,0,124";
  private static final String EVEN_SCORES = "250,250,250,250";

  /**
   * 三人打ちの自己対局の半荘と、抜いた北へのロンの局の全ての席。北抜きと、抜いた北への応答を含む。抜いた北への応答には北で待つ聴牌が
   * 要り、自己対局では起きにくいので、牌山を手で並べた局で起こす。セッションは終局の点を確かめるので、その局は北を抜く席 0 を
   * 1000 点で始め、放銃で飛ばして半荘を終える。
   */
  @Test
  public void everySeatOfThreePlayerGamesPlaysThroughTheSession() {
    List<LiveTranscript> transcripts = new ArrayList<>();
    for (int seed = 40; seed < 46; seed++) {
      String id = "self3-" + seed;
      String xml = TenhouXml.write(RecordedGames.play(THREE, seed, id));
      for (int seat = 0; seat < THREE.seats(); seat++) {
        transcripts.add(LiveTranscript.of(THREE, xml, id, seat, seed % 2 == 1));
      }
    }
    String scene = TenhouXml.write(SceneGames.ronOnNuki(1_000, 52_000, 52_000));
    for (int seat = 0; seat < THREE.seats(); seat++) {
      transcripts.add(LiveTranscript.of(THREE, scene, "ron-on-nuki", seat, seat % 2 == 0));
    }
    int decisions = 0;
    int nuki = 0;
    int nukiResponses = 0;
    for (LiveTranscript transcript : transcripts) {
      decisions += play(THREE, transcript);
      for (LiveTranscript.Decision decision : transcript.decisions()) {
        nuki += Action.of(decision.code()).type() == Type.NUKI ? 1 : 0;
        nukiResponses += decision.kind() == EngineDecisionKind.NUKI_RESPONSE ? 1 : 0;
      }
    }
    Assert.assertTrue(decisions > 500, decisions + " decisions");
    Assert.assertTrue(nuki > 0 && nukiResponses > 0, nuki + " nuki and " + nukiResponses + " responses to nuki");
  }

  /** 自分の北抜きは N の type 10 で返す。北抜きのメッセージには答えず、続く補充のツモで次の判断をする。 */
  @Test
  public void anOwnNukiIsSentAndTheReplacementDrawDecides() {
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            THREE,
            (view, decisions, slot) -> {
              if (turns.getAndIncrement() == 0) {
                return indexOf(decisions, slot, Action.code(Type.NUKI));
              }
              Assert.assertTrue(view.last().rinshan());
              Assert.assertEquals(view.nuki(0), 1);
              return indexOf(decisions, slot, Action.discard(Tiles.CHUN, false, false, false));
            });
    Assert.assertEquals(session.handle("<GO type=\"25\" lobby=\"0\"/>"), List.of(LiveTranscript.GOK));
    session.handle(
        "<INIT seed=\"0,0,0,0,0,128\" ten=\"350,350,350,0\" oya=\"0\""
            + " hai=\"0,1,32,33,36,37,38,68,69,72,73,120,132\"/>");
    Assert.assertEquals(session.handle("<T124/>"), List.of("{\"tag\":\"N\",\"type\":10}"));
    Assert.assertEquals(
        session.handle("<N who=\"0\" m=\"" + TenhouMeldCode.encodeNuki(120) + "\"/>"), List.of());
    Assert.assertEquals(session.handle("<T125/>"), List.of("{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(turns.get(), 2);
  }

  /**
   * 北を 2 枚持って抜くと、どの北を抜いたかはサーバが決めて、N の m で知らせる。手にあった北 120 とツモった北 122 のうち、サーバが
   * 120 を抜いたなら、手に残るのは 122 で、後の手出しでは 122 を送る。
   */
  @Test
  public void theNorthTheServerTookOutLeavesTheHand() {
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            THREE,
            (view, decisions, slot) ->
                indexOf(
                    decisions,
                    slot,
                    turns.getAndIncrement() == 0
                        ? Action.code(Type.NUKI)
                        : Action.discard(Tiles.PEI, false, false, false)));
    session.handle("<GO type=\"25\" lobby=\"0\"/>");
    session.handle(
        "<INIT seed=\"0,0,0,0,0,128\" ten=\"350,350,350,0\" oya=\"0\""
            + " hai=\"0,1,32,33,36,37,38,68,69,72,73,120,132\"/>");
    Assert.assertEquals(session.handle("<T122/>"), List.of("{\"tag\":\"N\",\"type\":10}"));
    Assert.assertEquals(
        session.handle("<N who=\"0\" m=\"" + TenhouMeldCode.encodeNuki(120) + "\"/>"), List.of());
    Assert.assertEquals(session.handle("<T125/>"), List.of("{\"tag\":\"D\",\"p\":122}"));
    Assert.assertEquals(turns.get(), 2);
  }

  @Test
  public void aThreePlayerSessionRejectsAFourPlayerTable() {
    TenhouSession session = new TenhouSession(THREE, noDecision());
    Assert.assertThrows(IllegalArgumentException.class, () -> session.handle("<GO type=\"9\" lobby=\"0\"/>"));
  }

  /**
   * 四人打ちの自己対局の半荘と、牌山を手で並べた包の局の全ての席。包の局は、包の席が全額を払うツモと、鳴かせた席からの包の無いロンで、
   * どちらも払った席が飛んで半荘が終わる。
   */
  @Test
  public void everySeatOfFourPlayerGamesPlaysThroughTheSession() {
    int decisions = 0;
    for (int seed = 40; seed < 48; seed++) {
      String id = "self-" + seed;
      String xml = TenhouXml.write(RecordedGames.play(FOUR, seed, id));
      for (int seat = 0; seat < FOUR.seats(); seat++) {
        decisions += play(FOUR, LiveTranscript.of(FOUR, xml, id, seat, seed % 2 == 0));
      }
    }
    for (RecordedGame game : List.of(SceneGames.paoTsumo(), SceneGames.paoFeederRon())) {
      String xml = TenhouXml.write(game);
      for (int seat = 0; seat < FOUR.seats(); seat++) {
        decisions += play(FOUR, LiveTranscript.of(FOUR, xml, game.meta().id(), seat, seat % 2 == 0));
      }
    }
    Assert.assertTrue(decisions > 1_000, decisions + " decisions");
  }

  /** サーバの点の移動が精算と合わない局と、和了点を読めない局は数えておき、次の INIT からサーバの卓の状態で続ける。 */
  @Test
  public void serverResultsThatDisagreeAreCountedAndPlayGoesOn() {
    String xml = TenhouXml.write(RecordedGames.play(FOUR, 40, "self-40"));
    LiveTranscript transcript = LiveTranscript.of(FOUR, xml, "self-40", 0, false);
    Pattern sc = Pattern.compile("sc=\"(-?\\d+),(-?\\d+),(-?\\d+),(-?\\d+),");
    Pattern ten = Pattern.compile("ten=\"(\\d+),(\\d+),");
    List<LiveTranscript.Line> lines = new ArrayList<>(transcript.lines());
    int moved = firstAfter(lines, "<AGARI", 0);
    Matcher shift = sc.matcher(lines.get(moved).message());
    Assert.assertTrue(shift.find());
    String shifted =
        "sc=\"" + shift.group(1) + "," + (Integer.parseInt(shift.group(2)) + 1) + "," + shift.group(3) + ","
            + (Integer.parseInt(shift.group(4)) - 1) + ",";
    lines.set(moved, new LiveTranscript.Line(shift.replaceFirst(shifted), lines.get(moved).replies()));
    int misread = firstAfter(lines, "<AGARI", firstAfter(lines, "<INIT", moved));
    Matcher points = ten.matcher(lines.get(misread).message());
    Assert.assertTrue(points.find());
    String wrong = "ten=\"" + points.group(1) + "," + (Integer.parseInt(points.group(2)) + 100) + ",";
    lines.set(misread, new LiveTranscript.Line(points.replaceFirst(wrong), lines.get(misread).replies()));
    TenhouSession session = new TenhouSession(FOUR, scripted(new ArrayDeque<>(transcript.decisions())));
    for (LiveTranscript.Line line : lines) {
      Assert.assertEquals(session.handle(line.message()), line.replies(), line.message());
    }
    Assert.assertTrue(session.ended());
    Assert.assertEquals(session.mismatches(), 2);
  }

  @DataProvider
  public Object[][] rejoins() {
    return new Object[][] {
      {"{\"tag\":\"REJOIN\",\"t\":\"1212,9,r\"}", "1212,9,r"},
      {"<REJOIN t=\"C12345678,9,r\"/>", "C12345678,9,r"}
    };
  }

  @Test(dataProvider = "rejoins")
  public void rejoinKeepsTheServerTargetAndFlags(String request, String target) {
    TenhouSession session = new TenhouSession(FOUR, noDecision());
    Assert.assertEquals(session.handle(request), List.of("{\"tag\":\"JOIN\",\"t\":\"" + target + "\"}"));
    Assert.assertEquals(session.handle("{\"tag\":\"REJOIN\"}"), List.of());
    Assert.assertEquals(session.handle("<REJOIN t=\"\"/>"), List.of());
    Assert.assertEquals(session.handle("<SAIKAI/>"), List.of());
    Assert.assertFalse(session.ended());
  }

  /** GO には GOK を返し、対局の始まりと局の終わりには NEXTREADY を 1 回だけ返す。読めない結果は数えて続け、owari で終局する。 */
  @Test
  public void readinessIsSentOncePerPauseAndOwariEndsTheGame() {
    TenhouSession session = new TenhouSession(FOUR, noDecision());
    Assert.assertEquals(session.handle("{\"tag\":\"GO\",\"type\":9,\"lobby\":1212}"), List.of(LiveTranscript.GOK));
    Assert.assertEquals(session.handle("<TAIKYOKU oya=\"0\"/>"), List.of(LiveTranscript.NEXT_READY));
    Assert.assertEquals(session.handle("{\"tag\":\"TAIKYOKU\"}"), List.of());
    session.handle(init(0, "0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handle("<AGARI who=\"1\" fromWho=\"3\"/>"), List.of(LiveTranscript.NEXT_READY));
    Assert.assertEquals(session.handle("<AGARI who=\"2\" fromWho=\"3\"/>"), List.of());
    Assert.assertEquals(session.mismatches(), 1);
    session.handle(init(0, "0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handle("<RYUUKYOKU/>"), List.of(LiveTranscript.NEXT_READY));
    Assert.assertFalse(session.ended());
    session.handle(init(0, "0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handle("<RYUUKYOKU owari=\"250,0,250,0,250,0,250,0\"/>"), List.of());
    Assert.assertTrue(session.ended());
  }

  @Test
  public void otherTenhouGameTypesAreRejected() {
    TenhouSession session = new TenhouSession(FOUR, noDecision());
    Assert.assertThrows(IllegalArgumentException.class, () -> session.handle("<GO type=\"25\" lobby=\"0\"/>"));
  }

  @DataProvider
  public Object[][] responseOffers() {
    return new Object[][] {
      {"", List.<Type>of(), List.<String>of()},
      {"1", List.of(Type.PON, Type.PASS), List.of("{\"tag\":\"N\",\"type\":1,\"hai0\":4,\"hai1\":5}")},
      {"4", List.of(Type.CHI, Type.PASS), List.of("{\"tag\":\"N\",\"type\":3,\"hai0\":8,\"hai1\":12}")},
      {"5", List.of(Type.PON, Type.CHI, Type.PASS), List.of("{\"tag\":\"N\",\"type\":1,\"hai0\":4,\"hai1\":5}")},
      {"8", List.<Type>of(), List.of(LiveTranscript.PASS)}
    };
  }

  /**
   * 上家が打った 2m に、エンジンはポンとチーを許す。Decider にはサーバの t が示した手だけを見せる。t が無ければ答えず、
   * t が示した手がエンジンの合法手に無ければ見送る。
   */
  @Test(dataProvider = "responseOffers")
  public void theServerBitsRestrictTheResponse(String offers, List<Type> shown, List<String> replies) {
    List<List<Type>> seen = new ArrayList<>();
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) -> {
              seen.add(types(decisions, slot));
              return 0;
            });
    session.handle(init(3, "4,5,8,12,36,40,44,72,76,80,108,112,116"));
    session.handle("<W/>");
    String discard = offers.isEmpty() ? "<G6/>" : "<G6 t=\"" + offers + "\"/>";
    Assert.assertEquals(session.handle(discard), replies);
    Assert.assertEquals(seen, shown.isEmpty() ? List.of() : List.of(shown));
  }

  /** 自分のツモの t。立直（32）とツモ和了（16）は bit があるときだけ出せる。打牌はいつでも出せる。 */
  @Test
  public void theDrawBitsGateRiichiAndTsumo() {
    List<List<Type>> seen = new ArrayList<>();
    Decider decider =
        (view, decisions, slot) -> {
          seen.add(types(decisions, slot));
          return decisions.legalCount(slot) - 1;
        };
    String hand = "0,4,8,36,40,44,72,76,80,96,100,104,108";
    TenhouSession plain = new TenhouSession(FOUR, decider);
    plain.handle(init(0, hand));
    plain.handle("<T132/>");
    TenhouSession riichi = new TenhouSession(FOUR, decider);
    riichi.handle(init(0, hand));
    riichi.handle("<T132 t=\"32\"/>");
    TenhouSession unoffered = new TenhouSession(FOUR, decider);
    unoffered.handle(init(0, hand));
    unoffered.handle("<T109/>");
    TenhouSession tsumo = new TenhouSession(FOUR, decider);
    tsumo.handle(init(0, hand));
    tsumo.handle("<T109 t=\"48\"/>");
    Assert.assertFalse(seen.get(0).contains(Type.RIICHI_DAHAI));
    Assert.assertTrue(seen.get(1).contains(Type.RIICHI_DAHAI));
    Assert.assertFalse(seen.get(2).contains(Type.TSUMO_AGARI));
    Assert.assertTrue(seen.get(3).contains(Type.TSUMO_AGARI));
    Assert.assertTrue(seen.stream().allMatch(types -> types.contains(Type.DAHAI)));
  }

  @DataProvider
  public Object[][] ownDiscardEchoes() {
    return new Object[][] {{"{\"tag\":\"D38\"}"}, {"<d38/>"}};
  }

  /** ツモ切りした物理牌は手牌に残らない。後の手出しでは、手牌に残っていたほうの物理牌を送る。 */
  @Test(dataProvider = "ownDiscardEchoes")
  public void aDiscardedDrawnCopyLeavesTheHand(String echo) {
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) -> {
              boolean first = turns.getAndIncrement() == 0;
              if (!first) {
                Assert.assertTrue(view.holds(37));
                Assert.assertFalse(view.holds(38));
                Assert.assertEquals(view.count(Tiles.P1), 1);
              }
              return indexOf(decisions, slot, Action.discard(Tiles.P1, false, first, false));
            });
    session.handle(init(0, "0,4,8,37,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handle("<T38/>"), List.of("{\"tag\":\"D\",\"p\":38}"));
    session.handle(echo);
    goAround(session);
    Assert.assertEquals(session.handle("<T106/>"), List.of("{\"tag\":\"D\",\"p\":37}"));
    Assert.assertEquals(session.handle("<D37/>"), List.of());
    Assert.assertEquals(turns.get(), 2);
  }

  /** 立直は宣言を打牌の前に送り、立直が成立したら供託を払う（親の第一打なのでダブル立直）。立直の後はツモ切りしかできない。 */
  @Test
  public void riichiIsSentBeforeItsDiscardAndPaysOnAcceptance() {
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) -> {
              if (turns.getAndIncrement() == 0) {
                return indexOf(decisions, slot, Action.discard(Tiles.CHUN, false, true, true));
              }
              Assert.assertEquals(view.riichi(0), RiichiState.DOUBLE_RIICHI);
              Assert.assertEquals(view.score(0), 24_000);
              Assert.assertEquals(view.kyotaku(), 1);
              Assert.assertEquals(types(decisions, slot), List.of(Type.DAHAI));
              return 0;
            });
    session.handle(init(0, "0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(
        session.handle("<T132 t=\"32\"/>"), List.of("{\"tag\":\"REACH\"}", "{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(session.handle("<REACH who=\"0\" step=\"1\"/>"), List.of());
    Assert.assertEquals(session.handle("<D132/>"), List.of());
    Assert.assertEquals(session.handle("<REACH who=\"0\" ten=\"240,250,250,250\" step=\"2\"/>"), List.of());
    goAround(session);
    Assert.assertEquals(session.handle("<T53/>"), List.of("{\"tag\":\"D\",\"p\":53}"));
    Assert.assertEquals(turns.get(), 2);
  }

  @DataProvider
  public Object[][] addedKans() {
    return new Object[][] {{true, false}, {false, true}};
  }

  /**
   * ポンした 5m に 4 枚目を足す加槓。加槓は t の bit が無くても出せ、足す牌（赤なら赤）を送る。後めくりの表示牌は、打牌の前に届いても
   * 打牌の後に受け取る。
   */
  @Test(dataProvider = "addedKans")
  public void anAddedKanSendsTheFourthTile(boolean addedRed, boolean json) {
    int called = addedRed ? 17 : 16;
    int added = addedRed ? 16 : 19;
    int[] hand = addedRed ? new int[] {18, 19} : new int[] {17, 18};
    Meld.RelativeSource from = addedRed ? Meld.RelativeSource.TOIMEN : Meld.RelativeSource.SHIMOCHA;
    Meld pon = Meld.pon(Tiles.M5, from, addedRed ? Meld.AkaSource.NONE : Meld.AkaSource.CALLED_TILE);
    int ponCode = new TenhouMeldCode(pon, called, hand).encode(FOUR, 0);
    int kakanCode =
        new TenhouMeldCode(Meld.kakan(pon, addedRed), called, new int[] {added}).encode(FOUR, 0);
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) ->
                switch (turns.getAndIncrement()) {
                  case 0 -> indexOf(decisions, slot, Action.code(Type.PON, Tiles.M5, TileSelection.NON_AKA));
                  case 1 -> indexOf(decisions, slot, Action.discard(Tiles.M1, false, false, false));
                  case 2 -> indexOf(decisions, slot, Action.code(Type.KAKAN, Tiles.M5, TileSelection.NOT_APPLICABLE));
                  case 3 -> {
                    Assert.assertTrue(view.last().rinshan());
                    yield indexOf(decisions, slot, Action.discard(Tiles.CHUN, false, true, false));
                  }
                  default -> throw new AssertionError("No more decisions");
                });
    int dealer = addedRed ? 2 : 1;
    session.handle(message(json, "INIT", "seed", INIT_SEED, "ten", EVEN_SCORES, "oya", Integer.toString(dealer), "hai",
        hand[0] + "," + hand[1] + ",0,4,8,36,40,44,72,76,80,108,109"));
    if (addedRed) {
      session.handle(message(json, "V"));
    } else {
      session.handle(message(json, "U"));
    }
    Assert.assertEquals(
        session.handle(message(json, (addedRed ? "F" : "E") + called, "t", "1")),
        List.of("{\"tag\":\"N\",\"type\":1,\"hai0\":" + hand[0] + ",\"hai1\":" + hand[1] + "}"));
    Assert.assertEquals(
        session.handle(message(json, "N", "who", "0", "m", Integer.toString(ponCode))),
        List.of("{\"tag\":\"D\",\"p\":0}"));
    session.handle(message(json, "D0"));
    goAround(session);
    Assert.assertEquals(
        session.handle(message(json, "T" + added)), List.of("{\"tag\":\"N\",\"type\":5,\"hai\":" + added + "}"));
    Assert.assertEquals(session.handle(message(json, "N", "who", "0", "m", Integer.toString(kakanCode))), List.of());
    Assert.assertEquals(session.handle(message(json, "T132")), List.of("{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(session.handle(message(json, "DORA", "hai", "128")), List.of());
    Assert.assertEquals(session.handle(message(json, "D132")), List.of());
    Assert.assertEquals(turns.get(), 4);
  }

  /** 暗槓は t の bit が無くても出せ、槓にする種類の牌を送る。表示牌はすぐにめくり、続く嶺上のツモで判断する。 */
  @Test
  public void aClosedKanWaitsForItsEchoAndTheReplacementDraw() {
    AtomicInteger turns = new AtomicInteger();
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) -> {
              if (turns.getAndIncrement() == 0) {
                return indexOf(decisions, slot, Action.code(Type.ANKAN, Tiles.P2, TileSelection.NOT_APPLICABLE));
              }
              Assert.assertTrue(view.last().rinshan());
              Assert.assertEquals(view.meldCount(0), 1);
              Assert.assertEquals(view.indicatorCount(), 2);
              return indexOf(decisions, slot, Action.discard(Tiles.CHUN, false, true, false));
            });
    session.handle(init(0, "0,4,8,36,40,41,42,72,76,80,108,109,112"));
    Assert.assertEquals(session.handle("<T43/>"), List.of("{\"tag\":\"N\",\"type\":4,\"hai\":40}"));
    Assert.assertEquals(session.handle("<N who=\"0\" m=\"10240\"/>"), List.of());
    Assert.assertEquals(session.handle("<DORA hai=\"128\"/>"), List.of());
    Assert.assertEquals(session.handle("<T132/>"), List.of("{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(turns.get(), 2);
  }

  @DataProvider
  public Object[][] openKanOffers() {
    return new Object[][] {
      {"", List.of()},
      {"1", List.of(LiveTranscript.PASS)},
      {"3", List.of("{\"tag\":\"N\",\"type\":2}")}
    };
  }

  /** 大明槓は t の bit 2 があるときだけ出せる。 */
  @Test(dataProvider = "openKanOffers")
  public void anOpenKanNeedsItsBit(String offers, List<String> replies) {
    TenhouSession session =
        new TenhouSession(
            FOUR,
            (view, decisions, slot) -> {
              List<Type> types = types(decisions, slot);
              return types.contains(Type.DAIMINKAN) ? types.indexOf(Type.DAIMINKAN) : types.indexOf(Type.PASS);
            });
    session.handle(init(1, "0,4,8,36,40,44,72,76,80,108,124,125,127"));
    session.handle("<U/>");
    Assert.assertEquals(session.handle(offers.isEmpty() ? "<E126/>" : "<E126 t=\"" + offers + "\"/>"), replies);
  }

  @Test
  public void liveTagsNameTheRelativeSeatTheTileAndTsumogiri() {
    Assert.assertEquals(
        TenhouCodec.move("T52", FOUR), new TenhouCodec.Move(TenhouTiles.Move.DRAW, 0, 52, false));
    Assert.assertEquals(TenhouCodec.move("W", FOUR).tile(), -1);
    TenhouCodec.Move discard = TenhouCodec.move("e52", FOUR);
    Assert.assertEquals(discard.seat(), 1);
    Assert.assertTrue(discard.tsumogiri());
    Assert.assertNull(TenhouCodec.move("DORA", FOUR));
    Assert.assertNull(TenhouCodec.move("UN", FOUR));
    Assert.assertNull(TenhouCodec.move("d", FOUR));
    Assert.assertEquals(TenhouCodec.parse("{\"tag\":\"T52\",\"t\":64}").integer("t"), 64);
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouCodec.parse("not a message"));
    Assert.assertThrows(IllegalArgumentException.class, () -> TenhouCodec.parse("{\"t\":1}"));
  }

  /** 記録をセッションに通し、Decider が選んだ判断の数を返す。 */
  private static int play(Variant variant, LiveTranscript transcript) {
    ArrayDeque<LiveTranscript.Decision> expected = new ArrayDeque<>(transcript.decisions());
    TenhouSession session = new TenhouSession(variant, scripted(expected));
    for (LiveTranscript.Line line : transcript.lines()) {
      Assert.assertEquals(session.handle(line.message()), line.replies(), line.message());
    }
    Assert.assertTrue(session.ended());
    Assert.assertEquals(session.mismatches(), 0);
    Assert.assertEquals(expected.size(), 0, "decisions left");
    return transcript.decisions().size();
  }

  /** 記録の判断を順に選ぶ。そのたびに、エンジンと同じ種類と合法手で判断を求められたことを確かめる。 */
  private static Decider scripted(ArrayDeque<LiveTranscript.Decision> expected) {
    return (view, slots, slot) -> {
      LiveTranscript.Decision decision = expected.removeFirst();
      Assert.assertEquals(view.kind(), decision.kind());
      Assert.assertEquals(codes(slots, slot), decision.legal());
      return decision.legal().indexOf(decision.code());
    };
  }

  /**
   * 指定した位置より後で、指定した文字列で始まる最初のメッセージの位置を返す。
   *
   * @param prefix メッセージの始まり
   * @param from この位置より後を探す
   */
  private static int firstAfter(List<LiveTranscript.Line> lines, String prefix, int from) {
    for (int index = from + 1; index < lines.size(); index++) {
      if (lines.get(index).message().startsWith(prefix)) {
        return index;
      }
    }
    throw new AssertionError("No " + prefix + " after line " + from);
  }

  /** 他家 3 人が順にツモって北を打つ。 */
  private static void goAround(TenhouSession session) {
    for (String message : List.of("<U/>", "<E120/>", "<V/>", "<F121/>", "<W/>", "<G122/>")) {
      Assert.assertEquals(session.handle(message), List.of(), message);
    }
  }

  private static String init(int dealer, String hand) {
    return "<INIT seed=\"" + INIT_SEED + "\" ten=\"" + EVEN_SCORES + "\" oya=\"" + dealer + "\" hai=\"" + hand + "\"/>";
  }

  private static String message(boolean json, String tag, String... fields) {
    StringBuilder text = new StringBuilder(json ? "{\"tag\":\"" + tag + "\"" : "<" + tag);
    for (int index = 0; index < fields.length; index += 2) {
      String key = fields[index];
      String value = fields[index + 1];
      text.append(json ? ",\"" + key + "\":\"" + value + "\"" : " " + key + "=\"" + value + "\"");
    }
    return text.append(json ? "}" : "/>").toString();
  }

  private static List<Integer> codes(Decisions decisions, int slot) {
    List<Integer> codes = new ArrayList<>();
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      codes.add(decisions.legal(slot, index));
    }
    return codes;
  }

  private static List<Type> types(Decisions decisions, int slot) {
    List<Type> types = new ArrayList<>();
    for (int code : codes(decisions, slot)) {
      Type type = Action.of(code).type();
      if (!types.contains(type)) {
        types.add(type);
      }
    }
    return types;
  }

  private static int indexOf(Decisions decisions, int slot, int code) {
    int index = codes(decisions, slot).indexOf(code);
    Assert.assertTrue(index >= 0, Action.of(code) + " is not offered");
    return index;
  }

  private static Decider noDecision() {
    return (view, decisions, slot) -> {
      throw new AssertionError("No decision is due");
    };
  }
}
