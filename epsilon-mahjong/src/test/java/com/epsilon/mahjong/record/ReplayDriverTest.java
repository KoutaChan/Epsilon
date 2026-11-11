package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 牌譜の再生のテスト。記録の手を同じエンジンで進め、合法でない手と局末の不一致を見つける。 */
public class ReplayDriverTest {

  private static final Variant VARIANT = RoundBuilder.VARIANT;

  /** 南 4 局（親は席 3）。 */
  private static final RoundStart SOUTH_FOUR =
      new RoundStart(VARIANT, 7, 0, 0, new int[] {24_000, 29_000, 23_000, 24_000});

  /** 三人打ちの東 1 局（親は席 0）。 */
  private static final RoundStart SANMA_EAST_ONE = RoundStart.first(Variant.TENHOU_3P);

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** 自己対局の記録（四人打ちと三人打ち）は、どの局も合法な手で進み、局末と次の卓も合う。 */
  @Test(dataProvider = "variants")
  public void selfPlayRecordsReplayCleanly(Variant variant) {
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    for (int seed = 0; seed < 30; seed++) {
      driver.replay(RecordedGames.play(variant, seed, "self-" + seed), ws);
    }
  }

  /**
   * ポンが優先されてチーが消えた席は、選んだ手が記録に無い（{@code UNOBSERVED}）。ロンは記録どおり。南 4 局の和了で終局し、終局の点も
   * 記録と合う。
   */
  @Test
  public void aPonCutsOffAChiAndTheRonEndsTheGame() {
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    Assert.assertEquals(driver.start(ponOverChi().round(ponOverChiResult()), ws), Boundary.DECISIONS);
    Decisions decisions = driver.engine().decisions();
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.TURN);
    Assert.assertEquals(chosen(driver, 0), "DAHAI(3p)");

    Assert.assertEquals(driver.step(ws), Boundary.DECISIONS);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.RESPONSE);
    Assert.assertEquals(decisions.size(), 2);
    Assert.assertEquals(driver.choice(decisions.slotOf(0)), ReplayDriver.UNOBSERVED);
    Assert.assertEquals(chosen(driver, decisions.slotOf(1)), "PON(3p)");

    Assert.assertEquals(driver.step(ws), Boundary.DECISIONS);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.POST_CALL_DAHAI);
    Assert.assertEquals(chosen(driver, 0), "DAHAI(6s)");

    Assert.assertEquals(driver.step(ws), Boundary.DECISIONS);
    Assert.assertEquals(decisions.seat(0), 2);
    Assert.assertEquals(chosen(driver, 0), "RON_AGARI");
    Assert.assertEquals(driver.step(ws), Boundary.GAME_END);

    driver.replay(game("pon-over-chi", ponOverChi().round(ponOverChiResult()), 24_000, 21_000, 31_000, 24_000), ws);
  }

  /** 手に無い牌の打牌は合法でない手になり、再生は例外を投げる。 */
  @Test
  public void aDiscardOfAMissingTileIsIllegal() {
    assertRejected(illegalDiscard(), "is not legal");
  }

  /** 記録の点の移動が精算と違えば、局末の不一致になる。 */
  @Test
  public void aWrongDeltaIsARoundEndMismatch() {
    assertRejected(wrongDelta(), "The engine ended with");
  }

  /**
   * 局を終える打牌（ここでは四風連打の 4 枚目の東）に誰もロンできなければ、エンジンは応答を求めずに局を終える。記録がその打牌へのロンなら、
   * 局末の不一致ではなく合法でない手になる。
   */
  @Test
  public void aRonOnADiscardThatEndsTheRoundIsIllegal() {
    assertRejected(
        game("four-winds-ron", fourWinds().round(furitenRon()), 24_000, 30_300, 21_700, 24_000),
        "which no seat can claim");
  }

  /**
   * 同巡振聴のまま上家・対面からポンした席は振聴が解けてロンでき、記録どおりに再生できる。下家からポンした席のロンはエンジンが出さない
   * ので、合法でない手になる。
   */
  @Test
  public void onlyACallFromTheKamichaOrTheToimenClearsTheSameTurnFuriten() {
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    driver.replay(game("toimen-furiten", toimenPon().round(furitenRon()), 24_000, 30_300, 21_700, 24_000), ws);
    driver.replay(game("kamicha-furiten", kamichaPon().round(furitenRon()), 24_000, 30_300, 21_700, 24_000), ws);
    RecordedGame shimocha =
        game("shimocha-furiten", shimochaPon().round(furitenRon()), 24_000, 30_300, 21_700, 24_000);
    assertRejected(shimocha, "which no seat can claim");
  }

  /**
   * 東 1 局: 席 0 が 3m を 2 枚持ち、天鳳の牌譜のように番号の大きい写しを手出しする。エンジンも記録の物理牌を打つので、手に残るのは
   * 番号の小さい写しになる。手に無い写しを指定して進めようとすると、エンジンは受け付けない。
   */
  @Test
  public void theEngineDiscardsTheCopyTheRecordDiscards() {
    RoundBuilder builder =
        new RoundBuilder(RoundStart.first(VARIANT))
            .hand(0, "33m147m258p369s12z")
            .hand(1, "19m147p258s369s34z")
            .hand(2, "28m369p147s258s36z")
            .hand(3, "19m258p369s147s57z")
            .indicator("9p")
            .draw(0, "6m")
            .discard(0, "3m")
            .draw(1, "9m")
            .discard(1, "9m");
    RoundResult draw = RoundResult.drawWithoutDetails(new int[4]);
    int kept = builder.round(draw).deal(0, 0);
    int discarded = builder.round(draw).deal(0, 1);
    Assert.assertEquals(Tiles.kind(discarded), Tiles.kind(kept));
    Assert.assertTrue(discarded > kept);
    List<RecordEvent> events = builder.events();
    events.set(1, new RecordEvent.Discard(0, discarded, false, false));
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    Assert.assertEquals(driver.start(builder.round(events, draw), ws), Boundary.DECISIONS);
    Assert.assertEquals(chosen(driver, 0), "DAHAI(3m)");
    int missing = Tiles.physicalId(Tiles.kind(kept), 2);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> driver.engine().commit(new int[] {driver.choice(0)}, missing, ws));

    Assert.assertEquals(driver.step(ws), Boundary.DECISIONS);
    Assert.assertEquals(driver.engine().decisions().seat(0), 1);
    Assert.assertTrue(driver.engine().view(0).holds(kept), "the copy the record keeps");
    Assert.assertFalse(driver.engine().view(0).holds(discarded), "the copy the record discards");
  }

  /** 三人打ち: 席 0 が北を 2 枚持ち、南をツモって番号の大きい北を抜く。エンジンも記録の北を抜くので、手に残るのは番号の小さい北になる。 */
  @Test
  public void theEngineTakesOutTheNorthTheRecordTakesOut() {
    RoundBuilder builder =
        new RoundBuilder(SANMA_EAST_ONE)
            .hand(0, "19m258p369p147s44z")
            .hand(1, "19m147p258s369s57z")
            .hand(2, "19m369p147s258s36z")
            .indicator("9s")
            .draw(0, "2z")
            .nuki(0)
            .draw(0, "6z")
            .discard(0, "6z");
    RoundResult draw = RoundResult.drawWithoutDetails(new int[3]);
    int kept = builder.round(draw).deal(0, 11);
    int taken = builder.round(draw).deal(0, 12);
    Assert.assertEquals(List.of(Tiles.kind(kept), Tiles.kind(taken)), List.of(Tiles.PEI, Tiles.PEI));
    Assert.assertTrue(taken > kept);
    List<RecordEvent> events = builder.events();
    events.set(1, new RecordEvent.Nuki(0, taken));
    ReplayDriver driver = new ReplayDriver(Variant.TENHOU_3P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_3P);
    Assert.assertEquals(driver.start(builder.round(events, draw), ws), Boundary.DECISIONS);
    Assert.assertEquals(chosen(driver, 0), "NUKI");

    Assert.assertEquals(driver.step(ws), Boundary.DECISIONS);
    Assert.assertEquals(driver.engine().decisions().kind(0), EngineDecisionKind.TURN);
    Assert.assertTrue(driver.engine().view(0).holds(kept), "the north the record keeps");
    Assert.assertFalse(driver.engine().view(0).holds(taken), "the north the record takes out");
  }

  /**
   * 三人打ち: 席 1（白の暗刻、1p と北の双碰）が、席 0 の抜いた北へのロンを見送って振聴になる。記録では、席 0 が補充の後に打った 1p に
   * 席 1 がロンする。エンジンはこのロンを出さないので合法でない手になる。
   */
  @Test
  public void aRonAfterPassingANukiIsIllegal() {
    RoundBuilder builder =
        new RoundBuilder(SANMA_EAST_ONE)
            .hand(0, "19m258s369s1267z1p")
            .hand(1, "555z456s789s11p44z")
            .hand(2, "19m147p258p369p34z")
            .indicator("9s")
            .draw(0, "4z")
            .nuki(0)
            .draw(0, "6p")
            .discard(0, "1p");
    RoundResult.Win win =
        new RoundResult.Win(1, 320, ScoringYaku.YAKUHAI_HAKU.bit(), 1, 40, 0, 0, 0, RoundOutcome.NO_SEAT);
    RoundResult ron =
        new RoundResult(EndKind.RON, null, List.of(win), 0, 0, 0, new int[] {-1_300, 1_300, 0}, new int[0]);
    assertRejected(game("nuki-furiten", builder.round(ron), 33_700, 36_300, 35_000), "Seat 1 cannot ron");
  }

  /**
   * 三人打ち: 北の単騎で立直した席 0 が、西をツモって手の中の北を抜く。エンジンは立直中にはツモった北しか抜かせないので合法でない手になる。
   */
  @Test
  public void aRiichiNukiOfAHeldNorthIsIllegal() {
    RoundBuilder builder = riichiOnNorth().draw(0, "3z").nuki(0);
    RoundResult draw =
        new RoundResult(EndKind.EXHAUSTIVE, null, List.of(), RoundOutcome.NO_SEAT, 0, 0, new int[3], new int[0]);
    assertRejected(game("riichi-nuki", builder.round(draw), 34_000, 35_000, 35_000), "is not legal");
  }

  /**
   * 三人打ち: 北の単騎で立直した席 0 が、和了牌の北をツモって抜き、以後ずっと振聴になる。記録では、席 1 が抜いた北に席 0 がロンする。
   * エンジンは誰もロンできないとみて応答を求めないので合法でない手になる。
   */
  @Test
  public void aRonAfterARiichiNukiOfTheWinningNorthIsIllegal() {
    RoundBuilder builder =
        riichiOnNorth().draw(0, "4z").nuki(0).draw(0, "6z").discard(0, "6z").draw(1, "4z").nuki(1);
    RoundResult.Win win =
        new RoundResult.Win(0, 2_000, ScoringYaku.RIICHI.bit(), 5, 40, 0, 0, 0, RoundOutcome.NO_SEAT);
    RoundResult ron =
        new RoundResult(EndKind.RON, null, List.of(win), 1, 0, 0, new int[] {13_000, -12_000, 0}, new int[0]);
    assertRejected(game("riichi-nuki-miss", builder.round(ron), 48_000, 23_000, 35_000), "which no seat can claim");
  }

  /**
   * 三人打ちの東 1 局（親は席 0）: 席 0（123p456p789s111s、北の単騎）が第一ツモの南をツモ切りして立直し、席 1 と席 2 もツモ切りする。
   */
  private static RoundBuilder riichiOnNorth() {
    return new RoundBuilder(SANMA_EAST_ONE)
        .hand(0, "123p456p789s111s4z")
        .hand(1, "19m258p369p147s56z")
        .hand(2, "19m147p258s369s17z")
        .indicator("9p")
        .draw(0, "2z")
        .riichi(0, "2z")
        .draw(1, "8p")
        .discard(1, "8p")
        .draw(2, "8s")
        .discard(2, "8s");
  }

  /**
   * 半荘の再生が IllegalStateException を投げ、そのメッセージが食い違いの種類を表す文字列を含むことを確かめる。
   *
   * @param reason 合法でない手（"is not legal" など）か局末の不一致（"The engine ended with" など）を表す文字列
   */
  private static void assertRejected(RecordedGame game, String reason) {
    Variant variant = game.variant();
    IllegalStateException failure =
        Assert.expectThrows(
            IllegalStateException.class, () -> new ReplayDriver(variant).replay(game, new RuleWorkspace(variant)));
    Assert.assertTrue(failure.getMessage().contains(reason), failure.getMessage());
  }

  /**
   * 南 4 局（親は席 3）: 席 3 の 3p を、下家の席 0 が 24p でチーでき、対面の席 1 が 33p でポンする。ポンの後の席 1 の 6s に、席 2
   * （234m567m345p66s78s、断么九・平和）がロンする。
   */
  static RoundBuilder ponOverChi() {
    return new RoundBuilder(new RoundStart(VARIANT, 7, 0, 0, new int[] {24_000, 23_000, 29_000, 24_000}))
        .hand(3, "19m19p19s11223z3p7z")
        .hand(0, "24p28m28s4455677z")
        .hand(1, "33p3377m33677s66z")
        .hand(2, "234567m345p6678s")
        .indicator("9s")
        .draw(3, "8p")
        .discard(3, "3p")
        .pon(1)
        .discard(1, "6s");
  }

  /** 2 翻 30 符（基本点 480）の子のロン 2000 点。 */
  static RoundResult ponOverChiResult() {
    RoundResult.Win win =
        new RoundResult.Win(
            2, 480, ScoringYaku.TANYAO.bit() | ScoringYaku.PINFU.bit(), 2, 30, 0, 0, 0, RoundOutcome.NO_SEAT);
    return new RoundResult(EndKind.RON, null, List.of(win), 1, 0, 0, new int[] {0, -2_000, 2_000, 0}, new int[0]);
  }

  /**
   * 南 4 局（親は席 3）: 席 1（中の暗刻、5s・8s 待ち）が席 2 の 5s のロンを見送って同巡振聴になり、対面の席 3 の 5p をポンして 6s を
   * 打つ。続く席 2 の 7s に席 1 がロンする（中のみ 40 符）。
   */
  static RoundBuilder toimenPon() {
    return new RoundBuilder(SOUTH_FOUR)
        .hand(3, "1478m369p149s123z")
        .hand(0, "258m19p258s34567z")
        .hand(1, "777z234m678m55p67s")
        .hand(2, "39m147p28p19s1234z")
        .indicator("9s")
        .draw(3, "9m")
        .discard(3, "9m")
        .draw(0, "9p")
        .discard(0, "9p")
        .draw(1, "1z")
        .discard(1, "1z")
        .draw(2, "5s")
        .discard(2, "5s")
        .draw(3, "5p")
        .discard(3, "5p")
        .pon(1)
        .discard(1, "6s")
        .draw(2, "7s")
        .discard(2, "7s");
  }

  /**
   * 南 4 局（親は席 3）: 席 1（中の暗刻、5s・8s 待ち）が席 2 の 5s のロンを見送って同巡振聴になり、席 3 の 1m も見送って、上家の席 0
   * の 9p をポンして 1m を打つ。続く席 2 の 8s に席 1 がロンする（中のみ 40 符）。
   */
  static RoundBuilder kamichaPon() {
    return new RoundBuilder(SOUTH_FOUR)
        .hand(3, "258m147p369s1234z")
        .hand(0, "369m258p147s1567z")
        .hand(1, "111m777z234m99p67s")
        .hand(2, "479m136p258s2345z")
        .indicator("9s")
        .draw(3, "8p")
        .discard(3, "8p")
        .draw(0, "1s")
        .discard(0, "1s")
        .draw(1, "6p")
        .discard(1, "6p")
        .draw(2, "5s")
        .discard(2, "5s")
        .draw(3, "1m")
        .discard(3, "1m")
        .draw(0, "9p")
        .discard(0, "9p")
        .pon(1)
        .discard(1, "1m")
        .draw(2, "8s")
        .discard(2, "8s");
  }

  /**
   * 南 4 局（親は席 3）: 席 1（中の暗刻、5s・8s 待ち）が席 0 の 5s へのロンを見送って同巡振聴になる（その 5s は席 2 がポンする）。
   * 続いて下家の席 2 が打った 5p をポンして 6s を打つ。記録では、次の席 2 の 7s に席 1 がロンする。
   */
  private static RoundBuilder shimochaPon() {
    return new RoundBuilder(SOUTH_FOUR)
        .hand(3, "1478m369p149s123z")
        .hand(0, "258m19p147s34567z")
        .hand(1, "777z234m678m55p67s")
        .hand(2, "55s5p147m28p369s14z")
        .indicator("9s")
        .draw(3, "9m")
        .discard(3, "9m")
        .draw(0, "5s")
        .discard(0, "5s")
        .pon(2)
        .discard(2, "5p")
        .pon(1)
        .discard(1, "6s")
        .draw(2, "7s")
        .discard(2, "7s");
  }

  /** 南 4 局（親は席 3）: 第一巡に全員が手の東を打つ。どの手も聴牌でなく、東を 2 枚持つ席も無い。 */
  private static RoundBuilder fourWinds() {
    return new RoundBuilder(SOUTH_FOUR)
        .hand(3, "147m258p369s1256z")
        .hand(0, "258m369p147s1347z")
        .hand(1, "369m147p258s1567z")
        .hand(2, "147m258p369s1234z")
        .indicator("9s")
        .draw(3, "9m")
        .discard(3, "1z")
        .draw(0, "9p")
        .discard(0, "1z")
        .draw(1, "1s")
        .discard(1, "1z")
        .draw(2, "8m")
        .discard(2, "1z");
  }

  /** 席 2 の打牌への席 1 のロン（中のみ 40 符、基本点 320 の子のロン 1300 点）。 */
  static RoundResult furitenRon() {
    RoundResult.Win win =
        new RoundResult.Win(1, 320, ScoringYaku.YAKUHAI_CHUN.bit(), 1, 40, 0, 0, 0, RoundOutcome.NO_SEAT);
    return new RoundResult(EndKind.RON, null, List.of(win), 2, 0, 0, new int[] {0, 1_300, -1_300, 0}, new int[0]);
  }

  /** チーよりポンが優先される場面で、席 3 の打牌を手に無い 5s に替えた記録。 */
  private static RecordedGame illegalDiscard() {
    RoundBuilder builder = ponOverChi();
    List<RecordEvent> events = builder.events();
    events.set(1, new RecordEvent.Discard(3, Tiles.physicalId(Tiles.S5, 1), false, false));
    return game("illegal-discard", builder.round(events, ponOverChiResult()), 24_000, 21_000, 31_000, 24_000);
  }

  /** チーよりポンが優先される場面で、点の移動を 1900 点に替えた記録。 */
  private static RecordedGame wrongDelta() {
    RoundResult right = ponOverChiResult();
    RoundResult wrong =
        new RoundResult(
            right.kind(),
            right.abortive(),
            right.wins(),
            right.loser(),
            right.tenpaiMask(),
            right.nagashiMask(),
            new int[] {0, -1_900, 1_900, 0},
            right.uraIndicators());
    return game("wrong-delta", ponOverChi().round(wrong), 24_000, 21_000, 31_000, 24_000);
  }

  /** 1 局で終わる半荘の記録。 */
  static RecordedGame game(String id, RecordedRound round, int... finalScores) {
    Variant variant = round.start().variant();
    return new RecordedGame(variant, List.of(round), finalScores, RecordedGames.unnamed(id, variant));
  }

  /**
   * 記録で選んだ手の名前（{@code Action.toString} の文字列）。
   *
   * @param slot 判断する席のうち何番目か
   */
  private static String chosen(ReplayDriver driver, int slot) {
    Decisions decisions = driver.engine().decisions();
    return Action.of(decisions.legal(slot, driver.choice(slot))).toString();
  }
}
