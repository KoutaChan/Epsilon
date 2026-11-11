package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 打牌と加槓への他家の応答のテスト。応答する席と合法手の並び、判断 ID、優先順位と実際に行う手の決め方、立直の供託、見逃しと振聴、槍槓を確かめる。 */
public class WindowTest {

  /** 1s・4s 待ち、一気通貫。 */
  private static final String WAITS_1S_4S = "123m456m789m23s11z";

  /** 5s 待ちの 3 つの手牌（順に一気通貫・一気通貫・断么九）。 */
  private static final String MAN_ITTSU = "123m456m789m46s11z";

  private static final String PIN_ITTSU = "123p456p789p46s22z";
  private static final String TANYAO = "234m345p678s456m5s";

  /** 席 0 が打った 5p に、席 1 はチーが 3 通り、席 2 は赤入りのポン・ポン・大明槓、席 3 はロン（一気通貫）ができる手牌。 */
  private static Scene fivePinHands(Scene scene) {
    return scene.hand(1, "3467p123m789s11z9m").hand(2, "055p123s456s789s2z").hand(3, "123m456m789m46p11z");
  }

  private static Scene fivePin() {
    return fivePinHands(Scene.first()).river(0, "5p").respond();
  }

  @Test
  public void responsesFollowTheSeatOrderAndTheContract() {
    Scene scene = fivePin();
    Assert.assertEquals(scene.decisions.size(), 3);
    for (int slot = 0; slot < 3; slot++) {
      Assert.assertEquals(scene.decisions.seat(slot), slot + 1);
      Assert.assertEquals(scene.decisions.kind(slot), EngineDecisionKind.RESPONSE);
      Assert.assertEquals(scene.decisions.id(slot), slot + 1L);
    }
    Assert.assertEquals(scene.nextId(), 4L);
    Assert.assertEquals(scene.legal(1), List.of("CHI(3p4p5p 5p)", "CHI(4p5p6p 5p)", "CHI(5p6p7p 5p)", "PASS"));
    Assert.assertEquals(scene.legal(2), List.of("PON_AKA(5pr)", "PON(5p)", "DAIMINKAN(5p)", "PASS"));
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PASS"));
    Assert.assertTrue(scene.decisions.win(2).has(ScoringYaku.ITTSU));
  }

  /** 1 席が全ての種類の応答を持つと、ロン → ポン（赤が先）→ 大明槓 → チー → 見送りの順に並ぶ。 */
  @Test
  public void oneSeatListsEveryResponseInOrder() {
    Scene scene = Scene.first().hand(1, "05567p234m456m88s").river(0, "5p").respond();
    Assert.assertEquals(
        scene.legal(1),
        List.of("RON_AGARI", "PON_AKA(5pr)", "PON(5p)", "DAIMINKAN(5p)", "CHI(5p6p7p 5p)", "PASS"));
    Assert.assertTrue(scene.decisions.win(0).has(ScoringYaku.TANYAO));
  }

  /** 選べる手の無い席は判断を持たず、判断 ID も使わない。見送りは、ほかに選べる手があるときだけ並ぶ。 */
  @Test
  public void onlySeatsWithAChoiceGetASlotAndAnId() {
    Scene scene =
        Scene.first()
            .hand(1, "13579m13579p135s")
            .hand(2, "77z123m456p789s19p")
            .hand(3, "2468m2468p2468s1z")
            .river(0, "7z")
            .respond();
    Assert.assertEquals(scene.decisions.size(), 1);
    Assert.assertEquals(scene.decisions.seat(0), 2);
    Assert.assertEquals(scene.decisions.id(0), 1L);
    Assert.assertEquals(scene.nextId(), 2L);
    Assert.assertEquals(scene.legal(2), List.of("PON(C)", "PASS"));
    Assert.assertEquals(scene.legal(1), List.of());
    Assert.assertEquals(scene.close("PASS"), Window.Outcome.PASS);
  }

  /** 最後の打牌には鳴けず、ロンすれば河底が付く。 */
  @Test
  public void theLastDiscardCanOnlyBeRonnedForHoutei() {
    Scene scene = fivePinHands(Scene.first().burn(Scene.VARIANT.liveWall())).river(0, "5p").respond();
    Assert.assertEquals(scene.decisions.size(), 1);
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PASS"));
    Assert.assertTrue(scene.decisions.win(0).has(ScoringYaku.HOUTEI));
  }

  /** 立直中の席は鳴けない。チーできるのは上家の打牌だけ。 */
  @Test
  public void callsNeedNoRiichiAndChiNeedsTheKamicha() {
    Scene riichi = fivePinHands(Scene.first()).riichi(1).riichi(2).river(0, "5p").respond();
    Assert.assertEquals(riichi.legal(1), List.of());
    Assert.assertEquals(riichi.legal(2), List.of());
    Assert.assertEquals(riichi.legal(3), List.of("RON_AGARI", "PASS"));
    Scene toimen = Scene.first().hand(2, "3467p123m789s11z9m").river(0, "5p").respond();
    Assert.assertEquals(toimen.decisions.size(), 0);
  }

  /** 鳴いた後に喰い替えになる牌しか残らないチーは、合法手に出さない。 */
  @Test
  public void aChiThatLeavesNoDiscardIsNotOffered() {
    Scene banned = threePons().hand(1, "3455m").river(0, "2m").respond();
    Assert.assertEquals(banned.decisions.size(), 0);
    Scene allowed = threePons().hand(1, "3456m").river(0, "2m").respond();
    Assert.assertEquals(allowed.legal(1), List.of("CHI(2m3m4m 2m)", "PASS"));
  }

  /** 優先順位はロン、ポン・大明槓、チーの順。席ごとに、選んだ手が行われたかと、選び替えで結果が変わりうるかを確かめる。 */
  @Test
  public void ronBeatsCallsAndPonBeatsChi() {
    Scene ron = fivePin();
    Assert.assertEquals(ron.close("CHI(4p5p6p 5p)", "PON_AKA(5pr)", "RON_AGARI"), Window.Outcome.RON);
    Assert.assertEquals(ron.window.winnerCount(), 1);
    Assert.assertEquals(ron.window.winner(0), 3);
    assertResolution(ron, "PREEMPTED false", "PREEMPTED false", "CAUSAL true");
    Assert.assertThrows(IllegalStateException.class, ron.window::caller);

    Scene pon = fivePin();
    Assert.assertEquals(pon.close("CHI(4p5p6p 5p)", "PON(5p)", "PASS"), Window.Outcome.CALL);
    Assert.assertEquals(pon.window.caller(), 2);
    Assert.assertEquals(pon.window.call(), Action.code(Type.PON, Tiles.P5, TileSelection.NON_AKA));
    assertResolution(pon, "PREEMPTED false", "CAUSAL true", "CAUSAL true");

    Scene chi = fivePin();
    Assert.assertEquals(chi.close("CHI(4p5p6p 5p)", "PASS", "PASS"), Window.Outcome.CALL);
    Assert.assertEquals(chi.window.caller(), 1);
    Assert.assertEquals(chi.window.call(), Action.chi(Tiles.P4, Tiles.P5, false));
    assertResolution(chi, "CAUSAL true", "CAUSAL true", "CAUSAL true");

    Scene pass = fivePin();
    Assert.assertEquals(pass.close("PASS", "PASS", "PASS"), Window.Outcome.PASS);
    assertResolution(pass, "CAUSAL true", "CAUSAL true", "CAUSAL true");
    Assert.assertEquals(pass.window.winnerCount(), 0);
  }

  /** ロンを選んだ席は全員が和了し、放銃した席から手番の近い順に並ぶ。四人打ちで 3 人がロンすれば三家和になる。 */
  @Test
  public void everyRonWinsInTurnOrderAndThreeAbort() {
    Scene triple = Scene.first().hand(1, MAN_ITTSU).hand(2, PIN_ITTSU).hand(3, TANYAO).river(0, "5s").respond();
    Assert.assertEquals(triple.legal(1), List.of("RON_AGARI", "CHI(4s5s6s 5s)", "PASS"));
    Assert.assertEquals(triple.close("RON_AGARI", "RON_AGARI", "RON_AGARI"), Window.Outcome.TRIPLE_RON);
    Assert.assertEquals(triple.window.winnerCount(), 3);

    Scene twice = Scene.first().hand(1, MAN_ITTSU).hand(2, PIN_ITTSU).hand(3, TANYAO).river(0, "5s").respond();
    Assert.assertEquals(twice.close("RON_AGARI", "RON_AGARI", "PASS"), Window.Outcome.RON);
    Assert.assertEquals(twice.window.winnerCount(), 2);
    Assert.assertEquals(twice.window.winner(0), 1);
    Assert.assertEquals(twice.window.winner(1), 2);

    Scene order = Scene.first().hand(3, MAN_ITTSU).hand(0, PIN_ITTSU).hand(1, TANYAO).river(2, "5s").respond();
    Assert.assertEquals(order.close("PASS", "RON_AGARI", "RON_AGARI"), Window.Outcome.RON);
    Assert.assertEquals(order.window.winner(0), 3);
    Assert.assertEquals(order.window.winner(1), 1);
  }

  /** 立直の供託は、宣言牌が通った後に払う。鳴きで途切れていない第一打ならダブル立直になる。 */
  @Test
  public void aRiichiPaysItsDepositWhenNoRonFollows() {
    Scene first = Scene.first().riichiDiscard(0, "9m").respond();
    Assert.assertEquals(first.state.riichi(0), RiichiState.NONE);
    Assert.assertEquals(first.close(), Window.Outcome.PASS);
    Assert.assertEquals(first.state.riichi(0), RiichiState.DOUBLE_RIICHI);
    Assert.assertEquals(first.state.kyotaku(), 1);
    Assert.assertEquals(first.state.score(0), 24_000);
    Assert.assertTrue(first.state.ippatsu(0));

    Scene later = Scene.first().river(0, "1m").riichiDiscard(0, "9m").respond();
    later.close();
    Assert.assertEquals(later.state.riichi(0), RiichiState.RIICHI);

    Scene interrupted = Scene.first().pon(2, 1, "7z").riichiDiscard(0, "9m").respond();
    interrupted.close();
    Assert.assertEquals(interrupted.state.riichi(0), RiichiState.RIICHI);

    Scene called = Scene.first().hand(1, "99m123p456p789s11z").riichiDiscard(0, "9m").respond();
    Assert.assertEquals(called.legal(1), List.of("PON(9m)", "PASS"));
    Assert.assertEquals(called.close("PON(9m)"), Window.Outcome.CALL);
    Assert.assertEquals(called.state.riichi(0), RiichiState.DOUBLE_RIICHI);
  }

  @Test
  public void aRonOnTheRiichiTileCancelsTheRiichi() {
    Scene scene = Scene.first().hand(3, WAITS_1S_4S).riichiDiscard(0, "1s").respond();
    Assert.assertEquals(scene.close("RON_AGARI"), Window.Outcome.RON);
    Assert.assertEquals(scene.state.riichi(0), RiichiState.NONE);
    Assert.assertEquals(scene.state.kyotaku(), 0);
    Assert.assertEquals(scene.state.score(0), 25_000);
  }

  /** 待ちのどれかが自分の河にあれば、どの待ちでもロンできない。 */
  @Test
  public void aWaitInTheOwnRiverBarsEveryRon() {
    Scene scene = Scene.first().hand(3, WAITS_1S_4S).river(3, "4s").river(0, "1s").respond();
    Assert.assertEquals(scene.decisions.size(), 0);
    Assert.assertEquals(scene.close(), Window.Outcome.PASS);
    Assert.assertTrue(scene.hands[3].missedRon());
  }

  /** 見逃しの印が立つと、消えるまでどの待ちにもロンが出ない。 */
  @Test
  public void aPassedRonBarsEveryWaitUntilTheMarkClears() {
    Scene scene = Scene.first().hand(3, WAITS_1S_4S).river(0, "1s").respond();
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PASS"));
    scene.close("PASS");
    Assert.assertTrue(scene.hands[3].missedRon());
    scene.river(1, "4s").respond();
    Assert.assertEquals(scene.decisions.size(), 0);
    scene.close();
    scene.hands[3].setMissedRon(false);
    scene.river(0, "4s").respond();
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PASS"));
  }

  /** 立直中に見逃しても同じ印が立ち、後の打牌ではロンできない。 */
  @Test
  public void aPassedRonInRiichiSetsTheSameMark() {
    Scene scene = Scene.first().hand(3, WAITS_1S_4S).riichi(3).river(0, "1s").respond();
    scene.close("PASS");
    Assert.assertTrue(scene.hands[3].missedRon());
    scene.river(1, "4s").respond();
    Assert.assertEquals(scene.decisions.size(), 0);
  }

  /** 役が無くてロンできない和了牌でも、見送れば同巡振聴になる。 */
  @Test
  public void aWinningTileWithoutYakuStillCountsAsPassed() {
    Scene scene = Scene.first().pon(3, 1, "2m").hand(3, "456m789p23s55p").river(0, "4s").respond();
    Assert.assertEquals(scene.decisions.size(), 0);
    scene.close();
    Assert.assertTrue(scene.hands[3].missedRon());
  }

  /** 加槓への応答は、加槓した席の次の席から手番の順に求め、ロンは槍槓になる。見送れば見逃しの印が付く。 */
  @Test
  public void anAddedKanOpensAChankanWindowInTurnOrder() {
    Scene scene =
        Scene.first()
            .pon(1, 3, "4s")
            .hand(1, "4s123m456m789p")
            .hand(2, "123p456p789p35s11z")
            .hand(0, "234m567m678p35s88s")
            .draw(1, "5z")
            .kakan(1, "4s")
            .respond();
    Assert.assertEquals(scene.decisions.size(), 2);
    Assert.assertEquals(scene.decisions.seat(0), 2);
    Assert.assertEquals(scene.decisions.seat(1), 0);
    for (int slot = 0; slot < 2; slot++) {
      Assert.assertEquals(scene.decisions.kind(slot), EngineDecisionKind.CHANKAN);
      Assert.assertEquals(scene.legal(scene.decisions.seat(slot)), List.of("RON_AGARI", "PASS"));
    }
    Assert.assertTrue(scene.decisions.win(0).has(ScoringYaku.CHANKAN));
    Assert.assertTrue(scene.decisions.win(1).has(ScoringYaku.TANYAO));
    Assert.assertEquals(scene.close("PASS", "PASS"), Window.Outcome.PASS);
    Assert.assertTrue(scene.hands[2].missedRon());
    Assert.assertTrue(scene.hands[0].missedRon());
  }

  /** 国士は加槓を槍槓できるが、暗槓には応答を求めない。 */
  @Test
  public void kokushiRobsAnAddedKanButAClosedKanOpensNoWindow() {
    Scene added =
        Scene.first()
            .hand(3, "19m19p19s2234567z")
            .pon(1, 0, "1z")
            .hand(1, "1z123m456p789s")
            .draw(1, "8s")
            .kakan(1, "1z")
            .respond();
    Assert.assertEquals(added.legal(3), List.of("RON_AGARI", "PASS"));
    Assert.assertTrue(added.decisions.win(0).has(ScoringYaku.KOKUSHI));
    Assert.assertEquals(added.decisions.win(0).yakuman(), 1);

    Scene closed = Scene.first().hand(3, "19m19p19s2234567z").hand(1, "123m456p789s").ankan(1, "1z");
    Assert.assertThrows(IllegalStateException.class, closed::respond);
  }

  /** 判断に付くロンの点は、立直なら裏ドラを含む。{@code FullView.ronScore} は同じ判定をするが、裏ドラは数えない。 */
  @Test
  public void aRiichiRonKeepsItsUraInTheSlot() {
    Scene scene =
        Scene.first().hand(3, WAITS_1S_4S).riichi(3).indicator("1m").ura(0, "1s").river(0, "1s").respond();
    WinScore slot = scene.decisions.win(scene.decisions.slotOf(3));
    Assert.assertEquals(slot.dora(), 1);
    Assert.assertEquals(slot.ura(), 1);
    WinScore without = new WinScore();
    Assert.assertTrue(scene.full.ronScore(3, Tiles.physicalId(Tiles.S1, 3), 0, scene.ws, without));
    Assert.assertEquals(without.dora(), 1);
    Assert.assertEquals(without.ura(), 0);
    Assert.assertEquals(without.han(), slot.han() - 1);
  }

  @Test
  public void aCopyEqualsTheOriginal() {
    Scene scene = fivePin();
    scene.close("CHI(4p5p6p 5p)", "PON(5p)", "PASS");
    Window window = new Window(Scene.VARIANT);
    window.copyFrom(scene.window);
    Fields.assertSameFields(scene.window, window);
    Decisions decisions = new Decisions(Scene.VARIANT);
    decisions.copyFrom(scene.decisions);
    Fields.assertSameFields(scene.decisions, decisions);
  }

  /** 席 1 が席 0・2・3 からポンした面子 3 つ（1p・9p・9s、役は無い）。 */
  private static Scene threePons() {
    return Scene.first().pon(1, 0, "1p").pon(1, 2, "9p").pon(1, 3, "9s");
  }

  /** 判断の順に、選び替えで結果が変わりうるか（{@code Resolution.Role} の名前）と手が行われたかを空白で区切って並べ、期待と照合する。 */
  private static void assertResolution(Scene scene, String... expected) {
    Resolution resolution = scene.resolution;
    Assert.assertEquals(resolution.size(), expected.length);
    for (int slot = 0; slot < expected.length; slot++) {
      Assert.assertEquals(resolution.id(slot), scene.decisions.id(slot));
      Assert.assertEquals(resolution.role(slot) + " " + resolution.executed(slot), expected[slot], "slot " + slot);
    }
  }
}
