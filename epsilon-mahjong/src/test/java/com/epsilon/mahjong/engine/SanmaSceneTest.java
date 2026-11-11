package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 三人打ちの場面のテスト。北抜きの流れと、三人打ちのルール（チー・北の鳴き・途中流局・不聴罰符・流し満貫・ダブロン・終局）を確かめる。 */
public class SanmaSceneTest {

  private static final Variant VARIANT = Variant.TENHOU_3P;

  /** 1z・2z の双碰の聴牌。 */
  private static final String TENPAI = "123p456p789p1122z";

  /** 3z・5z の双碰の聴牌。 */
  private static final String TENPAI_SOU = "123s456s789s3355z";

  /** どの 2 枚も塔子や対子にならない不聴の手（北を持たない。3 通り）。 */
  private static final String NOTEN_A = "19m258p147s369s67z";

  private static final String NOTEN_B = "19m369p258s147p57z";
  private static final String NOTEN_C = "19m369p258s369s57z";

  /**
   * 親の第一ツモの手番。合法手は暗槓 → 北抜き → 立直の打牌 → 打牌の順に並び、北のツモ切りも選べる（北は抜かなくてもよい）。
   * 抜いた北は河に入らず、新しいドラ表示牌もめくらない。補充は王牌から引き、山が 1 枚減る。補充の手番でも続けて抜ける。北抜きで第一巡が
   * 途切れるので、続く第一打の立直はダブル立直にならない。
   */
  @Test
  public void theDealerTakesOutNorthsBeforeTheFirstDiscard() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1111p23p456s78s4z5z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("4z")
            .rinshan("9s1z")
            .start();
    Assert.assertEquals(
        scene.legal(0),
        List.of(
            "ANKAN(1p)", "NUKI", "RIICHI_DAHAI(P)", "DAHAI(1p)", "DAHAI(2p)", "DAHAI(3p)", "DAHAI(4s)", "DAHAI(5s)",
            "DAHAI(6s)", "DAHAI(7s)", "DAHAI(8s)", "DAHAI(N)", "DAHAI_TSUMOGIRI(N)", "DAHAI(P)"));
    scene.play("0:NUKI");
    Assert.assertTrue(scene.turnOf(0));
    SeatView view = scene.view(1);
    Assert.assertTrue(scene.view(0).last().rinshan());
    Assert.assertEquals(view.nuki(0), 1);
    Assert.assertEquals(view.visible(Tiles.PEI), 1);
    Assert.assertEquals(view.events().kind(0), EventKind.NUKI);
    Assert.assertEquals(view.events().riverSize(0), 0);
    Assert.assertEquals(view.indicatorCount(), 1);
    Assert.assertEquals(scene.liveLeft(), VARIANT.liveWall() - 2);
    Assert.assertTrue(scene.legal(0).contains("NUKI"));

    scene.play("0:NUKI");
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertEquals(view.nuki(0), 2);
    Assert.assertEquals(view.visible(Tiles.PEI), 2);
    Assert.assertEquals(view.indicatorCount(), 1);
    Assert.assertEquals(scene.liveLeft(), VARIANT.liveWall() - 3);
    Assert.assertFalse(scene.legal(0).contains("NUKI"));
    ChanceLog chance = scene.engine.full().chance();
    Assert.assertEquals(chance.size(), 3);
    Assert.assertEquals(chance.kind(0), ChanceKind.DRAW);
    Assert.assertEquals(chance.kind(1), ChanceKind.RINSHAN);
    Assert.assertEquals(chance.kind(2), ChanceKind.RINSHAN);

    scene.play("0:RIICHI_DAHAI(P)");
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertEquals(view.riichi(0), RiichiState.RIICHI);
  }

  /**
   * 山の残りが 1 枚のときの北抜き。補充で山が 0 枚になる。補充のツモで和了すれば嶺上開花で、海底は付かない。続く打牌は最後の打牌なので、
   * そのロンは河底で、鳴くことはできない。
   */
  @Test
  public void aNukiOnTheLastLiveTileLeadsToRinshanAndThenHoutei() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "234p567p234s678s1s")
            .hand(1, "11s19m258p369p567z")
            .hand(2, "123p456p789s99s1s4z")
            .drawAt(53, "2s")
            .rinshan("3s")
            .start()
            .passUntil(s -> s.turnOf(2) && s.liveLeft() == 1);
    Assert.assertTrue(scene.legal(2).contains("NUKI"));
    scene.play("2:NUKI");
    Assert.assertTrue(scene.turnOf(2));
    Assert.assertEquals(scene.liveLeft(), 0);
    Assert.assertEquals(scene.legal(2).get(0), "TSUMO_AGARI");
    WinScore rinshan = scene.engine.decisions().win(0);
    Assert.assertTrue(rinshan.has(ScoringYaku.RINSHAN));
    Assert.assertFalse(rinshan.has(ScoringYaku.HAITEI));
    scene.play("2:DAHAI(1s)");
    Assert.assertEquals(scene.legal(0), List.of("RON_AGARI", "PASS"));
    Assert.assertTrue(scene.legal(1).isEmpty(), "nobody calls the last discard");
    Assert.assertEquals(scene.play("0:RON_AGARI"), Boundary.ROUND_END);
    Assert.assertTrue(scene.end().score(0).has(ScoringYaku.HOUTEI));
  }

  /** 山の最後のツモの手番では、北を持っていても抜けない。 */
  @Test
  public void noNorthIsTakenOutAfterTheLastLiveTile() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "19m258p147s369s47z")
            .hand(1, NOTEN_B)
            .hand(2, NOTEN_C)
            .start()
            .passUntil(s -> s.liveLeft() == 0);
    Assert.assertTrue(scene.turnOf(0));
    List<String> legal = scene.legal(0);
    Assert.assertFalse(legal.contains("NUKI"), legal.toString());
    Assert.assertTrue(legal.contains("DAHAI(N)"), legal.toString());
  }

  /**
   * 三人打ちではチーができず、北はポンできる。ポンの後の合法手には、鳴いた種類の打牌（喰い替え）と北抜きが出ない。
   */
  @Test
  public void noChiButANorthPonAndNoNukiRightAfterAPon() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "19m3p58p147s369s4z6z")
            .hand(1, "333p24p19m258s4z7z5z")
            .hand(2, "44z19m69p147s258s1z")
            .draws("2z3z7p8s")
            .start();
    Assert.assertTrue(scene.legal(0).contains("NUKI"));
    scene.play("0:DAHAI(N)");
    Assert.assertEquals(scene.legal(2), List.of("PON(N)", "PASS"));
    scene.play("2:PASS");
    Assert.assertTrue(scene.turnOf(1));
    scene.play("1:DAHAI_TSUMOGIRI(W)");
    scene.play("2:DAHAI_TSUMOGIRI(7p)");
    scene.play("0:DAHAI(3p)");
    Assert.assertEquals(scene.legal(1), List.of("PON(3p)", "DAIMINKAN(3p)", "PASS"), "no chi from the kamicha");
    scene.play("1:PON(3p)");
    Assert.assertEquals(scene.engine.decisions().kind(0), EngineDecisionKind.POST_CALL_DAHAI);
    Assert.assertEquals(
        scene.legal(1),
        List.of(
            "DAHAI(1m)", "DAHAI(9m)", "DAHAI(2p)", "DAHAI(4p)", "DAHAI(2s)", "DAHAI(5s)", "DAHAI(8s)", "DAHAI(N)",
            "DAHAI(P)", "DAHAI(C)"));
  }

  /** 北を 4 枚持つ親の第一ツモ。北の暗槓も北抜きも選べ、並びは暗槓 → 北抜き → 立直の打牌 → 打牌。 */
  @Test
  public void fourNorthsOfferTheAnkanBeforeTheNuki() {
    Assert.assertEquals(
        fourNorths().legal(0),
        List.of(
            "ANKAN(N)", "NUKI", "RIICHI_DAHAI(N)", "DAHAI_TSUMOGIRI(1m)", "DAHAI(1p)", "DAHAI(2p)", "DAHAI(3p)",
            "DAHAI(4p)", "DAHAI(5p)", "DAHAI(6p)", "DAHAI(7s)", "DAHAI(8s)", "DAHAI(9s)", "DAHAI(N)"));
  }

  /**
   * 打たれた北には、3 枚持つ席がポンも大明槓もできる。北の大明槓はほかの槓と同じ扱いで、嶺上を引き、槓ドラは後めくりになり、
   * 抜きの数は 0 のまま。
   */
  @Test
  public void aDiscardedNorthOffersAPonAndAnOpenKan() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "19m258p147s369s45z")
            .hand(1, "444z123p456p78s19m")
            .hand(2, NOTEN_C)
            .draws("6z")
            .start();
    scene.play("0:DAHAI(N)");
    Assert.assertEquals(scene.legal(1), List.of("PON(N)", "DAIMINKAN(N)", "PASS"));
    Assert.assertTrue(scene.legal(2).isEmpty());
    scene.play("1:DAIMINKAN(N)");
    Assert.assertTrue(scene.turnOf(1));
    SeatView view = scene.view(1);
    Assert.assertTrue(view.last().rinshan());
    Assert.assertTrue(view.last().doraPending());
    Assert.assertEquals(
        view.meld(1, 0), Meld.daiminkan(Tiles.PEI, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE).id());
    Assert.assertEquals(view.nuki(1), 0);
    Assert.assertFalse(scene.legal(1).contains("NUKI"));
  }

  /**
   * 北をポンした席が 4 枚目の北を引くと、加槓も北抜きも選べ、並びは加槓 → 北抜き → 打牌。北抜きなら、ポンはそのまま残り、抜きの数が
   * 1 になる。加槓なら、ポンが加槓になり、抜きの数は 0 のままで、槓ドラは後めくり。
   */
  @Test
  public void aNorthPonTakesTheFourthNorthAsAKakanOrANuki() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "19m258p147s369s45z")
            .hand(1, "44z123p456p789s19m")
            .hand(2, NOTEN_C)
            .draws("6z2z7p4z")
            .start();
    scene.play("0:DAHAI(N)");
    scene.play("1:PON(N)");
    scene.play("1:DAHAI(9m)");
    scene.play("2:DAHAI_TSUMOGIRI(S)");
    scene.play("0:DAHAI_TSUMOGIRI(7p)");
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertEquals(
        scene.legal(1),
        List.of(
            "KAKAN(N)", "NUKI", "DAHAI(1m)", "DAHAI(1p)", "DAHAI(2p)", "DAHAI(3p)", "DAHAI(4p)", "DAHAI(5p)",
            "DAHAI(6p)", "DAHAI(7s)", "DAHAI(8s)", "DAHAI(9s)", "DAHAI_TSUMOGIRI(N)"));
    Meld pon = Meld.pon(Tiles.PEI, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE);

    Engine extracted = scene.engine.copy();
    extracted.commit(new int[] {scene.legal(1).indexOf("NUKI")}, scene.ws);
    Assert.assertEquals(extracted.view(0).nuki(1), 1);
    Assert.assertEquals(extracted.view(0).meld(1, 0), pon.id());

    scene.play("1:KAKAN(N)");
    Assert.assertTrue(scene.turnOf(1));
    SeatView view = scene.view(1);
    Assert.assertTrue(view.last().rinshan());
    Assert.assertTrue(view.last().doraPending());
    Assert.assertEquals(view.meld(1, 0), Meld.kakan(pon, false).id());
    Assert.assertEquals(view.nuki(1), 0);
  }

  /**
   * 北の暗槓はふつうの槓。槓ドラをすぐにめくってから嶺上を引き、抜きの数は 0 のまま。ドラ表示牌が西なら、嶺上開花で和了したとき、
   * 暗槓の 4 枚はほかの牌と同じく表ドラとして 4 翻になり、抜きドラは付かない。
   */
  @Test
  public void aNorthAnkanIsAKanAndItsNorthsAreOrdinaryDora() {
    EngineScene scene = fourNorths();
    scene.play("0:ANKAN(N)");
    Assert.assertTrue(scene.turnOf(0));
    SeatView view = scene.view(0);
    Assert.assertTrue(view.last().rinshan());
    Assert.assertEquals(view.indicatorCount(), 2);
    ChanceLog chance = scene.engine.full().chance();
    Assert.assertEquals(chance.size(), 3);
    Assert.assertEquals(chance.kind(1), ChanceKind.INDICATOR);
    Assert.assertEquals(chance.kind(2), ChanceKind.RINSHAN);
    Assert.assertEquals(view.meld(0, 0), Meld.ankan(Tiles.PEI, false).id());
    Assert.assertEquals(view.nuki(0), 0);
    Assert.assertEquals(view.visible(Tiles.PEI), 4);
    Assert.assertEquals(view.doraWeight(Tiles.PEI), 1);
    Assert.assertEquals(scene.legal(0).get(0), "TSUMO_AGARI");
    Assert.assertFalse(scene.legal(0).contains("NUKI"));

    Assert.assertEquals(scene.play("0:TSUMO_AGARI"), Boundary.ROUND_END);
    WinScore score = scene.end().score(0);
    Assert.assertTrue(score.has(ScoringYaku.RINSHAN), score.toString());
    Assert.assertEquals(score.han(), 6, score.toString());
    Assert.assertEquals(score.dora(), 4, score.toString());
    Assert.assertEquals(score.nuki(), 0, score.toString());
  }

  /**
   * 抜いた北には、役のある和了の形ならロンできる。槍槓は付かず、立直の一発は残る。見送ると全員の一発が消え、見送った席は振聴になり、
   * 抜いた席が補充を引く。
   */
  @Test
  public void aNukiIsRobbedWithIppatsuOrPassedIntoFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "19m258p369s147s5z6z")
            .hand(1, "123p456p789s11z44z")
            .hand(2, "19m369p258s147s3z7z")
            .draws("9p2z8p4z")
            .rinshan("5s")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9p)");
    scene.play("1:RIICHI_TSUMOGIRI(S)");
    scene.play("2:DAHAI_TSUMOGIRI(8p)");
    Assert.assertTrue(scene.legal(0).contains("NUKI"));
    scene.play("0:NUKI");
    Decisions decisions = scene.engine.decisions();
    Assert.assertEquals(decisions.size(), 1);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.NUKI_RESPONSE);
    Assert.assertEquals(scene.legal(1), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.view(1).kind(), EngineDecisionKind.NUKI_RESPONSE);
    Assert.assertTrue(scene.view(1).ippatsu(1));

    Engine won = scene.engine.copy();
    Assert.assertEquals(won.commit(new int[] {0}, scene.ws), Boundary.ROUND_END);
    RoundEnd end = won.roundEnd();
    RoundOutcome outcome = end.outcome();
    Assert.assertEquals(outcome.kind(), EndKind.RON);
    Assert.assertEquals(outcome.winner(0), 1);
    Assert.assertEquals(outcome.loser(), 0);
    WinScore score = end.score(0);
    Assert.assertTrue(score.has(ScoringYaku.IPPATSU));
    Assert.assertFalse(score.has(ScoringYaku.CHANKAN));
    int points = ScoreMath.ron(score.base(), false);
    assertDeltas(end, -points, points, 0);

    scene.play();
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertTrue(scene.view(0).last().rinshan());
    Assert.assertFalse(scene.view(1).ippatsu(1));
    Assert.assertTrue(scene.engine.full().furiten(1));
    Assert.assertEquals(scene.view(1).nuki(0), 1);
  }

  /** 北抜きの補充で和了すれば嶺上開花で、天和にはならず、抜いた北は抜きドラに数える。親のツモはツモ損になる（子の 2 席だけが払う）。 */
  @Test
  public void aWinOnTheNukiReplacementIsRinshanKaihou() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "123p456p789p12s9s4z")
            .hand(1, NOTEN_A)
            .hand(2, "19m369p258s147s67z")
            .indicator("1z")
            .draws("9s")
            .rinshan("3s")
            .start();
    Assert.assertFalse(scene.legal(0).contains("TSUMO_AGARI"));
    scene.play("0:NUKI");
    Assert.assertEquals(scene.legal(0).get(0), "TSUMO_AGARI");
    Assert.assertEquals(scene.play("0:TSUMO_AGARI"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().kind(), EndKind.TSUMO);
    WinScore score = end.score(0);
    Assert.assertTrue(score.has(ScoringYaku.RINSHAN));
    Assert.assertTrue(score.has(ScoringYaku.ITTSU));
    Assert.assertFalse(score.has(ScoringYaku.HAITEI));
    Assert.assertFalse(score.has(ScoringYaku.TENHOU));
    Assert.assertEquals(score.nuki(), 1);
    Assert.assertEquals(scene.view(0).indicatorCount(), 1);
    int each = ScoreMath.tsumo(score.base(), true, false);
    assertDeltas(end, 2 * each, -each, -each);
    assertNext(end, 0, 1, 0);
  }

  /** 大明槓の嶺上ツモの手番で北を抜くと、保留していた槓ドラを補充の前にめくる。北抜きそのものでは新しいドラ表示牌をめくらない。 */
  @Test
  public void aPendingKanDoraIsRevealedBeforeTheNukiReplacement() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "9s19m258p147s36s5z6z")
            .hand(1, "999s4z123p456p78p1m")
            .hand(2, "19m689p258s147s37z")
            .draws("2z")
            .rinshan("5s1z")
            .start();
    scene.play("0:DAHAI(9s)");
    Assert.assertEquals(scene.legal(1), List.of("PON(9s)", "DAIMINKAN(9s)", "PASS"));
    scene.play("1:DAIMINKAN(9s)");
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertEquals(scene.view(0).indicatorCount(), 1);
    Assert.assertTrue(scene.view(1).last().doraPending());
    scene.play("1:NUKI");
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertEquals(scene.view(0).indicatorCount(), 2);
    Assert.assertFalse(scene.view(1).last().doraPending());
    ChanceLog chance = scene.engine.full().chance();
    Assert.assertEquals(chance.kind(chance.size() - 3), ChanceKind.RINSHAN);
    Assert.assertEquals(chance.kind(chance.size() - 2), ChanceKind.INDICATOR);
    Assert.assertEquals(chance.kind(chance.size() - 1), ChanceKind.RINSHAN);
    scene.play("1:DAHAI_TSUMOGIRI(E)");
    Assert.assertTrue(scene.turnOf(2));
    Assert.assertEquals(scene.view(0).indicatorCount(), 2);
  }

  /**
   * 立直中に抜けるのは、ツモった北だけ。ツモ和了できる北を抜いた立直の席は、局の終わりまで振聴になる。抜いた北は河に入らないが、
   * 次の和了牌にもロンできない。
   */
  @Test
  public void aRiichiSeatTakesOutOnlyTheDrawnNorthAndSkippingATsumoIsFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "123p456p789s11z44z")
            .hand(1, NOTEN_A)
            .hand(2, "19m369p258s147s67z")
            .draws("9m3z8p4z1z")
            .rinshan("5p")
            .start();
    scene.play("0:RIICHI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(W)");
    scene.play("2:DAHAI_TSUMOGIRI(8p)");
    Assert.assertEquals(scene.legal(0), List.of("TSUMO_AGARI", "NUKI", "DAHAI_TSUMOGIRI(N)"));
    scene.play("0:NUKI");
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertEquals(scene.legal(0), List.of("DAHAI_TSUMOGIRI(5p)"), "the north in the hand stays");
    Assert.assertTrue(scene.engine.full().furiten(0));
    scene.play("0:DAHAI_TSUMOGIRI(5p)");
    scene.play("1:DAHAI_TSUMOGIRI(E)");
    Assert.assertTrue(scene.turnOf(2), "the furiten riichi seat has no ron on its winning tile");
    Assert.assertTrue(scene.engine.full().furiten(0));
  }

  /** 九種九牌は三人打ちでも親の第一ツモで選べる。先に自分で北を抜くと第一巡が途切れ、補充の手番では選べない。 */
  @Test
  public void nineTerminalsAbortUnlessANukiCameFirst() {
    EngineScene abort = nineTerminals();
    List<String> legal = abort.legal(0);
    Assert.assertEquals(legal.get(legal.size() - 1), "KYUSHU_KYUHAI");
    Assert.assertEquals(abort.play("0:KYUSHU_KYUHAI"), Boundary.ROUND_END);
    assertAbortive(abort.end(), AbortiveDraw.NINE_TERMINALS);
    assertDeltas(abort.end(), 0, 0, 0);
    assertNext(abort.end(), 0, 1, 0);

    EngineScene nuki = nineTerminals();
    nuki.play("0:NUKI");
    Assert.assertTrue(nuki.turnOf(0));
    Assert.assertFalse(nuki.legal(0).contains("KYUSHU_KYUHAI"));
  }

  /** 第一巡に 3 人が同じ風牌を打っても流局しない。 */
  @Test
  public void threeSameWindsDoNotAbort() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1z19m258p147s369s5z")
            .hand(1, "1z19m369p258s147s6z")
            .hand(2, "1z19m147p258p369s7z")
            .draws("2z3z6p")
            .start();
    scene.play("0:DAHAI(E)");
    scene.play("1:DAHAI(E)");
    Assert.assertEquals(scene.play("2:DAHAI(E)"), Boundary.DECISIONS);
    Assert.assertTrue(scene.turnOf(0));
  }

  /** 3 人全員が立直しても流局しない。 */
  @Test
  public void threeRiichiDoNotAbort() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, TENPAI)
            .hand(1, TENPAI_SOU)
            .hand(2, "789p123s456s6677z")
            .draws("9m1m9m")
            .start();
    scene.play("0:RIICHI_TSUMOGIRI(9m)");
    scene.play("1:RIICHI_TSUMOGIRI(1m)");
    Assert.assertEquals(scene.play("2:RIICHI_TSUMOGIRI(9m)"), Boundary.DECISIONS);
    Assert.assertTrue(scene.turnOf(0));
    for (int seat = 0; seat < VARIANT.seats(); seat++) {
      Assert.assertEquals(scene.view(0).riichi(seat), RiichiState.DOUBLE_RIICHI);
    }
    Assert.assertEquals(scene.view(0).kyotaku(), 3);
  }

  /** 2 人の槓が合わせて 4 つになった後、次の打牌が通ると四槓散了になる。 */
  @Test
  public void fourKansByTwoSeatsAbortAfterTheNextDiscard() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1111p2222p58s1s67z")
            .hand(1, "3333p4444p137s56z")
            .hand(2, "19m689p258s147s57z")
            .draws("9s9p")
            .rinshan("2z3z2s3s")
            .start();
    scene.play("0:ANKAN(1p)");
    scene.play("0:ANKAN(2p)");
    scene.play("0:DAHAI(9s)");
    scene.play("1:ANKAN(3p)");
    Assert.assertEquals(scene.play("1:ANKAN(4p)"), Boundary.DECISIONS);
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertFalse(scene.legal(1).stream().anyMatch(name -> name.contains("KAN")));
    Assert.assertEquals(scene.view(0).indicatorCount(), 5);
    Assert.assertEquals(scene.play("1:DAHAI(9p)"), Boundary.ROUND_END);
    assertAbortive(scene.end(), AbortiveDraw.FOUR_KANS);
    assertNext(scene.end(), 0, 1, 0);
  }

  /** 不聴罰符の総額は 2000 点。1 人聴牌なら聴牌の席が +2000 で不聴の席が −1000 ずつ、2 人聴牌なら聴牌の席が +1000 ずつで不聴の席が −2000。 */
  @Test
  public void notenPaymentsTotalTwoThousand() {
    EngineScene one = exhaustive(EngineScene.first(VARIANT), TENPAI, NOTEN_A, NOTEN_B);
    Assert.assertEquals(one.end().outcome().tenpaiMask(), 0b001);
    assertDeltas(one.end(), 2_000, -1_000, -1_000);
    assertNext(one.end(), 0, 1, 0);

    EngineScene two = exhaustive(EngineScene.first(VARIANT), TENPAI, TENPAI_SOU, NOTEN_B);
    Assert.assertEquals(two.end().outcome().tenpaiMask(), 0b011);
    assertDeltas(two.end(), 1_000, 1_000, -2_000);
  }

  /** 流し満貫は満貫のツモと同じ払いで、ツモ損がある（子 +6000、親 +8000）。本場は付かない。 */
  @Test
  public void nagashiManganPaysAManganTsumoWithTheMissingSeat() {
    EngineScene child = nagashi(1);
    Assert.assertEquals(child.end().outcome().nagashiMask(), 0b010);
    assertDeltas(child.end(), -4_000, 6_000, -2_000);

    EngineScene dealer = nagashi(0);
    Assert.assertEquals(dealer.end().outcome().nagashiMask(), 0b001);
    assertDeltas(dealer.end(), 8_000, -4_000, -4_000);
  }

  /** ダブロンでは、本場（1 本 200 点）と供託は、放銃者の下家から手番の順に数えて最初の和了者だけが受け取る。 */
  @Test
  public void aDoubleRonPaysHonbaAndDepositsToTheFirstWinner() {
    EngineScene scene =
        EngineScene.at(VARIANT, 0, 2, 1, 35_000, 35_000, 34_000)
            .hand(0, NOTEN_A)
            .hand(1, "123s456s789s46p11z")
            .hand(2, "234s678s678p34p55s")
            .draws("5p")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(5p)");
    Assert.assertEquals(scene.legal(1), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("1:RON_AGARI", "2:RON_AGARI"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    RoundOutcome outcome = end.outcome();
    Assert.assertEquals(outcome.winnerCount(), 2);
    Assert.assertEquals(outcome.winner(0), 1);
    Assert.assertEquals(outcome.winner(1), 2);
    int first = ScoreMath.ron(end.score(0).base(), false);
    int second = ScoreMath.ron(end.score(1).base(), false);
    int honba = 2 * VARIANT.ronHonba();
    assertDeltas(end, -first - honba - second, first + honba + 1_000, second);
    assertNext(end, 1, 0, 0);
  }

  /** 南 3 局で親が流れたとき、トップが 40000 点に届いていれば終局し、届かなければ西入する。 */
  @Test
  public void theSouthThreeEndsOnlyWithATopAtTheReturnPoints() {
    EngineScene stop = exhaustive(EngineScene.at(VARIANT, 5, 0, 0, 39_000, 33_000, 33_000), TENPAI, NOTEN_A, NOTEN_B);
    Assert.assertEquals(stop.boundary(), Boundary.GAME_END);
    Assert.assertEquals(stop.end().next().score(0), 41_000);

    EngineScene west = exhaustive(EngineScene.at(VARIANT, 5, 0, 0, 35_000, 35_000, 35_000), TENPAI, NOTEN_A, NOTEN_B);
    Assert.assertEquals(west.boundary(), Boundary.ROUND_END);
    assertNext(west.end(), 6, 1, 0);
  }

  /** 西 3 局は、親が流れれば点に関わらず終局する。親が連荘すれば、親が 40000 点以上のトップでない限り同じ局を続ける。 */
  @Test
  public void theWestThreeEndsWhenTheDealerPasses() {
    EngineScene passed =
        exhaustive(EngineScene.at(VARIANT, 8, 0, 0, 35_000, 35_000, 35_000), TENPAI, NOTEN_A, NOTEN_B);
    Assert.assertEquals(passed.boundary(), Boundary.GAME_END);

    EngineScene stays =
        exhaustive(EngineScene.at(VARIANT, 8, 0, 0, 36_000, 35_000, 34_000), NOTEN_A, NOTEN_B, TENPAI);
    Assert.assertEquals(stays.boundary(), Boundary.ROUND_END);
    assertNext(stays.end(), 8, 1, 0);

    EngineScene stops =
        exhaustive(EngineScene.at(VARIANT, 8, 0, 0, 30_000, 35_000, 40_000), NOTEN_A, NOTEN_B, TENPAI);
    Assert.assertEquals(stops.boundary(), Boundary.GAME_END);
    Assert.assertEquals(stops.end().next().score(2), 42_000);
  }

  /** 親は北を含む 13 種の么九牌を 1 枚ずつ持ち、第一ツモは 5p。 */
  private static EngineScene nineTerminals() {
    return EngineScene.first(VARIANT)
        .hand(0, "19m19p19s1234567z")
        .hand(1, NOTEN_A)
        .hand(2, "19m369p258s147s67z")
        .draws("5p")
        .rinshan("5s")
        .start();
  }

  /** 親の配牌は北 4 枚・123p・456p・789s で、第一ツモと嶺上ツモは 1m。ドラ表示牌は西（北がドラ）、槓ドラの表示牌は東。 */
  private static EngineScene fourNorths() {
    return EngineScene.first(VARIANT)
        .hand(0, "4444z123p456p789s")
        .hand(1, NOTEN_A)
        .hand(2, NOTEN_C)
        .indicator("3z")
        .indicator(1, "1z")
        .draws("1m")
        .rinshan("1m")
        .start();
  }

  /** 席の順に配牌を置き、全員がツモ切りと見送りを続けて荒牌まで打つ。流し満貫にならないように、最初のツモは么九牌にしない。 */
  private static EngineScene exhaustive(EngineScene scene, String... hands) {
    for (int seat = 0; seat < hands.length; seat++) {
      scene.hand(seat, hands[seat]);
    }
    return scene.drawAt(0, "4s4s4s").start().passUntil(EngineScene::ended);
  }

  /**
   * 1 本場で、その席のツモを全て么九牌にし、ほかの席の最初のツモを么九牌でない牌にして、全員がツモ切りで荒牌まで打つ。山の k 番目の
   * ツモは、k を 3 で割った余りの席が引く。
   *
   * @param seat 流し満貫にする席
   */
  private static EngineScene nagashi(int seat) {
    String[] terminals = "1m9m1p9p1s9s1z2z3z5z6z7z1m9m1p9p1s9s1z".split("(?<=[mpsz])");
    EngineScene scene = EngineScene.at(VARIANT, 0, 1, 0, 35_000, 35_000, 35_000);
    int seats = VARIANT.seats();
    for (int other = 0; other < seats; other++) {
      if (other != seat) {
        scene.drawAt(other, "5s");
      }
    }
    for (int k = seat, index = 0; k < VARIANT.liveWall(); k += seats, index++) {
      scene.drawAt(k, terminals[index]);
    }
    scene.start().passUntil(EngineScene::ended);
    Assert.assertEquals(scene.end().outcome().kind(), EndKind.EXHAUSTIVE);
    return scene;
  }

  private static void assertAbortive(RoundEnd end, AbortiveDraw draw) {
    Assert.assertEquals(end.outcome().kind(), EndKind.ABORTIVE);
    Assert.assertEquals(end.outcome().abortive(), draw);
  }

  private static void assertDeltas(RoundEnd end, int... expected) {
    for (int seat = 0; seat < expected.length; seat++) {
      Assert.assertEquals(end.delta(seat), expected[seat], "delta of seat " + seat);
    }
  }

  private static void assertNext(RoundEnd end, int round, int honba, int kyotaku) {
    TableState next = end.next();
    Assert.assertFalse(next.gameOver(), next.toString());
    Assert.assertEquals(next.round(), round, next.toString());
    Assert.assertEquals(next.honba(), honba, next.toString());
    Assert.assertEquals(next.kyotaku(), kyotaku, next.toString());
  }
}
