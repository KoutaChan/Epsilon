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

/** 四人打ちの場面のテスト。手で組んだ牌山でエンジンに局を打たせ、ルールどおりの流れと局末を確かめる。 */
public class EngineSceneTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** 1z・2z の双碰の聴牌。 */
  private static final String TENPAI = "123m456m789m1122z";

  /** どの 2 枚も塔子や対子にならない不聴の手（3 通り）。 */
  private static final String NOTEN_A = "147m258p369s2357z";

  private static final String NOTEN_B = "147m258p369s3456z";
  private static final String NOTEN_C = "147m258p369s4567z";

  /** 九種九牌は親の第一ツモで選べる。選ぶと流局し、親が続いて本場が 1 つ増える。 */
  @Test
  public void nineTerminalsAbortAndTheDealerStays() {
    EngineScene scene = EngineScene.first(VARIANT).hand(0, "19m19p19s1234z234m").draws("5z").start();
    List<String> legal = scene.legal(0);
    Assert.assertEquals(legal.get(legal.size() - 1), "KYUSHU_KYUHAI");
    Assert.assertEquals(scene.play("0:KYUSHU_KYUHAI"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    assertAbortive(end, AbortiveDraw.NINE_TERMINALS);
    assertDeltas(end, 0, 0, 0, 0);
    assertNext(end, 0, 1, 0);
    Assert.assertEquals(end.waits(0), 0L);
  }

  /** 鳴きや槓で途切れていない第一巡で 4 人が同じ風牌を打つと、4 枚目が通った時点で四風連打の流局になる。 */
  @Test
  public void fourWindsAbortAfterTheFourthWind() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "147m258p369s1235z")
            .hand(1, "258m369p147s1246z")
            .hand(2, "369m147p258s1347z")
            .hand(3, "147m258p369s1567z")
            .draws("5m5s6m6p")
            .start();
    for (int seat = 0; seat < 3; seat++) {
      scene.play(seat + ":DAHAI(E)");
      Assert.assertTrue(scene.turnOf(seat + 1));
    }
    Assert.assertEquals(scene.play("3:DAHAI(E)"), Boundary.ROUND_END);
    assertAbortive(scene.end(), AbortiveDraw.FOUR_WINDS);
    assertNext(scene.end(), 0, 1, 0);
  }

  /** 4 人目の立直の宣言牌が通ると四家立直。4 本の供託は場に残り、親が続く。 */
  @Test
  public void fourRiichiAbortAndKeepTheDeposits() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, TENPAI)
            .hand(1, "123p456p789p3344z")
            .hand(2, "123s456s789s5566z")
            .hand(3, "234m567p678s99m77z")
            .draws("1s9p1p1s")
            .start();
    scene.play("0:RIICHI_TSUMOGIRI(1s)");
    scene.play("1:RIICHI_TSUMOGIRI(9p)");
    scene.play("2:RIICHI_TSUMOGIRI(1p)");
    Assert.assertEquals(scene.play("3:RIICHI_TSUMOGIRI(1s)"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    assertAbortive(end, AbortiveDraw.FOUR_RIICHI);
    Assert.assertEquals(end.outcome().riichiMask(), 0b1111);
    for (int seat = 0; seat < 4; seat++) {
      Assert.assertEquals(scene.view(0).riichi(seat), RiichiState.DOUBLE_RIICHI);
    }
    assertDeltas(end, -1_000, -1_000, -1_000, -1_000);
    assertNext(end, 0, 1, 4);
  }

  /** ダブロンでは、本場と供託は、放銃者の下家から手番の順に数えて最初の和了者だけが受け取る。 */
  @Test
  public void aDoubleRonPaysHonbaAndDepositsToTheFirstWinner() {
    EngineScene scene =
        EngineScene.at(VARIANT, 1, 2, 1, 25_000, 25_000, 24_000, 25_000)
            .hand(1, "5p147m147s369m123z")
            .hand(2, "123m456m789m46p11z")
            .hand(3, "234s567s678m34p55s")
            .hand(0, "19p258m1479s4567z")
            .draws("4z")
            .start();
    Assert.assertTrue(scene.turnOf(1));
    scene.play("1:DAHAI(5p)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "CHI(4p5p6p 5p)", "PASS"));
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("2:RON_AGARI", "3:RON_AGARI"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    RoundOutcome outcome = end.outcome();
    Assert.assertEquals(outcome.kind(), EndKind.RON);
    Assert.assertEquals(outcome.loser(), 1);
    Assert.assertEquals(outcome.winnerCount(), 2);
    Assert.assertEquals(outcome.winner(0), 2);
    Assert.assertEquals(outcome.winner(1), 3);
    Assert.assertTrue(end.score(0).has(ScoringYaku.ITTSU));
    Assert.assertTrue(end.score(1).has(ScoringYaku.TANYAO));
    int first = ScoreMath.ron(end.score(0).base(), false);
    int second = ScoreMath.ron(end.score(1).base(), false);
    int honba = 2 * VARIANT.ronHonba();
    assertDeltas(end, 0, -first - honba - second, first + honba + 1_000, second);
    assertNext(end, 2, 0, 0);
  }

  /** 3 人が同じ打牌にロンすると三家和で流局する。点は動かず、親が続く。 */
  @Test
  public void threeRonsAbortTheRound() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "5p147m369m19p2467z")
            .hand(1, "123m456m789m46p99s")
            .hand(2, "234s567s678m34p55s")
            .hand(3, "111z123s789s55p33z")
            .draws("6z")
            .start();
    scene.play("0:DAHAI(5p)");
    Assert.assertEquals(scene.legal(3), List.of("RON_AGARI", "PON(5p)", "PASS"));
    Assert.assertEquals(scene.play("1:RON_AGARI", "2:RON_AGARI", "3:RON_AGARI"), Boundary.ROUND_END);
    assertAbortive(scene.end(), AbortiveDraw.TRIPLE_RON);
    assertDeltas(scene.end(), 0, 0, 0, 0);
    assertNext(scene.end(), 0, 1, 0);
  }

  /** 役の無い待ちでも、加槓した牌には槍槓の役でロンできる。打牌への応答ではロンは出ない。 */
  @Test
  public void anAddedKanIsRobbedWithChankan() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "22p147m58s9p12345z")
            .hand(1, "147p258m369s1234z")
            .hand(2, "34p111m789s456m99s")
            .hand(3, "369p147s258m4567z")
            .draws("6m2p8p1z9m2p")
            .start();
    scene.play("0:DAHAI(E)");
    scene.play("1:DAHAI_TSUMOGIRI(2p)");
    Assert.assertEquals(scene.legal(2), List.of("CHI(2p3p4p 2p)", "PASS"));
    scene.play("0:PON(2p)");
    scene.play("0:DAHAI(S)");
    scene.play("1:DAHAI_TSUMOGIRI(8p)");
    scene.play("2:DAHAI_TSUMOGIRI(E)");
    scene.play("3:DAHAI_TSUMOGIRI(9m)");
    Assert.assertTrue(scene.legal(0).contains("KAKAN(2p)"));
    scene.play("0:KAKAN(2p)");
    Decisions decisions = scene.engine.decisions();
    Assert.assertEquals(decisions.size(), 1);
    Assert.assertEquals(decisions.kind(0), EngineDecisionKind.CHANKAN);
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("2:RON_AGARI"), Boundary.ROUND_END);
    RoundOutcome outcome = scene.end().outcome();
    Assert.assertEquals(outcome.kind(), EndKind.RON);
    Assert.assertEquals(outcome.winner(0), 2);
    Assert.assertEquals(outcome.loser(), 0);
    Assert.assertTrue(scene.end().score(0).has(ScoringYaku.CHANKAN));
  }

  /** 大三元の 3 つ目の三元牌（暗槓 2 つの後のポン）を鳴かせた席が、ツモの全額を払う（親の役満 48000 で飛んで終局）。 */
  @Test
  public void theSeatThatFedTheThirdDragonPaysTheWholeTsumo() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "5555z6666z77z1m23p")
            .hand(1, "147p258s369s1234z")
            .hand(2, "258p147s369m1234z")
            .hand(3, "369p258s147m1234z")
            .draws("9s7z5m5m5s1p")
            .rinshan("9s8m")
            .start();
    scene.play("0:ANKAN(P)");
    scene.play("0:ANKAN(F)");
    scene.play("0:DAHAI(1m)");
    scene.play("1:DAHAI_TSUMOGIRI(C)");
    scene.play("0:PON(C)");
    scene.play("0:DAHAI(8m)");
    scene.play("1:DAHAI_TSUMOGIRI(5m)");
    scene.play("2:DAHAI_TSUMOGIRI(5m)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(0).get(0), "TSUMO_AGARI");
    Assert.assertEquals(scene.play("0:TSUMO_AGARI"), Boundary.GAME_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().kind(), EndKind.TSUMO);
    Assert.assertEquals(end.outcome().pao(0), 1);
    Assert.assertTrue(end.score(0).has(ScoringYaku.DAISANGEN));
    assertDeltas(end, 48_000, -48_000, 0, 0);
    Assert.assertEquals(end.next().score(1), -23_000);
  }

  /** 子の大三元のツモ。3 つ目の三元牌を鳴かせた子が、親の分を含む全額 32000 を 1 人で払う。親は払わない。 */
  @Test
  public void theFeederPaysTheDealerShareOfAChildTsumo() {
    EngineScene scene = thirdDragonFedBySeatTwo();
    scene.play("2:DAHAI(W)");
    scene.play("3:DAHAI(W)");
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    Assert.assertEquals(scene.legal(1).get(0), "TSUMO_AGARI");
    Assert.assertEquals(scene.play("1:TSUMO_AGARI"), Boundary.GAME_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().kind(), EndKind.TSUMO);
    Assert.assertEquals(end.outcome().pao(0), 2);
    Assert.assertTrue(end.score(0).has(ScoringYaku.DAISANGEN));
    assertDeltas(end, 0, 32_000, -32_000, 0);
  }

  /** 3 つ目の三元牌を鳴かせた席からのロンは、包の無い普通のロン。放銃者が 32000 を払う。 */
  @Test
  public void aRonFromTheFeederIsAPlainRon() {
    EngineScene scene = thirdDragonFedBySeatTwo();
    scene.play("2:DAHAI_TSUMOGIRI(1p)");
    Assert.assertEquals(scene.legal(1), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("1:RON_AGARI"), Boundary.GAME_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().loser(), 2);
    Assert.assertEquals(end.outcome().pao(0), RoundOutcome.NO_SEAT);
    Assert.assertTrue(end.score(0).has(ScoringYaku.DAISANGEN));
    assertDeltas(end, 0, 32_000, -32_000, 0);
  }

  /** 3 つ目の三元牌を鳴かせた親からのロンも、包の無い普通のロン。親が 32000 を払う。 */
  @Test
  public void aRonFromTheDealerThatFedTheThirdDragonIsAPlainRon() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "147m258p369s1234z")
            .hand(1, "5555z6666z77z1m23p")
            .hand(2, "258m369p147s1234z")
            .hand(3, "369m147p258s1234z")
            .draws("9m9s5m5s7z5m5s1p")
            .rinshan("9s8m")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:ANKAN(P)");
    scene.play("1:ANKAN(F)");
    scene.play("1:DAHAI(1m)");
    scene.play("2:DAHAI_TSUMOGIRI(5m)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    scene.play("0:DAHAI_TSUMOGIRI(C)");
    scene.play("1:PON(C)");
    scene.play("1:DAHAI(8m)");
    scene.play("2:DAHAI_TSUMOGIRI(5m)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    scene.play("0:DAHAI_TSUMOGIRI(1p)");
    Assert.assertEquals(scene.legal(1).get(0), "RON_AGARI");
    Assert.assertEquals(scene.play("1:RON_AGARI"), Boundary.GAME_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().loser(), 0);
    Assert.assertEquals(end.outcome().pao(0), RoundOutcome.NO_SEAT);
    assertDeltas(end, -32_000, 32_000, 0, 0);
  }

  /** 親の大三元のロン。3 つ目の三元牌のポンを加槓しても、包の席は元のポンを鳴かせた席のまま。その席からのロンは普通のロン（48000）。 */
  @Test
  public void aRonFromTheFeederOfAnAddedKanIsAPlainRon() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "5555z6666z77z1m23p")
            .hand(1, "147p258s369s1234z")
            .hand(2, "258p147s369m1234z")
            .hand(3, "369p258s147m1234z")
            .draws("9s7z5m5m5s7z1p")
            .rinshan("9s8m9p")
            .start();
    scene.play("0:ANKAN(P)");
    scene.play("0:ANKAN(F)");
    scene.play("0:DAHAI(1m)");
    scene.play("1:DAHAI_TSUMOGIRI(C)");
    scene.play("0:PON(C)");
    scene.play("0:DAHAI(8m)");
    scene.play("1:DAHAI_TSUMOGIRI(5m)");
    scene.play("2:DAHAI_TSUMOGIRI(5m)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    scene.play("0:KAKAN(C)");
    scene.play("0:DAHAI_TSUMOGIRI(9p)");
    scene.play("1:DAHAI_TSUMOGIRI(1p)");
    Assert.assertEquals(scene.legal(0), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("0:RON_AGARI"), Boundary.GAME_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().loser(), 1);
    Assert.assertEquals(end.outcome().pao(0), RoundOutcome.NO_SEAT);
    Assert.assertTrue(end.score(0).has(ScoringYaku.DAISANGEN));
    assertDeltas(end, 48_000, -48_000, 0, 0);
  }

  /** 3 つ目の三元牌を鳴かせた席とは別の席からのロンでは、放銃者と包の席が 16000 ずつ払う。 */
  @Test
  public void aRonFromAnotherSeatIsSplitWithTheFeeder() {
    EngineScene scene = thirdDragonFedBySeatTwo();
    scene.play("2:DAHAI(W)");
    scene.play("3:DAHAI_TSUMOGIRI(4p)");
    Assert.assertEquals(scene.legal(1), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("1:RON_AGARI"), Boundary.ROUND_END);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().loser(), 3);
    Assert.assertEquals(end.outcome().pao(0), 2);
    assertDeltas(end, 0, 32_000, -16_000, -16_000);
  }

  /**
   * 山の残りが 1 枚のときの暗槓。嶺上ツモで山が 0 枚になる。そのツモで和了すれば嶺上開花で、海底は付かない。続く打牌へのロンは河底。
   */
  @Test
  public void aKanOnTheLastLiveTileLeadsToRinshanAndThenHoutei() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "9999p123m456m77z8s")
            .hand(1, "7z147p258m36s2345z")
            .hand(2, "123m456m789s111z7z")
            .hand(3, "258p147s369m3456z")
            .draws("1m2p3s4m")
            .drawAt(68, "9s")
            .rinshan("7s")
            .start();
    scene.passUntil(s -> s.turnOf(0) && s.liveLeft() == 1);
    Assert.assertTrue(scene.legal(0).contains("ANKAN(9p)"));
    scene.play("0:ANKAN(9p)");
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertEquals(scene.liveLeft(), 0);
    Assert.assertEquals(scene.view(0).indicatorCount(), 2);
    Assert.assertEquals(scene.legal(0).get(0), "TSUMO_AGARI");
    WinScore rinshan = scene.engine.decisions().win(0);
    Assert.assertTrue(rinshan.has(ScoringYaku.RINSHAN));
    Assert.assertFalse(rinshan.has(ScoringYaku.HAITEI));
    scene.play("0:DAHAI(C)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("2:RON_AGARI"), Boundary.ROUND_END);
    Assert.assertEquals(scene.end().outcome().winner(0), 2);
    Assert.assertTrue(scene.end().score(0).has(ScoringYaku.HOUTEI));
  }

  /** 2 人の槓が合わせて 4 つになっても局は続く。次の打牌にロンが無ければ四槓散了で、その打牌には鳴けない。 */
  @Test
  public void fourKansByTwoSeatsAbortAfterTheNextDiscard() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1111m2222m58p1s67z")
            .hand(1, "3333m4444m137s56z")
            .hand(2, "99p147s258s9m1234z")
            .hand(3, "147p258p369s4567z")
            .draws("9s9p")
            .rinshan("2z3z2p3p")
            .start();
    scene.play("0:ANKAN(1m)");
    scene.play("0:ANKAN(2m)");
    scene.play("0:DAHAI(9s)");
    scene.play("1:ANKAN(3m)");
    Assert.assertEquals(scene.play("1:ANKAN(4m)"), Boundary.DECISIONS);
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertFalse(scene.legal(1).stream().anyMatch(name -> name.contains("KAN")));
    Assert.assertEquals(scene.view(0).indicatorCount(), 5);
    Assert.assertEquals(scene.play("1:DAHAI(9p)"), Boundary.ROUND_END);
    assertAbortive(scene.end(), AbortiveDraw.FOUR_KANS);
    assertNext(scene.end(), 0, 1, 0);
  }

  /** チーの後は、鳴いた牌と、筋の喰い替えになる牌（2m を 234m でチーしたなら 5m）を打てない。 */
  @Test
  public void aChiBansTheCalledKindAndItsSuji() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "2m147p258s369p457z")
            .hand(1, "2345m678p159s123z")
            .hand(2, "19m19p19s4567z238s")
            .hand(3, "389m258p1477s366z")
            .draws("1z")
            .start();
    scene.play("0:DAHAI(2m)");
    Assert.assertEquals(scene.legal(1), List.of("CHI(2m3m4m 2m)", "PASS"));
    scene.play("1:CHI(2m3m4m 2m)");
    Assert.assertEquals(scene.engine.decisions().kind(0), EngineDecisionKind.POST_CALL_DAHAI);
    Assert.assertEquals(scene.view(1).meld(1, 0), Meld.chi(Tiles.M2, Tiles.M2, Meld.AkaSource.NONE).id());
    Assert.assertEquals(scene.view(1).count(Tiles.M2), 1);
    Assert.assertEquals(scene.view(1).count(Tiles.M5), 1);
    List<String> legal = scene.legal(1);
    Assert.assertFalse(legal.contains("DAHAI(2m)"));
    Assert.assertFalse(legal.contains("DAHAI(5m)"));
    Assert.assertEquals(
        legal,
        List.of("DAHAI(6p)", "DAHAI(7p)", "DAHAI(8p)", "DAHAI(1s)", "DAHAI(5s)", "DAHAI(9s)", "DAHAI(E)",
            "DAHAI(S)", "DAHAI(W)"));
  }

  /**
   * 見逃しの同巡振聴は、自分が対面からポンして打牌すると解ける。自分のツモより前でも、次の打牌にロンできる（天鳳の牌譜にある形:
   * 下家の打牌を見送り、対面からポンして打牌し、次の打牌にロン）。
   */
  @Test
  public void aToimenPonClearsTheSameTurnFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1478m369p149s123z")
            .hand(1, "258m19p258s34567z")
            .hand(2, "777z234m678m55p67s")
            .hand(3, "39m147p28p19s1234z")
            .draws("9m9p1z5s5p7s")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(9p)");
    scene.play("2:DAHAI_TSUMOGIRI(E)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    scene.play();
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("0:DAHAI_TSUMOGIRI(5p)");
    scene.play("2:PON(5p)");
    scene.play("2:DAHAI(6s)");
    Assert.assertFalse(scene.engine.full().furiten(2));
    Assert.assertEquals(scene.engine.full().waits(2), Tiles.bit(Tiles.S7));
    scene.play("3:DAHAI_TSUMOGIRI(7s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("2:RON_AGARI"), Boundary.ROUND_END);
    Assert.assertEquals(scene.end().outcome().winner(0), 2);
    Assert.assertEquals(scene.end().outcome().loser(), 3);
  }

  /**
   * 見逃しの同巡振聴は、自分が対面から大明槓し、嶺上ツモの後に打牌すると解ける。上家からポンして打牌しても解ける。どちらも、
   * 自分のツモより前の和了牌にロンできる。
   */
  @Test
  public void aToimenOpenKanAndAKamichaPonClearTheSameTurnFuriten() {
    EngineScene kan = furitenSeatFacesAToimenDiscard();
    kan.play("2:DAIMINKAN(1m)");
    kan.play("2:DAHAI_TSUMOGIRI(P)");
    Assert.assertFalse(kan.engine.full().furiten(2));
    kan.play("3:DAHAI_TSUMOGIRI(9p)");
    Assert.assertEquals(kan.legal(2), List.of("PON(9p)", "PASS"));
    kan.play();
    kan.play("0:DAHAI_TSUMOGIRI(8s)");
    Assert.assertEquals(kan.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(kan.play("2:RON_AGARI"), Boundary.ROUND_END);
    Assert.assertEquals(kan.end().outcome().loser(), 0);

    EngineScene pon = furitenSeatFacesAToimenDiscard();
    pon.play();
    pon.play("1:DAHAI_TSUMOGIRI(9p)");
    pon.play("2:PON(9p)");
    pon.play("2:DAHAI(1m)");
    Assert.assertFalse(pon.engine.full().furiten(2));
    pon.play("3:DAHAI_TSUMOGIRI(8s)");
    Assert.assertEquals(pon.legal(2), List.of("RON_AGARI", "PASS"));
  }

  /** 見逃しの同巡振聴は、自分が上家からチーして打牌すると解け、自分のツモより前の和了牌にロンできる。 */
  @Test
  public void aKamichaChiClearsTheSameTurnFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1478m369p149s123z")
            .hand(1, "258m19p258s34567z")
            .hand(2, "777z234m678m456s9p")
            .hand(3, "39m147p28p19s1234z")
            .draws("9m1p1z9p2s3s6s")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(1p)");
    scene.play("2:DAHAI_TSUMOGIRI(E)");
    scene.play("3:DAHAI_TSUMOGIRI(9p)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    scene.play();
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("0:DAHAI_TSUMOGIRI(2s)");
    scene.play("1:DAHAI_TSUMOGIRI(3s)");
    Assert.assertEquals(scene.legal(2), List.of("CHI(3s4s5s 3s)", "PASS"));
    scene.play("2:CHI(3s4s5s 3s)");
    scene.play("2:DAHAI(9p)");
    Assert.assertFalse(scene.engine.full().furiten(2));
    Assert.assertEquals(scene.engine.full().waits(2), Tiles.bit(Tiles.S6));
    scene.passUntil(s -> s.turnOf(3));
    scene.play("3:DAHAI_TSUMOGIRI(6s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    Assert.assertEquals(scene.play("2:RON_AGARI"), Boundary.ROUND_END);
    Assert.assertEquals(scene.end().outcome().loser(), 3);
  }

  /**
   * 見逃しの同巡振聴は、自分が下家からポンして打牌しても解けず、自分のツモより前の和了牌にはロンできない。次に自分がツモって打牌すると解ける。
   */
  @Test
  public void aShimochaPonKeepsTheSameTurnFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1478m369p149s123z")
            .hand(1, "258m19p147s34567z")
            .hand(2, "777z234m678m55p67s")
            .hand(3, "55s5p147m28p369s14z")
            .draws("9m5s7s9p1z2z7s")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(2).get(0), "RON_AGARI");
    scene.play("3:PON(5s)");
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("3:DAHAI(5p)");
    scene.play("2:PON(5p)");
    scene.play("2:DAHAI(6s)");
    Assert.assertTrue(scene.engine.full().furiten(2));
    Assert.assertEquals(scene.engine.full().waits(2), Tiles.bit(Tiles.S7));
    scene.play("3:DAHAI_TSUMOGIRI(7s)");
    Assert.assertTrue(scene.turnOf(0), "the furiten seat has no ron on its winning tile");
    scene.play("0:DAHAI_TSUMOGIRI(9p)");
    scene.play("1:DAHAI_TSUMOGIRI(E)");
    scene.play("2:DAHAI_TSUMOGIRI(S)");
    Assert.assertFalse(scene.engine.full().furiten(2));
    scene.play("3:DAHAI_TSUMOGIRI(7s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
  }

  /**
   * 見逃しの同巡振聴は、自分が下家から大明槓し、嶺上ツモの後に打牌しても解けず、自分のツモより前の和了牌にはロンできない。
   * 次に自分がツモって打牌すると解ける。
   */
  @Test
  public void aShimochaOpenKanKeepsTheSameTurnFuriten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1478m369p149s123z")
            .hand(1, "258m19p147s34567z")
            .hand(2, "777z234m777p88m67s")
            .hand(3, "55s7p147m28p369s14z")
            .draws("9m5s8s9p1z2z8s")
            .rinshan("4z")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(2).get(0), "RON_AGARI");
    scene.play("3:PON(5s)");
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("3:DAHAI(7p)");
    Assert.assertEquals(scene.legal(2), List.of("PON(7p)", "DAIMINKAN(7p)", "PASS"));
    scene.play("2:DAIMINKAN(7p)");
    scene.play("2:DAHAI_TSUMOGIRI(N)");
    Assert.assertTrue(scene.engine.full().furiten(2));
    Assert.assertEquals(scene.engine.full().waits(2), Tiles.bit(Tiles.S5) | Tiles.bit(Tiles.S8));
    scene.play("3:DAHAI_TSUMOGIRI(8s)");
    Assert.assertTrue(scene.turnOf(0), "the furiten seat has no ron on its winning tile");
    scene.play("0:DAHAI_TSUMOGIRI(9p)");
    scene.play("1:DAHAI_TSUMOGIRI(E)");
    scene.play("2:DAHAI_TSUMOGIRI(S)");
    Assert.assertFalse(scene.engine.full().furiten(2));
    scene.play("3:DAHAI_TSUMOGIRI(8s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
  }

  /** 立直中にロンを見逃すと、自分が打牌した後も見逃しの印が残り、次の和了牌にもロンできない。 */
  @Test
  public void aRonPassedInRiichiStaysForTheRound() {
    EngineScene scene = waitingOnFiveAndEightSou();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(9p)");
    scene.play("2:RIICHI_TSUMOGIRI(E)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    scene.play();
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("0:DAHAI_TSUMOGIRI(1p)");
    scene.play("1:DAHAI_TSUMOGIRI(S)");
    scene.play("2:DAHAI_TSUMOGIRI(W)");
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("3:DAHAI_TSUMOGIRI(8s)");
    Assert.assertTrue(scene.turnOf(0), "the riichi seat has no ron after passing one");
  }

  /** 立直していない席の見逃しの印は、次に自分が打牌すると消える。その打牌が立直の宣言牌でも消える。 */
  @Test
  public void theRiichiDeclarationDiscardClearsAnEarlierPassedRon() {
    EngineScene scene = waitingOnFiveAndEightSou();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:DAHAI_TSUMOGIRI(9p)");
    scene.play("2:DAHAI_TSUMOGIRI(E)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    scene.play();
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("0:DAHAI_TSUMOGIRI(1p)");
    scene.play("1:DAHAI_TSUMOGIRI(S)");
    scene.play("2:RIICHI_TSUMOGIRI(W)");
    Assert.assertFalse(scene.engine.full().furiten(2));
    Assert.assertEquals(scene.view(2).riichi(2), RiichiState.RIICHI);
    scene.play("3:DAHAI_TSUMOGIRI(8s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
  }

  /** 河が么九牌だけで、どの牌も鳴かれていない席は流し満貫。満貫のツモと同じ払いで、本場は付かず、不聴罰符も無い。 */
  @Test
  public void aRiverOfTerminalsAndHonorsWinsNagashiMangan() {
    String[] terminals = "1m9m1p9p1s9s1z2z3z4z5z6z7z1m9m1p9p1s".split("(?<=[mpsz])");
    EngineScene scene =
        EngineScene.at(VARIANT, 0, 1, 0, 25_000, 25_000, 25_000, 25_000).drawAt(0, "5m").drawAt(2, "5p5s");
    for (int index = 0; index < terminals.length; index++) {
      scene.drawAt(1 + 4 * index, terminals[index]);
    }
    scene.start().passUntil(EngineScene::ended);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().kind(), EndKind.EXHAUSTIVE);
    Assert.assertEquals(end.outcome().nagashiMask(), 0b0010);
    assertDeltas(end, -4_000, 8_000, -2_000, -2_000);
    boolean dealerTenpai = (end.outcome().tenpaiMask() & 1) != 0;
    assertNext(end, dealerTenpai ? 0 : 1, 2, 0);
  }

  /** 立直できるのは、山の残りがルールの下限（四人打ちは 4 枚）以上ある手番だけ。 */
  @Test
  public void riichiNeedsAtLeastTheMinimumWall() {
    EngineScene scene =
        EngineScene.first(VARIANT).hand(1, TENPAI).hand(2, "123p456p789p3344z").drawAt(0, "5m5p5s6m").start();
    int minimum = VARIANT.riichiMinWall();
    scene.passUntil(s -> s.turnOf(1) && s.liveLeft() == minimum);
    Assert.assertTrue(scene.legal(1).stream().anyMatch(name -> name.startsWith("RIICHI")));
    scene.passUntil(s -> s.turnOf(2));
    Assert.assertEquals(scene.liveLeft(), minimum - 1);
    Assert.assertFalse(scene.legal(2).stream().anyMatch(name -> name.startsWith("RIICHI")));
  }

  /** ちょうど 0 点では飛ばない。 */
  @Test
  public void exactlyZeroPointsKeepThePlayerInTheGame() {
    EngineScene scene =
        EngineScene.at(VARIANT, 0, 0, 0, 33_000, 33_000, 33_000, 1_000)
            .hand(0, TENPAI)
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_B)
            .hand(3, NOTEN_C)
            .drawAt(0, "5m5m5s5s")
            .start()
            .passUntil(EngineScene::ended);
    Assert.assertEquals(scene.boundary(), Boundary.ROUND_END);
    Assert.assertEquals(scene.end().outcome().tenpaiMask(), 0b0001);
    assertDeltas(scene.end(), 3_000, -1_000, -1_000, -1_000);
    Assert.assertEquals(scene.end().next().score(3), 0);
    assertNext(scene.end(), 0, 1, 0);
  }

  /** 南 4 局で親が流れても、トップが返し点に届いていなければ西入する。西場では、誰かが返し点に届いた局で終わる。 */
  @Test
  public void theWestRoundEndsAsSoonAsSomeoneReachesTheReturnPoints() {
    EngineScene south =
        EngineScene.at(VARIANT, 7, 0, 0, 25_000, 25_000, 25_000, 25_000)
            .hand(0, TENPAI)
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_B)
            .hand(3, NOTEN_C)
            .drawAt(0, "5m5m5s5s")
            .start()
            .passUntil(EngineScene::ended);
    Assert.assertEquals(south.boundary(), Boundary.ROUND_END);
    assertNext(south.end(), 8, 1, 0);

    EngineScene west =
        EngineScene.at(VARIANT, 8, 0, 0, 25_000, 28_000, 24_000, 23_000)
            .hand(0, NOTEN_A)
            .hand(1, TENPAI)
            .hand(2, NOTEN_B)
            .hand(3, NOTEN_C)
            .drawAt(0, "5m5m5s5s")
            .start()
            .passUntil(EngineScene::ended);
    Assert.assertEquals(west.boundary(), Boundary.GAME_END);
    Assert.assertTrue(west.end().next().gameOver());
    Assert.assertEquals(west.end().next().score(1), 31_000);
  }

  /** オーラスで親が連荘しても、親が返し点以上のトップなら終局する。届かなければ同じ局を続ける。 */
  @Test
  public void theLastDealerStopsOnlyAsATopAtTheReturnPoints() {
    EngineScene stop =
        EngineScene.at(VARIANT, 7, 0, 0, 24_000, 24_000, 24_000, 28_000)
            .hand(0, NOTEN_A)
            .hand(1, NOTEN_B)
            .hand(2, NOTEN_C)
            .hand(3, TENPAI)
            .drawAt(0, "5m5m5s5s")
            .start()
            .passUntil(EngineScene::ended);
    Assert.assertEquals(stop.boundary(), Boundary.GAME_END);
    Assert.assertEquals(stop.end().next().score(3), 31_000);

    EngineScene go =
        EngineScene.at(VARIANT, 7, 0, 0, 25_000, 25_000, 25_000, 25_000)
            .hand(0, NOTEN_A)
            .hand(1, NOTEN_B)
            .hand(2, NOTEN_C)
            .hand(3, TENPAI)
            .drawAt(0, "5m5m5s5s")
            .start()
            .passUntil(EngineScene::ended);
    Assert.assertEquals(go.boundary(), Boundary.ROUND_END);
    assertNext(go.end(), 7, 1, 0);
  }

  /** 荒牌のとき、純手牌に 4 枚ある種類だけを待つ形は不聴。 */
  @Test
  public void aWaitOnAKindHeldFourTimesIsNoten() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "1111m234p567p789s")
            .hand(1, "234m567m789p1122z")
            .hand(2, "258m147p369s3456z")
            .hand(3, "369m258p147s4567z")
            .drawAt(0, "6m6s6m6s")
            .start()
            .passUntil(EngineScene::ended);
    RoundEnd end = scene.end();
    Assert.assertEquals(end.outcome().tenpaiMask(), 0b0010);
    Assert.assertEquals(end.waits(0), 0L);
    assertDeltas(end, -1_000, 3_000, -1_000, -1_000);
  }

  /**
   * 大明槓の槓ドラは後めくり。嶺上ツモの手番ではめくらず、その手番の打牌の後、他家の応答の前にめくる。嶺上ツモで和了すればめくらない。
   */
  @Test
  public void anOpenKanRevealsItsDoraAfterTheNextDiscard() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "9s147m258p36p2357z")
            .hand(1, "999s123m456m78p11z")
            .hand(2, "11z258m147s369m26z")
            .hand(3, "147p258s368m4567z")
            .draws("2s")
            .rinshan("9p")
            .start();
    scene.play("0:DAHAI(9s)");
    Assert.assertEquals(scene.legal(1), List.of("PON(9s)", "DAIMINKAN(9s)", "PASS"));
    scene.play("1:DAIMINKAN(9s)");
    Assert.assertTrue(scene.turnOf(1));
    Assert.assertEquals(scene.view(0).indicatorCount(), 1);
    Assert.assertTrue(scene.view(1).last().doraPending());
    Assert.assertEquals(scene.legal(1).get(0), "TSUMO_AGARI");

    Engine won = scene.engine.copy();
    Assert.assertEquals(won.commit(new int[] {0}, scene.ws), Boundary.ROUND_END);
    Assert.assertEquals(won.view(0).indicatorCount(), 1);
    Assert.assertTrue(won.roundEnd().score(0).has(ScoringYaku.RINSHAN));

    scene.play("1:DAHAI(E)");
    Assert.assertEquals(scene.legal(2), List.of("PON(E)", "PASS"));
    Assert.assertEquals(scene.view(0).indicatorCount(), 2);
    ChanceLog chance = scene.engine.full().chance();
    Assert.assertEquals(chance.kind(chance.size() - 2), ChanceKind.RINSHAN);
    Assert.assertEquals(chance.kind(chance.size() - 1), ChanceKind.INDICATOR);
  }

  /**
   * 席 2（中の暗刻、5s・8s 待ち）が 5s のロンを見送って同巡振聴になり、対面の席 0 が打った 1m にポン・大明槓・見送りを選べる判断の時点。
   * 続く山のツモは 9p・8s で、嶺上は白。
   */
  private static EngineScene furitenSeatFacesAToimenDiscard() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "258m147p369s1234z")
            .hand(1, "369m258p147s1567z")
            .hand(2, "111m777z234m99p67s")
            .hand(3, "479m136p258s2345z")
            .draws("8p1s6p5s1m9p8s")
            .rinshan("5z")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(8p)");
    scene.play("1:DAHAI_TSUMOGIRI(1s)");
    scene.play("2:DAHAI_TSUMOGIRI(6p)");
    scene.play("3:DAHAI_TSUMOGIRI(5s)");
    Assert.assertEquals(scene.legal(2), List.of("RON_AGARI", "PASS"));
    scene.play();
    Assert.assertTrue(scene.engine.full().furiten(2));
    scene.play("0:DAHAI_TSUMOGIRI(1m)");
    Assert.assertEquals(scene.legal(2), List.of("PON(1m)", "DAIMINKAN(1m)", "PASS"));
    return scene;
  }

  /**
   * 子の席 1 が最初の手番で白と發を暗槓し、席 2 が打った中を 3 つ目の三元牌としてポンして大三元の 1p・4p 待ち（2p3p 9s9s）になった後、
   * 席 2 が 1p をツモった手番。続く山のツモは 4p（席 3）、9m（席 0）、1p（席 1）。
   */
  private static EngineScene thirdDragonFedBySeatTwo() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "147m258p369s1234z")
            .hand(1, "5555z6666z77z1m23p")
            .hand(2, "258m369p147s1234z")
            .hand(3, "369m147p258s1234z")
            .draws("9m9s7z1p4p9m1p")
            .rinshan("9s8m")
            .start();
    scene.play("0:DAHAI_TSUMOGIRI(9m)");
    scene.play("1:ANKAN(P)");
    scene.play("1:ANKAN(F)");
    scene.play("1:DAHAI(1m)");
    scene.play("2:DAHAI_TSUMOGIRI(C)");
    scene.play("1:PON(C)");
    scene.play("1:DAHAI(8m)");
    Assert.assertTrue(scene.turnOf(2));
    Assert.assertEquals(scene.engine.full().waits(1), Tiles.bit(Tiles.P1) | Tiles.bit(Tiles.P4));
    return scene;
  }

  /** 席 2（中の暗刻、5s・8s 待ち）と、山のツモ（9m・9p・東・5s・1p・南・西・8s）をどれも鳴けない 3 席の場面。5s と 8s は席 3 が引く。 */
  private static EngineScene waitingOnFiveAndEightSou() {
    return EngineScene.first(VARIANT)
        .hand(0, "1478m369p149s123z")
        .hand(1, "258m19p258s34567z")
        .hand(2, "777z234m678m55p67s")
        .hand(3, "39m147p28p19s1234z")
        .draws("9m9p1z5s1p2z3z8s")
        .start();
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
