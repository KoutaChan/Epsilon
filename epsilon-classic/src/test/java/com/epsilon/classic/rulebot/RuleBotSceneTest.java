package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import org.testng.annotations.Test;

/**
 * ルールベースの打ち手の決まりを 1 つずつ、場面を作って確かめる。残りの牌は決まったシードで並べるので、どの場面も他家の手牌まで
 * 毎回同じになる。
 */
public class RuleBotSceneTest {

  private final RuleBot bot = new RuleBot(RuleScene.VARIANT);

  /** 配牌で和了の形になっていれば（天和）、ツモ和了する。 */
  @Test
  public void aCompleteHandWinsByTsumo() {
    RuleScene scene = new RuleScene().hand(0, "123m456m789m123p5p").draws("5p").indicator("2z").start();
    assertChoice(scene, 0, "TSUMO_AGARI");
  }

  /** 和了牌が打たれたら、チーできてもロンする。 */
  @Test
  public void aWinningDiscardIsRonnedBeforeAnyCall() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "5p147m258s369s123z")
            .hand(1, "123m456m789m234p5p")
            .draws("4z")
            .indicator("2z")
            .start()
            .play("0:DAHAI(5p)");
    assertTrue(scene.legal(1).contains("CHI(3p4p5p 5p)"), scene.legal(1).toString());
    assertChoice(scene, 1, "RON_AGARI");
  }

  /** ダマではロンできる役が無い聴牌は、立直する。 */
  @Test
  public void aTenpaiWithoutAYakuDeclaresRiichi() {
    RuleScene scene = new RuleScene().hand(0, "123m789p555s99s46p").draws("1z").indicator("2z").start();
    assertChoice(scene, 0, "RIICHI_TSUMOGIRI(E)");
  }

  /** ダマのロンで 7700 点以上ある聴牌（ここでは親の断么九・平和・三色）は、立直しない。 */
  @Test
  public void aValuableDamatenStaysDama() {
    RuleScene scene = new RuleScene().hand(0, "234m234p234s67s55p").draws("1z").indicator("2z").start();
    assertTrue(scene.legal(0).contains("RIICHI_TSUMOGIRI(E)"), scene.legal(0).toString());
    assertChoice(scene, 0, "DAHAI_TSUMOGIRI(E)");
  }

  /** 他家が立直したら、遠い手（3 向聴）は浮いた字牌ではなく現物を打って降りる。宣言牌はポンせずに見送る。 */
  @Test
  public void aFarHandFoldsToARiichiWithTheSafeTile() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s1357z")
            .hand(1, "123m456m789m11z23p")
            .hand(2, "99s13m57p24s68m157z")
            .hand(3, "258m147p369p2468s")
            .draws("6z9s2z")
            .indicator("3z")
            .start()
            .play("0:DAHAI_TSUMOGIRI(F)")
            .play("1:RIICHI_TSUMOGIRI(9s)");
    assertTrue(scene.legal(2).contains("PON(9s)"), scene.legal(2).toString());
    assertChoice(scene, 2, "PASS");
    scene.play();
    assertChoice(scene, 2, "DAHAI(9s)");
  }

  /** 宣言牌が通る前から、立直を宣言した席を脅威として扱う。脅威が無ければポンする役牌の対子でも、宣言牌はポンしない。 */
  @Test
  public void aRiichiDeclarationTileIsNotCalledEvenForADragonPon() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s2367z")
            .hand(1, "123m456m789m23p15z")
            .hand(2, "55z234m56p78s9m124z")
            .hand(3, "258m147p369p2468s")
            .draws("9p1z")
            .indicator("8p")
            .start()
            .play("0:DAHAI_TSUMOGIRI(9p)")
            .play("1:RIICHI_DAHAI(P)");
    assertTrue(scene.legal(2).contains("PON(P)"), scene.legal(2).toString());
    assertChoice(scene, 2, "PASS");
  }

  /** 役牌の対子は、ポンして向聴が進むならポンする。 */
  @Test
  public void aDragonPairIsPonnedWhenItAdvancesTheHand() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "5z147m258p369s246z")
            .hand(2, "55z234m56p78s9m124z")
            .draws("1s")
            .indicator("8p")
            .start()
            .play("0:DAHAI(P)");
    assertChoice(scene, 2, "PON(P)");
  }

  /** 鳴いても役の見込みが無い手（么九牌があり、3 色にまたがる）は、チーしない。 */
  @Test
  public void aChiWithoutAYakuIsDeclined() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s1357z")
            .hand(1, "46s123m789p13s9m14z")
            .draws("5s")
            .indicator("8p")
            .start()
            .play("0:DAHAI_TSUMOGIRI(5s)");
    assertTrue(scene.legal(1).contains("CHI(4s5s6s 5s)"), scene.legal(1).toString());
    assertChoice(scene, 1, "PASS");
  }

  /**
   * 役が無くてロンできない和了牌を見送っても、立直していなければ次の自分の打牌で見逃しの振聴は消えるので、打牌を比べるときはロンを
   * 数える。6m をポンした聴牌で 3p を引いたとき、3p を打つ両面待ち（役があるのは 4p だけ）より、2p を打つ 5s と 3p のシャンポン待ち
   * （ロンの平均の点数が高い）を選ぶ。見送りが無いときと同じ手になる。
   */
  @Test
  public void aMissedWinDoesNotCostTheRonOfTheNextDiscard() {
    assertChoice(openTenpaiAfter("9p"), 1, "DAHAI(2p)");
    RuleScene missed = openTenpaiAfter("1p");
    assertTrue(missed.observe(1).furitenTemporary(), "the 1p was a wait without a yaku");
    assertChoice(missed, 1, "DAHAI(2p)");
  }

  /**
   * 席 1 が 6m をポンして 1p と 4p の待ちの聴牌になり、下家がツモ切りした後、次の自分のツモで 3p を引いた場面を作る。ドラは 5s。
   *
   * @param shimochaDiscard 下家がツモ切りする牌
   */
  private static RuleScene openTenpaiAfter(String shimochaDiscard) {
    return new RuleScene()
        .hand(0, "147m258p369s1235z")
        .hand(1, "66m234m678s55s23p4z")
        .hand(2, "159m159p19s12345z")
        .hand(3, "258m147p369p2468s")
        .draws("6m" + shimochaDiscard + "7z6z3p")
        .indicator("4s")
        .start()
        .play("0:DAHAI_TSUMOGIRI(6m)")
        .play("1:PON(6m)")
        .play("1:DAHAI(N)")
        .play("2:DAHAI_TSUMOGIRI(" + shimochaDiscard + ")")
        .play("3:DAHAI_TSUMOGIRI(C)")
        .play("0:DAHAI_TSUMOGIRI(F)");
  }

  /**
   * ルールベースの打ち手が、その席の判断で期待どおりの手を選ぶことを確かめる。同じ判断の記録で 2 回選んでも、評価が最も高い手を取っても、
   * 同じ手になることも確かめる。
   *
   * @param scene 判断を待っている場面
   * @param seat 判断する席
   * @param expected 選ぶはずの手の名前（{@code Action.toString} の形）
   */
  private void assertChoice(RuleScene scene, int seat, String expected) {
    Observation o = scene.observe(seat);
    int chosen = bot.act(o);
    assertEquals(o.action(chosen).toString(), expected, scene.legal(seat).toString());
    assertEquals(bot.act(o), chosen);
    assertEquals(bot.evaluate(o).greedy(), chosen);
  }
}
