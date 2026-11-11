package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 仮のツモの点（{@code FullView.tsumoScore}）を確かめる。実際に引いた種類なら、席のビューから求めた和了の点
 * （WinSettlementProjection）と同じになる。ほかの種類でも、ツモの出どころで決まる役（嶺上開花・海底・天和と地和）は実際のツモと同じ。
 * 引いた赤は数えない。引けない種類と、ツモの直後でない呼び出しは拒む。
 */
public class TsumoScoreTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int TSUMO_CODE = Action.code(Action.Type.TSUMO_AGARI);

  /** 1z・2z の双碰の聴牌。 */
  private static final String TENPAI = "123m456m789m1122z";

  /** どの 2 枚も塔子や対子にならない不聴の手（3 通り）。 */
  private static final String NOTEN_A = "147m258p369s2357z";

  private static final String NOTEN_B = "147m258p369s3456z";
  private static final String NOTEN_C = "147m258p369s4567z";

  private final WinScore out = new WinScore();

  /** 暗槓の後の嶺上牌で、単騎待ちの牌を引く。ほかの種類で求めても嶺上開花が付く。槓子にした種類は 5 枚目が無いので拒む。 */
  @Test
  public void aRinshanDrawScoresEveryKindAsRinshan() {
    EngineScene scene = EngineScene.first(VARIANT).hand(0, "1111m234p567p789s").draws("5z").rinshan("5z").start();
    scene.play("0:ANKAN(1m)");
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertTrue(scene.view(0).last().rinshan());
    assertLikeTheProjection(scene, 0);
    Assert.assertTrue(out.has(ScoringYaku.RINSHAN), out.toString());
    Assert.assertTrue(out.has(ScoringYaku.MENZEN_TSUMO), out.toString());
    Assert.assertFalse(scene.engine.full().tsumoScore(0, Tiles.P1, scene.ws, out));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> scene.engine.full().tsumoScore(0, Tiles.M1, scene.ws, out));
  }

  /** 山の最後のツモは海底で、待ちのどちらの種類で求めても海底が付く。 */
  @Test
  public void theLastLiveTileScoresEveryKindAsHaitei() {
    EngineScene scene = EngineScene.first(VARIANT).hand(1, TENPAI).drawAt(VARIANT.liveWall() - 1, "1z").start();
    scene.passUntil(s -> s.liveLeft() == 0 && s.turnOf(1));
    assertLikeTheProjection(scene, 1);
    Assert.assertTrue(out.has(ScoringYaku.HAITEI), out.toString());
    Assert.assertTrue(out.has(ScoringYaku.YAKUHAI_ROUND), out.toString());
    Assert.assertTrue(scene.engine.full().tsumoScore(1, Tiles.NAN, scene.ws, out));
    Assert.assertTrue(out.has(ScoringYaku.HAITEI), out.toString());
    Assert.assertTrue(out.has(ScoringYaku.YAKUHAI_SEAT), out.toString());
    Assert.assertFalse(scene.engine.full().tsumoScore(1, Tiles.SHA, scene.ws, out));
  }

  /** 鳴きや槓で巡が途切れていない局での自分の最初のツモなら、和了になるどの種類で求めても、親は天和、子は地和になる。 */
  @Test
  public void theFirstDrawScoresEveryKindAsTenhouOrChiihou() {
    EngineScene dealer = EngineScene.first(VARIANT).hand(0, TENPAI).draws("1z").start();
    assertLikeTheProjection(dealer, 0);
    Assert.assertTrue(out.has(ScoringYaku.TENHOU), out.toString());
    Assert.assertTrue(dealer.engine.full().tsumoScore(0, Tiles.NAN, dealer.ws, out));
    Assert.assertTrue(out.has(ScoringYaku.TENHOU), out.toString());
    Assert.assertFalse(dealer.engine.full().tsumoScore(0, Tiles.SHA, dealer.ws, out));

    EngineScene child =
        EngineScene.first(VARIANT)
            .hand(0, NOTEN_C)
            .hand(1, TENPAI)
            .hand(2, NOTEN_A)
            .hand(3, NOTEN_B)
            .draws("9p1z")
            .start();
    child.play("0:DAHAI_TSUMOGIRI(9p)");
    Assert.assertTrue(child.turnOf(1));
    assertLikeTheProjection(child, 1);
    Assert.assertTrue(out.has(ScoringYaku.CHIIHOU), out.toString());
    Assert.assertTrue(child.engine.full().tsumoScore(1, Tiles.NAN, child.ws, out));
    Assert.assertTrue(out.has(ScoringYaku.CHIIHOU), out.toString());
  }

  /** 引いた赤の 5 は、同じ種類で求めた点には数えない。翻は赤 1 枚分だけ減り、役と符は同じ。 */
  @Test
  public void aDrawnRedFiveIsReplacedByAPlainTile() {
    EngineScene scene =
        EngineScene.first(VARIANT)
            .hand(0, "123789m46p11222z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_B)
            .hand(3, NOTEN_C)
            .draws("9p9p9p9p0p")
            .start();
    for (int seat = 0; seat < VARIANT.seats(); seat++) {
      scene.play(seat + ":DAHAI_TSUMOGIRI(9p)");
    }
    Assert.assertTrue(scene.turnOf(0));
    Assert.assertTrue(scene.view(0).last().red());
    WinScore projected = projection(scene, 0);
    Assert.assertEquals(projected.aka(), 1);
    Assert.assertTrue(scene.engine.full().tsumoScore(0, Tiles.P5, scene.ws, out));
    Assert.assertEquals(out.aka(), 0, out.toString());
    Assert.assertEquals(out.han(), projected.han() - 1, out.toString());
    Assert.assertEquals(out.fu(), projected.fu(), out.toString());
    Assert.assertEquals(out.yaku(), projected.yaku(), out.toString());
    Assert.assertFalse(out.has(ScoringYaku.TENHOU), out.toString());
    Assert.assertFalse(scene.engine.full().tsumoScore(0, Tiles.P3, scene.ws, out));
  }

  /** ツモの直後の席でなければ IllegalStateException を投げる。作業領域のルールがエンジンと違うときと、ルールに無い種類には IllegalArgumentException を投げる。 */
  @Test
  public void onlyTheSeatThatJustDrewCanScoreATsumo() {
    EngineScene pon = EngineScene.first(VARIANT).hand(1, "99p147m258s369s12z").draws("9p").start();
    FullView full = pon.engine.full();
    Assert.assertThrows(IllegalStateException.class, () -> full.tsumoScore(1, Tiles.P9, pon.ws, out));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> full.tsumoScore(0, Tiles.P9, new RuleWorkspace(Variant.TENHOU_3P), out));
    pon.play("0:DAHAI_TSUMOGIRI(9p)");
    Assert.assertEquals(pon.engine.decisions().kind(0), EngineDecisionKind.RESPONSE);
    Assert.assertThrows(IllegalStateException.class, () -> full.tsumoScore(0, Tiles.P9, pon.ws, out));

    EngineScene three = EngineScene.first(Variant.TENHOU_3P).start();
    Assert.assertThrows(
        IllegalArgumentException.class, () -> three.engine.full().tsumoScore(0, Tiles.M2, three.ws, out));
  }

  /**
   * 実際に引いた種類で求めた仮のツモの点が、ビューから求めたツモ和了の点と同じになることを確かめる。求めた点はフィールドの
   * {@code out} に残るので、呼び出し元は続けて役を確かめられる。
   *
   * @param seat 手番の席
   */
  private void assertLikeTheProjection(EngineScene scene, int seat) {
    WinScore projected = projection(scene, seat);
    Assert.assertTrue(scene.engine.full().tsumoScore(seat, scene.view(seat).last().tile(), scene.ws, out));
    Assert.assertEquals(out.toString(), projected.toString());
  }

  /**
   * 手番の席のツモ和了の点を、席のビューから求める。ツモ和了は合法手の先頭にある。
   *
   * @param seat 手番の席
   */
  private static WinScore projection(EngineScene scene, int seat) {
    Assert.assertEquals(scene.engine.decisions().legal(0, 0), TSUMO_CODE, scene.legal(seat).toString());
    WinSettlementProjection projection = new WinSettlementProjection(VARIANT);
    projection.load(scene.view(seat), TSUMO_CODE, scene.ws);
    return projection.score();
  }
}
