package com.epsilon.prism3p.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism3p.input.InputSchema.DiscardExact;
import com.epsilon.prism3p.input.InputSchema.Furiten;
import com.epsilon.prism3p.input.InputSchema.RoundField;
import com.epsilon.prism3p.input.InputSchema.TransitionField;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 遷移の後の振聴が、その手を打った後のエンジンの見逃しの印と合うことを、手で作った局面で確かめる。見るのは、立直中に和了牌の北を抜く遷移と、
 * 役が無くてロンできない和了牌を見送る遷移。
 */
public class FuritenTransitionTest {

  /** 北を持たない不聴の手。 */
  private static final String NOTEN_A = "19m258p147s369s67z";

  private static final String NOTEN_B = "19m369p258s147p57z";
  private static final String NOTEN_C = "19m369p258s369s57z";

  private final RowEncoder r = new RowEncoder();

  private Observation at(PlayedScene scene, int seat) {
    return r.encode(scene.engine, scene.engine.decisions().slotOf(seat));
  }

  private static int action(Observation o, Action.Type type) {
    for (int index = 0; index < o.legalActions(); index++) {
      if (o.action(index).type() == type) {
        return index;
      }
    }
    throw new AssertionError("no " + type);
  }

  /** 1 索と北のシャンポンで立直した席が、北をツモる。 */
  @Test
  public void aRiichiSeatThatTakesOutAWinningNorthStaysInFuriten() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "123p456p789s11s4z4z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("7z9p8p4z")
            .indicator("1m")
            .start();
    scene.play("0:RIICHI_TSUMOGIRI(C)");
    scene.passUntil(s -> s.turnOf(0));
    assertNukiOfAWinningNorth(scene);
  }

  /**
   * 北を 1 枚抜いた後に 1 索と北のシャンポンで立直した席が、4 枚目の北をツモる。抜いた後の手は北を 4 枚とも持つので、待ちは 1 索だけになる。
   */
  @Test
  public void theFourthNorthTakenOutInRiichiIsAlsoAMissedWin() {
    PlayedScene scene =
        PlayedScene.first()
            .hand(0, "123p456p789s11s4z4z")
            .hand(1, NOTEN_A)
            .hand(2, NOTEN_C)
            .draws("4z9p8p4z")
            .rinshan("7z")
            .indicator("1m")
            .start();
    scene.play("0:NUKI");
    scene.play("0:RIICHI_TSUMOGIRI(C)");
    scene.passUntil(s -> s.turnOf(0));
    assertNukiOfAWinningNorth(scene);
  }

  /**
   * 立直中の席 0 が、ツモ和了できる北を引いた手番を確かめる。北を抜く遷移の振聴は立直後の見逃しで、ロンの見込みは 0、ツモの見込みは残る。
   * 北を抜くと、エンジンもその席を振聴にする。
   */
  private void assertNukiOfAWinningNorth(PlayedScene scene) {
    Observation o = at(scene, 0);
    action(o, Action.Type.TSUMO_AGARI);
    int transition = r.column(action(o, Action.Type.NUKI), ActionColumn.FIRST_TRANSITION);
    assertEquals(
        r.transition(transition, TransitionField.FURITEN),
        RowEncoder.id(TransitionField.FURITEN, Furiten.RIICHI_PERSISTENT));
    assertEquals(r.discardExact(transition, DiscardExact.RON_ENDS), 0f);
    assertEquals(r.discardExact(transition, DiscardExact.RON_UTILITY), 0f);
    assertEquals(r.discardExact(transition, DiscardExact.RON_POINTS), 0f);
    assertTrue(r.discardExact(transition, DiscardExact.TSUMO_POINTS) > 0f, "a tsumo is still possible");

    scene.play("0:NUKI");
    assertTrue(scene.engine.view(0).furiten(), "the engine keeps the seat in furiten");
    assertTrue(scene.turnOf(0));
    at(scene, 0);
    assertEquals(r.round(RoundField.FURITEN_RIICHI), 1);
  }

  /**
   * 3 索と白のシャンポンのダマの手は、3 索のロンに役が無い。上家が 3 索を打つと合法手はポンと見送りだけになる。見送りの遷移の振聴は同巡の
   * 見逃しで、見送ると、エンジンも次の自分の打牌までその席を振聴にする。
   */
  @Test
  public void passingAWinningTileWithoutYakuIsATemporaryFuriten() {
    PlayedScene scene =
        PlayedScene.first().hand(0, NOTEN_B).hand(1, "111p456p789s33s55z").hand(2, NOTEN_C).draws("3s").start();
    scene.play("0:DAHAI_TSUMOGIRI(3s)");
    assertEquals(scene.legal(1), List.of("PON(3s)", "PASS"));
    Observation o = at(scene, 1);
    int transition = r.column(action(o, Action.Type.PASS), ActionColumn.FIRST_TRANSITION);
    assertEquals(
        r.transition(transition, TransitionField.FURITEN), RowEncoder.id(TransitionField.FURITEN, Furiten.TEMPORARY));

    scene.step();
    assertTrue(scene.engine.view(1).furiten(), "the engine puts the seat in furiten");
    assertTrue(scene.turnOf(1));
    at(scene, 1);
    assertEquals(r.round(RoundField.FURITEN_TEMPORARY), 1);
  }
}
