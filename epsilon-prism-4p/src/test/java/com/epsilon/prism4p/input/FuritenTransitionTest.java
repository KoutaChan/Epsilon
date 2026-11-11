package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism4p.input.InputSchema.Furiten;
import com.epsilon.prism4p.input.InputSchema.RoundField;
import com.epsilon.prism4p.input.InputSchema.TransitionField;
import org.testng.annotations.Test;

/** 遷移の後の振聴が、その手を打った後のエンジンの見逃しの印と合うことを、手で作った局面で確かめる。 */
public class FuritenTransitionTest {

  private final RowEncoder r = new RowEncoder();
  private final Engine engine = new Engine(TENHOU_4P);
  private final RuleWorkspace ws = new RuleWorkspace(TENHOU_4P);

  /**
   * 3 索と白のシャンポンのダマの手は、3 索のロンに役が無い。上家が 3 索を打つと合法手はポンと見送りだけになる。見送ると、エンジンは次の自分の
   * 打牌まで振聴にする。見送りの遷移の振聴は同巡の見逃し。
   */
  @Test
  public void passingAWinningTileWithoutYakuIsATemporaryFuriten() {
    Scenes.first(TENHOU_4P)
        .hand(0, "19m369p258s147p57z")
        .hand(1, "111m456m789p33s55z")
        .hand(2, "29m258p147s369s67z")
        .hand(3, "28m369p258s147p46z")
        .draws("3s")
        .start(engine, ws);
    play(0, Action.Type.DAHAI, Tiles.S3);
    Decisions decisions = engine.decisions();
    assertEquals(decisions.size(), 1);
    assertEquals(decisions.seat(0), 1);
    Observation o = r.encode(engine, 0);
    assertEquals(o.legalActions(), 2);
    assertEquals(o.action(0).type(), Action.Type.PON);
    assertEquals(o.action(1).type(), Action.Type.PASS);
    int transition = r.column(1, ActionColumn.FIRST_TRANSITION);
    assertEquals(
        category(Tensor.TRANSITION_CATEGORY, transition, InputSchema.TRANSITION.column(TransitionField.FURITEN)),
        Ids.of(TransitionField.FURITEN.spec().values(), Furiten.TEMPORARY));

    play(1, Action.Type.PASS, -1);
    assertTrue(engine.view(1).furiten(), "the engine puts the seat in furiten");
    assertEquals(engine.decisions().kind(0), EngineDecisionKind.TURN);
    assertEquals(engine.decisions().seat(0), 1);
    r.encode(engine, 0);
    assertEquals(category(Tensor.ROUND_CATEGORY, 0, InputSchema.ROUND.column(RoundField.FURITEN_TEMPORARY)), 1);
  }

  /** 判断する席が 1 つだけの判断の時点で、その席が指定した種類と牌の手を選ぶ。牌を見ないときは負の値を渡す。 */
  private void play(int seat, Action.Type type, int tile) {
    Decisions decisions = engine.decisions();
    assertEquals(decisions.size(), 1);
    assertEquals(decisions.seat(0), seat);
    for (int index = 0; index < decisions.legalCount(0); index++) {
      Action action = Action.of(decisions.legal(0, index));
      if (action.type() == type && (tile < 0 || action.tile() == tile)) {
        assertEquals(engine.commit(new int[] {index}, ws), Boundary.DECISIONS);
        return;
      }
    }
    throw new AssertionError("no " + type + " of tile " + tile + " for seat " + seat);
  }

  private int category(Tensor tensor, int element, int column) {
    int at = r.batch.plan().offset(tensor) + element * InputSchema.FORMAT.width(tensor) + column;
    return r.batch.blob().get(at) & 0xFF;
  }
}
