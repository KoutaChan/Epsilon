package com.epsilon.classic.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicVocabulary.RonFuritenKind;
import com.epsilon.classic.rulebot.RuleScene;
import com.epsilon.mahjong.action.Action;
import java.util.Map;
import org.testng.annotations.Test;

/**
 * 見逃しの振聴が自分の打牌の後に残るかを、場面を作って確かめる。席 1 は配牌で 5s と 3p のシャンポン待ち（断么九）の聴牌。立直して
 * いない席の見逃しは自分の打牌で消え、下家から鳴いた後の最初の打牌でだけ残る。遷移の入力も、エンジンのこの進み方と同じ値を持つ。
 */
public class MissedRonSceneTest {

  /** 上家の 3p を見送ると、次の手番の打牌で見逃しの振聴が消える。打牌の後は、下家の 5s をロンできる。 */
  @Test
  public void aDiscardAfterADrawClearsTheMissedRon() {
    RuleScene scene = scene("3p1z5s").play("0:DAHAI_TSUMOGIRI(3p)");
    assertRonFuriten(
        scene.observe(1),
        Map.of(
            Action.Type.RON_AGARI, RonFuritenKind.NONE,
            Action.Type.PON, RonFuritenKind.NONE,
            Action.Type.PASS, RonFuritenKind.TEMPORARY));
    Observation turn = scene.play().observe(1);
    assertTrue(turn.furitenTemporary());
    assertFalse(RoundRules.afterCallFromShimocha(turn));
    assertTrue(scene.legal(1).contains("RIICHI_TSUMOGIRI(E)"), scene.legal(1).toString());
    assertRonFuriten(
        turn, Map.of(Action.Type.DAHAI, RonFuritenKind.NONE, Action.Type.RIICHI_DAHAI, RonFuritenKind.NONE));
    scene.play("1:DAHAI_TSUMOGIRI(E)").play("2:DAHAI_TSUMOGIRI(5s)");
    assertTrue(scene.legal(1).contains("RON_AGARI"), scene.legal(1).toString());
  }

  /** 上家の 3p をポンすると、その後の打牌で見逃しの振聴が消える。 */
  @Test
  public void aDiscardAfterAPonFromTheLeftClearsTheMissedRon() {
    RuleScene scene = scene("3p5s").play("0:DAHAI_TSUMOGIRI(3p)").play("1:PON(3p)");
    Observation afterPon = scene.observe(1);
    assertTrue(afterPon.furitenTemporary());
    assertFalse(RoundRules.afterCallFromShimocha(afterPon));
    assertRonFuriten(afterPon, Map.of(Action.Type.DAHAI, RonFuritenKind.NONE));
    scene.play("1:DAHAI(2m)").play("2:DAHAI_TSUMOGIRI(5s)");
    assertFalse(scene.observe(1).furitenTemporary());
  }

  /** 下家の 3p をポンすると、その後の打牌でも見逃しの振聴が残る。 */
  @Test
  public void aDiscardAfterAPonFromTheRightKeepsTheMissedRon() {
    RuleScene scene =
        scene("1z9m3p5s")
            .play("0:DAHAI_TSUMOGIRI(E)")
            .play("1:DAHAI_TSUMOGIRI(9m)")
            .play("2:DAHAI_TSUMOGIRI(3p)");
    assertRonFuriten(
        scene.observe(1),
        Map.of(
            Action.Type.RON_AGARI, RonFuritenKind.NONE,
            Action.Type.PON, RonFuritenKind.TEMPORARY,
            Action.Type.PASS, RonFuritenKind.TEMPORARY));
    Observation afterPon = scene.play("1:PON(3p)").observe(1);
    assertTrue(afterPon.furitenTemporary());
    assertTrue(RoundRules.afterCallFromShimocha(afterPon));
    assertRonFuriten(afterPon, Map.of(Action.Type.DAHAI, RonFuritenKind.TEMPORARY));
    scene.play("1:DAHAI(2m)").play("2:DAHAI_TSUMOGIRI(5s)");
    assertTrue(scene.observe(1).furitenTemporary());
  }

  /**
   * 席 1 が配牌で聴牌している場面を作り、親の最初の手番まで進める。
   *
   * @param draws 局の最初からの山のツモ
   */
  private static RuleScene scene(String draws) {
    return new RuleScene()
        .hand(0, "147m258p369s1235z")
        .hand(1, "234m666m678s55s33p")
        .hand(2, "159m159p19s12345z")
        .hand(3, "258m147p69p2468s9m")
        .draws(draws)
        .indicator("7z")
        .start();
  }

  /**
   * Nano と Major の入力で、全ての合法手の全ての遷移が、手の種別ごとに期待した「遷移した後も残る見逃しの振聴」を持つことを確かめる。
   *
   * @param o 判断の記録
   * @param expected 手の種別ごとの期待する値。判断の合法手の種別を全て含む
   */
  private static void assertRonFuriten(Observation o, Map<Action.Type, RonFuritenKind> expected) {
    for (Architecture architecture : Architecture.values()) {
      ClassicSchema schema = ClassicSchema.of(architecture);
      ClassicInputs in = new ClassicInputs(schema, Buckets.required(o), 1);
      new ClassicTokenizer(architecture, RuleScene.VARIANT).write(o, ClassicExtras.empty(), in, 0);
      FieldLayout<ActionTransitionInt> layout = schema.layout(ActionTransitionInt.class);
      int first =
          in.start(Tensor.TRANSITION_CATEGORIES, 0) + layout.offset(ActionTransitionInt.RESULTING_RON_FURITEN_KIND);
      for (int action = 0; action < o.legalActions(); action++) {
        Action legal = o.action(action);
        for (int transition = 0; transition < Buckets.transitions(o, action); transition++) {
          int index = action * in.bucket().transitions() + transition;
          int value = in.categorical()[first + index * layout.stride()];
          assertEquals(
              RonFuritenKind.values()[value - 1], expected.get(legal.type()), architecture + " " + legal);
        }
      }
    }
  }
}
