package com.epsilon.prism.luck;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import org.testng.Assert;

/**
 * 裏ドラの期待値のテストで使う場面。手で作った牌山で四人打ちの東 1 局を、席 2 が立直して和了するまで打つ。
 *
 * <p>席 2 の手は 45p111m789s456m99s で、3p・6p 待ち。ドラ表示牌は 1s（ドラの 2s は誰も持たない）で、1 枚目の裏ドラ表示牌は 4m
 * （裏ドラの 5m を席 2 が 1 枚持つ）。親の席 0 が 2p と 3p をポンして、加槓する場面もある。
 */
final class UraScenes {

  static final Variant VARIANT = Variant.TENHOU_4P;

  /** 保留のままの槓ドラの表示牌。この種類の牌は、牌山の外では誰の手にも無い。 */
  static final String PENDING_INDICATOR = "9p";

  private UraScenes() {}

  /** 局末まで打ち終えたエンジンと、局の始めの卓の状態。 */
  record Played(Engine engine, TableState start) {}

  /** 席 2 の立直の後、一発の巡目に席 1 が打った 3p を席 2 がロンする。槓は無く、表示牌は 1 枚。 */
  static Played plainRon() {
    Scene scene = deal("9m6z6z7z8m3p", null, false);
    scene.play("0:DAHAI(N)");
    scene.play("1:DAHAI_TSUMOGIRI(F)");
    scene.play("2:RIICHI_TSUMOGIRI(F)");
    scene.play("3:DAHAI_TSUMOGIRI(C)");
    scene.play("0:DAHAI_TSUMOGIRI(8m)");
    scene.play("1:DAHAI_TSUMOGIRI(3p)");
    return scene.finish("2:RON_AGARI");
  }

  /** 席 0 が 3p を加槓し、立直している席 2 が槍槓でロンする。保留の表示牌は無く、表示牌は 1 枚。 */
  static Played chankan() {
    Scene scene = deal("9m2p3p6z6z7z3p", null, false);
    ponTwice(scene);
    scene.play("0:KAKAN(3p)");
    return scene.finish("2:RON_AGARI");
  }

  /**
   * 席 0 が 2p を加槓して嶺上牌の 3p を引き、続けて 3p を加槓する。立直している席 2 が槍槓でロンする。2p の槓ドラは打牌の前なので
   * 保留のままで、めくってある表示牌は 1 枚。
   */
  static Played chankanWithPendingIndicator() {
    Scene scene = deal("9m2p3p6z6z7z2p", "3p", true);
    ponTwice(scene);
    scene.play("0:KAKAN(2p)");
    scene.play("0:KAKAN(3p)");
    return scene.finish("2:RON_AGARI");
  }

  /** 席 0 が席 1 の 2p と 3p をポンし、席 2 が立直する。この後は、席 0 がツモした後の手番になる。 */
  private static void ponTwice(Scene scene) {
    scene.play("0:DAHAI(N)");
    scene.play("1:DAHAI_TSUMOGIRI(2p)");
    scene.play("0:PON(2p)");
    scene.play("0:DAHAI(E)");
    scene.play("1:DAHAI_TSUMOGIRI(3p)");
    scene.play("0:PON(3p)");
    scene.play("0:DAHAI(S)");
    scene.play("1:DAHAI_TSUMOGIRI(F)");
    scene.play("2:RIICHI_TSUMOGIRI(F)");
    scene.play("3:DAHAI_TSUMOGIRI(C)");
  }

  private static Scene deal(String draws, String rinshan, boolean pending) {
    Scenes wall =
        Scenes.first(VARIANT)
            .hand(0, "22p33p147m58s1234z")
            .hand(1, "147p258m369s1234z")
            .hand(2, "45p111m789s456m99s")
            .hand(3, "147p368s258m4567z")
            .draws(draws)
            .indicator(0, "1s")
            .ura(0, "4m");
    if (rinshan != null) {
      wall.rinshan(rinshan);
    }
    if (pending) {
      wall.indicator(1, PENDING_INDICATOR);
    }
    return new Scene(wall);
  }

  /** 手で作った牌山の局を打つ。選ぶ手は「席:行動の名前」で書き、書かなかった席は見送る。 */
  private static final class Scene {

    private final Engine engine = new Engine(VARIANT);
    private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
    private Boundary boundary;

    Scene(Scenes wall) {
      boundary = wall.start(engine, ws);
    }

    void play(String... picks) {
      Assert.assertEquals(boundary, Boundary.DECISIONS, "no decision is pending");
      Decisions decisions = engine.decisions();
      int[] choices = new int[decisions.size()];
      boolean[] matched = new boolean[picks.length];
      for (int slot = 0; slot < decisions.size(); slot++) {
        int seat = decisions.seat(slot);
        String name = Action.Type.PASS.name();
        for (int pick = 0; pick < picks.length; pick++) {
          if (picks[pick].startsWith(seat + ":")) {
            name = picks[pick].substring(2);
            matched[pick] = true;
          }
        }
        choices[slot] = -1;
        StringBuilder legal = new StringBuilder();
        for (int index = 0; index < decisions.legalCount(slot); index++) {
          String action = Action.of(decisions.legal(slot, index)).toString();
          legal.append(action).append(' ');
          if (action.equals(name)) {
            choices[slot] = index;
          }
        }
        Assert.assertTrue(choices[slot] >= 0, name + " is not legal for seat " + seat + ": " + legal);
      }
      for (int pick = 0; pick < picks.length; pick++) {
        Assert.assertTrue(matched[pick], picks[pick] + " has no decision at this point");
      }
      boundary = engine.commit(choices, ws);
    }

    Played finish(String... picks) {
      play(picks);
      Assert.assertEquals(boundary, Boundary.ROUND_END);
      return new Played(engine, engine.handStart());
    }
  }
}
