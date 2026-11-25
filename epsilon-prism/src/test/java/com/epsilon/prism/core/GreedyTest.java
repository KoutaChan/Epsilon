package com.epsilon.prism.core;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.input.WinFacts;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 貪欲の比べ方（{@link Greedy}）: s の大きい手、同点なら p0 の大きい手、それも同じなら番号の小さい手が上。s が NaN の手は、どの位置に
 * あっても NaN でない手より下になり、NaN どうしは同点として p0 と番号で決まる。
 */
public class GreedyTest {
  private static final float NAN = Float.NaN;

  @Test
  public void tiesGoToThePriorThenTheIndex() {
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, null, 3), 1);
    Assert.assertEquals(
        Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, new float[] {0.5f, 0.2f, 0.3f}, 3), 2);
    Assert.assertEquals(
        Greedy.argmax(new float[] {0.1f, 0.3f, 0.3f}, new float[] {0.5f, 0.25f, 0.25f}, 3), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, 0.3f, 0.9f}, null, 2), 1);
  }

  /** NaN が先頭にあっても選ばない。 */
  @Test
  public void nanIsBelowEveryNumberAtAnyPosition() {
    Assert.assertEquals(Greedy.argmax(new float[] {NAN, 0.3f, 0.1f}, null, 3), 1);
    Assert.assertEquals(Greedy.argmax(new float[] {0.1f, NAN, 0.3f}, null, 3), 2);
    Assert.assertEquals(Greedy.argmax(new float[] {0.3f, 0.1f, NAN}, null, 3), 0);
    Assert.assertEquals(Greedy.argmax(new float[] {NAN, Float.NEGATIVE_INFINITY}, null, 2), 1);
    Assert.assertEquals(
        Greedy.argmax(new float[] {NAN, 0.3f, 0.3f}, new float[] {0.6f, 0.1f, 0.3f}, 3), 2);
    float[] scores = {NAN, 0.3f};
    Assert.assertTrue(Greedy.before(1, 0, scores, null));
    Assert.assertFalse(Greedy.before(0, 1, scores, null));
  }

  /** NaN どうしは同点。全ての手が NaN なら、p0 の大きい手、それも同じなら番号の小さい手。 */
  @Test
  public void nanTiesWithNan() {
    float[] scores = {NAN, NAN, NAN};
    Assert.assertEquals(Greedy.argmax(scores, null, 3), 0);
    Assert.assertEquals(Greedy.argmax(scores, new float[] {0.2f, 0.5f, 0.3f}, 3), 1);
    Assert.assertTrue(Greedy.before(0, 1, scores, null));
    Assert.assertFalse(Greedy.before(1, 0, scores, null));
  }

  /** 挿入ソートの順位は、NaN がどこにあっても同じ（NaN でない手の降順の後に NaN の手）。 */
  @Test
  public void insertionSortDoesNotDependOnWhereNanIs() {
    Assert.assertEquals(sort(new float[] {0.1f, NAN, 0.3f}), new int[] {2, 0, 1});
    Assert.assertEquals(sort(new float[] {NAN, 0.1f, 0.3f}), new int[] {2, 1, 0});
    Assert.assertEquals(sort(new float[] {0.3f, NAN, NAN, 0.5f}), new int[] {3, 0, 1, 2});
  }

  private static int[] sort(float[] scores) {
    int[] order = new int[scores.length];
    for (int i = 0; i < scores.length; i++) {
      int j = i - 1;
      while (j >= 0 && Greedy.before(i, order[j], scores, null)) {
        order[j + 1] = order[j];
        j--;
      }
      order[j + 1] = i;
    }
    return order;
  }

  /**
   * 証明の判定（{@link AgariOrder#certify}）: ツモ・ロンの首位確定・最下位確定・中位・非終局、裏ドラの高速 2 経路と全列挙の一致、
   * 西入の反例（裏 0 枚では終局するが裏 1 枚でトップが返しを下回り西入する）、四人打ちの槍槓、三人打ちの北抜きロンを確かめる。
   */
  @Test
  public void agariOrderCertificationMatchesRulesAndFullUraEnumeration() {
    AgariOrder four = new AgariOrder(Variant.TENHOU_4P);
    AgariOrder three = new AgariOrder(Variant.TENHOU_3P);

    // 1. 南 4 局（親=3）の立直ツモ首位確定（高速首位経路と全列挙の一致）
    Observation topTsumo =
        southFourTsumo(new int[] {40_000, 20_000, 20_000, 20_000}, "123m456m789m11z56p", true);
    AgariOrder.Certification fastTop = four.certify(topTsumo, true);
    AgariOrder.Certification fullTop = four.certify(topTsumo, false);
    Assert.assertEquals(fastTop, fullTop);
    Assert.assertEquals(four.certifyEdge(topTsumo), fastTop);
    Assert.assertEquals(fastTop.kind(), AgariOrder.Kind.CERT_TOP);
    Assert.assertTrue(fastTop.tsumo());
    Assert.assertEquals(fastTop.bestRank(), 0);
    Assert.assertEquals(fastTop.worstRank(), 0);

    // 2. 南 4 局（親=3）の立直ツモ最下位確定（同時和了が無いので順序辺あり、全列挙と一致）
    Observation bottomTsumo =
        southFourTsumo(new int[] {1_000, 39_000, 30_000, 30_000}, "123m456m789m11z56p", true);
    AgariOrder.Certification fastBottom = four.certify(bottomTsumo, true);
    AgariOrder.Certification fullBottom = four.certify(bottomTsumo, false);
    Assert.assertEquals(fastBottom, fullBottom);
    Assert.assertEquals(four.certifyEdge(bottomTsumo), fastBottom);
    Assert.assertEquals(fastBottom.kind(), AgariOrder.Kind.CERT_BOTTOM);
    Assert.assertTrue(fastBottom.hasEdge());
    Assert.assertTrue(fastBottom.tsumo());
    Assert.assertEquals(fastBottom.bestRank(), 3);
    Assert.assertEquals(fastBottom.worstRank(), 3);

    // 3. 南 4 局（親=3）のダマツモ中位確定（順序辺なし）
    Observation midTsumo =
        southFourTsumo(new int[] {25_000, 40_000, 15_000, 20_000}, "123m456m789m11z56p", false);
    AgariOrder.Certification mid = four.certify(midTsumo);
    Assert.assertEquals(mid.kind(), AgariOrder.Kind.CERT_MID);
    Assert.assertFalse(mid.hasEdge());
    Assert.assertEquals(four.certifyEdge(midTsumo), AgariOrder.Certification.NONE);
    Assert.assertEquals(mid.bestRank(), 1);
    Assert.assertEquals(mid.worstRank(), 1);

    // 4. 西入の反例: 裏 0 枚（両立直＋ツモの 3 翻 30 符 1000/2000）では親が 30,500 点で終局するが、裏 1 枚（2000/3900）で全員 30,000
    // 点未満になり西入する
    Observation westCounterexample =
        southFourTsumo(new int[] {21_500, 23_000, 23_000, 32_500}, "123m678m789s22z23p", true);
    Assert.assertTrue(
        WinFacts.settlesGame(westCounterexample, 0, new int[4]), "0 ura must finish at South 4");
    Assert.assertEquals(four.certify(westCounterexample, true), AgariOrder.Certification.NONE);
    Assert.assertEquals(four.certify(westCounterexample, false), AgariOrder.Certification.NONE);
    Assert.assertEquals(four.certifyEdge(westCounterexample), AgariOrder.Certification.NONE);

    // 5. 東 1 局の非飛び棄却（高速経路と全列挙の一致）
    Observation eastOneTsumo = eastOneRiichiTsumo();
    Assert.assertEquals(four.certify(eastOneTsumo, true), AgariOrder.Certification.NONE);
    Assert.assertEquals(four.certify(eastOneTsumo, false), AgariOrder.Certification.NONE);
    Assert.assertEquals(four.certifyEdge(eastOneTsumo), AgariOrder.Certification.NONE);

    // 6. 四人打ちの槍槓ロン: 単独最下位でもダブロンがあり得るので順序辺なし（CERT_MID）、首位確定のみ CERT_TOP
    Observation chankanBottom = southFourChankan(new int[] {1_000, 33_000, 20_000, 46_000});
    Assert.assertEquals(chankanBottom.decisionKind(), EngineDecisionKind.CHANKAN);
    AgariOrder.Certification chankanCert = four.certify(chankanBottom, true);
    Assert.assertEquals(chankanCert, four.certify(chankanBottom, false));
    Assert.assertEquals(chankanCert.kind(), AgariOrder.Kind.CERT_MID);
    Assert.assertFalse(chankanCert.hasEdge());
    Assert.assertFalse(chankanCert.tsumo());
    Assert.assertEquals(chankanCert.bestRank(), 3);
    Assert.assertEquals(chankanCert.worstRank(), 3);
    Assert.assertEquals(four.certifyEdge(chankanBottom), AgariOrder.Certification.NONE);
    float[] chankanScores = new float[chankanBottom.legalActions()];
    for (int a = 0; a < chankanScores.length; a++) {
      chankanScores[a] = chankanBottom.action(a).type() == Action.Type.RON_AGARI ? 0.9f : -0.5f;
    }
    Assert.assertFalse(four.clamp(chankanBottom, chankanScores));

    Observation chankanTop = southFourChankan(new int[] {40_000, 20_000, 20_000, 20_000});
    AgariOrder.Certification chankanTopCert = four.certify(chankanTop, true);
    Assert.assertEquals(chankanTopCert.kind(), AgariOrder.Kind.CERT_TOP);
    Assert.assertTrue(chankanTopCert.hasEdge());
    Assert.assertEquals(chankanTopCert, four.certify(chankanTop, false));
    Assert.assertEquals(four.certifyEdge(chankanTop), chankanTopCert);

    // 7. 三人打ちの北抜きロン: 単独最下位は順序辺なし（CERT_MID）、首位確定のみ CERT_TOP
    Observation nukiBottom = sanmaNukiRon(new int[] {1_000, 60_000, 44_000});
    Assert.assertEquals(nukiBottom.decisionKind(), EngineDecisionKind.NUKI_RESPONSE);
    AgariOrder.Certification nukiCert = three.certify(nukiBottom, true);
    Assert.assertEquals(nukiCert, three.certify(nukiBottom, false));
    Assert.assertEquals(nukiCert.kind(), AgariOrder.Kind.CERT_MID);
    Assert.assertFalse(nukiCert.hasEdge());
    Assert.assertFalse(nukiCert.tsumo());
    Assert.assertEquals(nukiCert.bestRank(), 2);
    Assert.assertEquals(nukiCert.worstRank(), 2);
    Assert.assertEquals(three.certifyEdge(nukiBottom), AgariOrder.Certification.NONE);

    Observation nukiTop = sanmaNukiRon(new int[] {60_000, 25_000, 20_000});
    AgariOrder.Certification nukiTopCert = three.certify(nukiTop, true);
    Assert.assertEquals(nukiTopCert.kind(), AgariOrder.Kind.CERT_TOP);
    Assert.assertTrue(nukiTopCert.hasEdge());
    Assert.assertEquals(nukiTopCert, three.certify(nukiTop, false));
    Assert.assertEquals(three.certifyEdge(nukiTop), nukiTopCert);
  }

  /** クランプ（{@link AgariOrder#clamp}）: FP32 の厳密な大小（同点の引き離し込み）と、証明辺の無い行のビット同一を確かめる。 */
  @Test
  public void agariOrderClampEnforcesStrictFp32AndLeavesNonEdgeRowsBitIdentical() {
    float[] topTie = {0.5f, 0.5f, -0.2f};
    Assert.assertTrue(AgariOrder.clamp(topTie, 1, true));
    Assert.assertEquals(
        Float.floatToRawIntBits(topTie[1]), Float.floatToRawIntBits(Math.nextUp(0.5f)));
    Assert.assertTrue(topTie[1] > topTie[0] && topTie[1] > topTie[2]);
    Assert.assertFalse(AgariOrder.clamp(topTie, 1, true));

    float[] bottomTie = {0.2f, -0.5f, -0.5f};
    Assert.assertTrue(AgariOrder.clamp(bottomTie, 2, false));
    Assert.assertEquals(
        Float.floatToRawIntBits(bottomTie[2]), Float.floatToRawIntBits(Math.nextDown(-0.5f)));
    Assert.assertTrue(bottomTie[2] < bottomTie[0] && bottomTie[2] < bottomTie[1]);
    Assert.assertFalse(AgariOrder.clamp(bottomTie, 2, false));

    Assert.assertThrows(
        IllegalStateException.class, () -> AgariOrder.clamp(new float[] {Float.NaN, 0f}, 0, true));
    Assert.assertThrows(
        IllegalStateException.class,
        () -> AgariOrder.clamp(new float[] {Float.POSITIVE_INFINITY, 0f}, 1, false));

    AgariOrder four = new AgariOrder(Variant.TENHOU_4P);
    Observation mid =
        southFourTsumo(new int[] {25_000, 40_000, 15_000, 20_000}, "123m456m789m11z56p", false);
    float[] scores = new float[mid.legalActions()];
    for (int a = 0; a < scores.length; a++) {
      scores[a] = a == 0 ? -0.0f : (a - 3) * 0.25f;
    }
    int[] beforeBits = new int[scores.length];
    for (int a = 0; a < scores.length; a++) {
      beforeBits[a] = Float.floatToRawIntBits(scores[a]);
    }
    Assert.assertFalse(four.clamp(mid, scores));
    for (int a = 0; a < scores.length; a++) {
      Assert.assertEquals(Float.floatToRawIntBits(scores[a]), beforeBits[a]);
    }
  }

  private static Observation southFourTsumo(int[] scores, String seat0Hand, boolean riichi) {
    Engine engine = new Engine(Variant.TENHOU_4P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_4P);
    Scenes wall =
        new Scenes(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, scores))
            .hand(0, seat0Hand)
            .hand(1, "19m258p369s147s5z6z")
            .hand(2, "19m369p258s147p6z7z")
            .hand(3, "19m258p369s258s3z7z")
            .indicator("9s")
            .draws(riichi ? "8s8p7p3z1z8p7p3z1z4p" : "8s8p7p3z1z4p");
    wall.start(engine, ws);
    stepTo(engine, ws, 3, EngineDecisionKind.TURN);
    play(engine, ws, "3:DAHAI_TSUMOGIRI(8s)");
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    if (riichi) {
      play(engine, ws, "0:RIICHI_TSUMOGIRI(8p)");
    } else {
      play(engine, ws, "0:DAHAI_TSUMOGIRI(8p)");
    }
    passUntilTsumo(engine, ws, 0);
    return capture(Variant.TENHOU_4P, engine, 0);
  }

  private static Observation eastOneRiichiTsumo() {
    Engine engine = new Engine(Variant.TENHOU_4P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_4P);
    Scenes wall =
        new Scenes(RoundStart.first(Variant.TENHOU_4P))
            .hand(0, "123m456m789m22z56p")
            .hand(1, "19m258p369s147s5z6z")
            .hand(2, "19m369p258s147p6z7z")
            .hand(3, "19m258p369s258s3z7z")
            .indicator("9s")
            .draws("8p7p3z1z8p7p3z1z4p");
    wall.start(engine, ws);
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    play(engine, ws, "0:RIICHI_TSUMOGIRI(8p)");
    passUntilTsumo(engine, ws, 0);
    return capture(Variant.TENHOU_4P, engine, 0);
  }

  private static Observation southFourChankan(int[] scores) {
    Engine engine = new Engine(Variant.TENHOU_4P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_4P);
    Scenes wall =
        new Scenes(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, scores))
            .hand(0, "123m456m789m11z5p8s")
            .hand(1, "19m258p369s147s5z6z")
            .hand(2, "19m369p258s17p3s6z7z")
            .hand(3, "44p123s456s789s1z2z")
            .indicator("7z")
            .draws("9s9p4p6p8p9p4p");
    wall.start(engine, ws);
    stepTo(engine, ws, 3, EngineDecisionKind.TURN);
    play(engine, ws, "3:DAHAI_TSUMOGIRI(9s)");
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    play(engine, ws, "0:DAHAI_TSUMOGIRI(9p)");
    stepTo(engine, ws, 1, EngineDecisionKind.TURN);
    play(engine, ws, "1:DAHAI_TSUMOGIRI(4p)");
    stepTo(engine, ws, 3, EngineDecisionKind.RESPONSE);
    play(engine, ws, "3:PON(4p)");
    stepTo(engine, ws, 3, EngineDecisionKind.POST_CALL_DAHAI);
    play(engine, ws, "3:DAHAI(S)");
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    play(engine, ws, "0:RIICHI_DAHAI(8s)");
    stepTo(engine, ws, 3, EngineDecisionKind.TURN);
    play(engine, ws, "3:KAKAN(4p)");
    return capture(Variant.TENHOU_4P, engine, 0);
  }

  private static Observation sanmaNukiRon(int[] scores) {
    Engine engine = new Engine(Variant.TENHOU_3P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_3P);
    Scenes wall =
        new Scenes(new RoundStart(Variant.TENHOU_3P, 5, 0, 0, scores))
            .hand(0, "123p456p789s11z44z")
            .hand(1, "19m369p258s147s3z7z")
            .hand(2, "19m258p369s147s5z6z")
            .indicator("1m")
            .draws("2z4z8p4z")
            .rinshan("2z");
    wall.start(engine, ws);
    stepTo(engine, ws, 2, EngineDecisionKind.TURN);
    play(engine, ws, "2:DAHAI_TSUMOGIRI(S)");
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    play(engine, ws, "0:NUKI");
    stepTo(engine, ws, 0, EngineDecisionKind.TURN);
    play(engine, ws, "0:RIICHI_TSUMOGIRI(S)");
    stepTo(engine, ws, 2, EngineDecisionKind.TURN);
    play(engine, ws, "2:NUKI");
    return capture(Variant.TENHOU_3P, engine, 0);
  }

  private static void stepTo(Engine engine, RuleWorkspace ws, int seat, EngineDecisionKind kind) {
    for (int guard = 0; guard < 32; guard++) {
      Decisions decisions = engine.decisions();
      int slot = decisions.slotOf(seat);
      if (slot >= 0 && decisions.kind(slot) == kind) {
        return;
      }
      passOrTsumogiri(engine, ws);
    }
    throw new AssertionError("never reached " + kind + " for seat " + seat);
  }

  private static void passUntilTsumo(Engine engine, RuleWorkspace ws, int seat) {
    for (int guard = 0; guard < 32; guard++) {
      Decisions decisions = engine.decisions();
      int slot = decisions.slotOf(seat);
      if (slot >= 0 && decisions.kind(slot) == EngineDecisionKind.TURN) {
        for (int a = 0; a < decisions.legalCount(slot); a++) {
          if (Action.of(decisions.legal(slot, a)).type() == Action.Type.TSUMO_AGARI) {
            return;
          }
        }
      }
      passOrTsumogiri(engine, ws);
    }
    throw new AssertionError("never reached TSUMO_AGARI for seat " + seat);
  }

  private static Observation capture(Variant variant, Engine engine, int seat) {
    Decisions decisions = engine.decisions();
    int slot = decisions.slotOf(seat);
    Assert.assertTrue(slot >= 0, "seat " + seat + " has no decision");
    ObservationWriter writer = new ObservationWriter(variant);
    ObservationBatch batch = new ObservationBatch(variant, 1 << 12);
    int index = writer.capture(engine.view(seat), decisions, slot, batch);
    return new Observation(variant).bind(batch, index);
  }

  private static void play(Engine engine, RuleWorkspace ws, String... picks) {
    Decisions decisions = engine.decisions();
    int[] choices = new int[decisions.size()];
    for (int slot = 0; slot < decisions.size(); slot++) {
      int seat = decisions.seat(slot);
      String name = Action.Type.PASS.name();
      for (String pick : picks) {
        if (pick.startsWith(seat + ":")) {
          name = pick.substring(2);
        }
      }
      int found = -1;
      for (int a = 0; a < decisions.legalCount(slot); a++) {
        if (Action.of(decisions.legal(slot, a)).toString().equals(name)) {
          found = a;
          break;
        }
      }
      Assert.assertTrue(found >= 0, name + " is not legal for seat " + seat);
      choices[slot] = found;
    }
    Assert.assertEquals(engine.commit(choices, ws), Boundary.DECISIONS);
  }

  private static void passOrTsumogiri(Engine engine, RuleWorkspace ws) {
    Decisions decisions = engine.decisions();
    int[] choices = new int[decisions.size()];
    for (int slot = 0; slot < decisions.size(); slot++) {
      int chosen = 0;
      for (int a = 0; a < decisions.legalCount(slot); a++) {
        Action action = Action.of(decisions.legal(slot, a));
        if (action.type() == Action.Type.PASS
            || (action.type() == Action.Type.DAHAI && action.selection().tsumogiri())) {
          chosen = a;
          break;
        }
      }
      choices[slot] = chosen;
    }
    Assert.assertEquals(engine.commit(choices, ws), Boundary.DECISIONS);
  }
}
