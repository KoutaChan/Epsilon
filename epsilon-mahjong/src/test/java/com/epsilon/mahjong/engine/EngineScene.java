package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import org.testng.Assert;

/**
 * 手で組んだ牌山でエンジンに 1 局を打たせる、テスト用のヘルパー。牌山は Scenes で組み、指定しなかった位置は既定の乱数の種で
 * 並べた牌で埋めて、牌譜の局として始める。牌の表記は Scenes と同じ（0 は赤の 5。赤と書かない牌には赤でない物理牌を使う）。
 *
 * <p>選ぶ手は「席:手の名前」（Action の文字列）で書く。名前を書かない席は見送りを選ぶ。
 */
final class EngineScene {

  final Variant variant;
  final Engine engine;
  final RuleWorkspace ws;

  private final Scenes wall;
  private Boundary boundary;

  EngineScene(RoundStart start) {
    wall = new Scenes(start);
    variant = start.variant();
    engine = new Engine(variant);
    ws = new RuleWorkspace(variant);
  }

  /** 東 1 局から、全員が配給原点で始める。 */
  static EngineScene first(Variant variant) {
    return new EngineScene(RoundStart.first(variant));
  }

  static EngineScene at(Variant variant, int round, int honba, int kyotaku, int... scores) {
    return new EngineScene(new RoundStart(variant, round, honba, kyotaku, scores));
  }

  /** 席の配牌 13 枚を置く。 */
  EngineScene hand(int seat, String tiles) {
    wall.hand(seat, tiles);
    return this;
  }

  /** 局の最初から、山のツモを局全体のツモの順に置く（鳴きがあれば、その分だけ手番の順とずれる）。 */
  EngineScene draws(String tiles) {
    wall.draws(tiles);
    return this;
  }

  /**
   * 途中からの山のツモを、局全体のツモの順に置く。
   *
   * @param k 最初の牌を置くツモの番号（0 始まり）
   * @param tiles 続けて置く牌の表記
   */
  EngineScene drawAt(int k, String tiles) {
    wall.drawAt(k, tiles);
    return this;
  }

  /** 嶺上ツモ（北を抜いた後の補充も含む）を、1 回目から順に置く。 */
  EngineScene rinshan(String tiles) {
    wall.rinshan(tiles);
    return this;
  }

  /** 最初のドラ表示牌を置く。 */
  EngineScene indicator(String tile) {
    wall.indicator(tile);
    return this;
  }

  /**
   * ドラ表示牌を置く。
   *
   * @param index 何番目のドラ表示牌か（0 始まり。1 から先は槓ドラの表示牌）
   * @param tile 置く牌の表記
   */
  EngineScene indicator(int index, String tile) {
    wall.indicator(index, tile);
    return this;
  }

  /** 残りの位置を埋めて局を始め、親の最初の手番まで進める。 */
  EngineScene start() {
    boundary = wall.start(engine, ws);
    return this;
  }

  Boundary boundary() {
    return boundary;
  }

  boolean ended() {
    return boundary != Boundary.DECISIONS;
  }

  /** 今の判断が、その席がツモった後の手番の判断か。 */
  boolean turnOf(int seat) {
    return boundary == Boundary.DECISIONS
        && engine.decisions().kind(0) == EngineDecisionKind.TURN
        && engine.decisions().seat(0) == seat;
  }

  int liveLeft() {
    return engine.full().seat(0).liveLeft();
  }

  SeatView view(int seat) {
    return engine.view(seat);
  }

  RoundEnd end() {
    return engine.roundEnd();
  }

  /** その席の合法手の名前。その席の判断が無ければ空。 */
  List<String> legal(int seat) {
    List<String> names = new ArrayList<>();
    Decisions decisions = engine.decisions();
    int slot = decisions.slotOf(seat);
    for (int index = 0; slot >= 0 && index < decisions.legalCount(slot); index++) {
      names.add(Action.of(decisions.legal(slot, index)).toString());
    }
    return names;
  }

  /**
   * 判断する席ごとに、「席:名前」で書いた手を選ぶ。名前を書かない席は見送りで答える。書いた手は、どれも今判断する席のものでなければ
   * ならない。
   */
  Boundary play(String... picks) {
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
      choices[slot] = legal(seat).indexOf(name);
      Assert.assertTrue(choices[slot] >= 0, name + " is not legal for seat " + seat + ": " + legal(seat));
    }
    for (int pick = 0; pick < picks.length; pick++) {
      Assert.assertTrue(matched[pick], picks[pick] + " has no decision at this point");
    }
    boundary = engine.commit(choices, ws);
    return boundary;
  }

  /** 手番ではツモ切りし、応答は全て見送って進める。止める条件が真になった判断の時点で止まる。 */
  EngineScene passUntil(Predicate<EngineScene> stop) {
    while (!stop.test(this)) {
      Assert.assertEquals(boundary, Boundary.DECISIONS, "the round ended before the stop");
      Decisions decisions = engine.decisions();
      int[] choices = new int[decisions.size()];
      for (int slot = 0; slot < decisions.size(); slot++) {
        boolean turn = decisions.kind(slot) == EngineDecisionKind.TURN;
        choices[slot] = turn ? tsumogiri(decisions, slot) : pass(decisions, slot);
      }
      boundary = engine.commit(choices, ws);
    }
    return this;
  }

  private static int tsumogiri(Decisions decisions, int slot) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (action.type() == Action.Type.DAHAI && action.selection().tsumogiri()) {
        return index;
      }
    }
    throw new AssertionError("No tsumogiri for seat " + decisions.seat(slot));
  }

  private static int pass(Decisions decisions, int slot) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      if (Action.of(decisions.legal(slot, index)).type() == Action.Type.PASS) {
        return index;
      }
    }
    throw new AssertionError("No pass for seat " + decisions.seat(slot));
  }
}
