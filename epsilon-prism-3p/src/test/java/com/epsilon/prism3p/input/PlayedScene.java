package com.epsilon.prism3p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.rules.RoundStart;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;
import org.testng.Assert;

/**
 * 手で作った三人打ちの牌山を打つテスト用のヘルパー。牌山は {@link Scenes} で置き、手は「席:手の名前」で選ぶ（名前を書かない席は見送る）。
 * 牌の表記は Scenes と同じ。
 */
final class PlayedScene {

  final Engine engine = new Engine(TENHOU_3P);
  final RuleWorkspace ws = new RuleWorkspace(TENHOU_3P);
  private final Scenes wall;
  private Boundary boundary;

  private PlayedScene(RoundStart start) {
    wall = new Scenes(start);
  }

  /** 東 1 局で、全員が配給原点から始める。 */
  static PlayedScene first() {
    return new PlayedScene(RoundStart.first(TENHOU_3P));
  }

  static PlayedScene at(int round, int honba, int kyotaku, int... scores) {
    return new PlayedScene(new RoundStart(TENHOU_3P, round, honba, kyotaku, scores));
  }

  PlayedScene hand(int seat, String tiles) {
    wall.hand(seat, tiles);
    return this;
  }

  /** 局の最初から山で引くツモを、全員を通した順に置く。 */
  PlayedScene draws(String tiles) {
    wall.draws(tiles);
    return this;
  }

  /** 嶺上ツモと北抜きの補充のツモを、1 回目から順に置く。 */
  PlayedScene rinshan(String tiles) {
    wall.rinshan(tiles);
    return this;
  }

  /** 最初の表示牌を置く。 */
  PlayedScene indicator(String tile) {
    wall.indicator(tile);
    return this;
  }

  /** 残りの位置を埋めて局を始め、親の最初の手番まで進める。 */
  PlayedScene start() {
    boundary = wall.start(engine, ws);
    return this;
  }

  Boundary boundary() {
    return boundary;
  }

  boolean ended() {
    return boundary != Boundary.DECISIONS;
  }

  /** 今の判断の時点が、その席がツモった後の手番かを返す。 */
  boolean turnOf(int seat) {
    return boundary == Boundary.DECISIONS
        && engine.decisions().kind(0) == EngineDecisionKind.TURN
        && engine.decisions().seat(0) == seat;
  }

  /** その席の合法手の名前を返す。その席に判断が無ければ空。 */
  List<String> legal(int seat) {
    List<String> names = new ArrayList<>();
    Decisions decisions = engine.decisions();
    int slot = decisions.slotOf(seat);
    for (int index = 0; slot >= 0 && index < decisions.legalCount(slot); index++) {
      names.add(Action.of(decisions.legal(slot, index)).toString());
    }
    return names;
  }

  /** 判断する席ごとに「席:名前」で書いた手を選ぶ。名前を書かない席は見送る。 */
  Boundary play(String... picks) {
    Assert.assertEquals(boundary, Boundary.DECISIONS, "no decision is pending");
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
      choices[slot] = legal(seat).indexOf(name);
      Assert.assertTrue(choices[slot] >= 0, name + " is not legal for seat " + seat + ": " + legal(seat));
    }
    boundary = engine.commit(choices, ws);
    return boundary;
  }

  /** 手番はツモ切り、応答は全て見送りで打ち、止める条件が真になる判断の時点まで進める。 */
  PlayedScene passUntil(Predicate<PlayedScene> stop) {
    while (!stop.test(this)) {
      Assert.assertEquals(boundary, Boundary.DECISIONS, "the round ended before the stop");
      boundary = passOrTsumogiri(engine, ws);
    }
    return this;
  }

  /** 今の判断の時点を 1 つ進める。手番はツモ切り、応答は全て見送る。 */
  void step() {
    Assert.assertEquals(boundary, Boundary.DECISIONS, "no decision is pending");
    boundary = passOrTsumogiri(engine, ws);
  }

  /** 判断する全ての席で、手番はツモ切り、応答は見送りを打つ。 */
  static Boundary passOrTsumogiri(Engine target, RuleWorkspace workspace) {
    Decisions decisions = target.decisions();
    int[] choices = new int[decisions.size()];
    for (int slot = 0; slot < decisions.size(); slot++) {
      choices[slot] = passOrTsumogiri(decisions, slot);
    }
    return target.commit(choices, workspace);
  }

  /** 手番ならツモ切り、ほかなら見送りの手が、合法手の何番目かを返す。 */
  private static int passOrTsumogiri(Decisions decisions, int slot) {
    boolean turn = decisions.kind(slot) == EngineDecisionKind.TURN;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (turn ? action.type() == Action.Type.DAHAI && action.selection().tsumogiri() : action.type() == Action.Type.PASS) {
        return index;
      }
    }
    throw new AssertionError("No " + (turn ? "tsumogiri" : "pass") + " for seat " + decisions.seat(slot));
  }
}
