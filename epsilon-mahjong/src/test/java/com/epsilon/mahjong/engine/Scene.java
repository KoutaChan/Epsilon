package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.action.Meld.AkaSource;
import com.epsilon.mahjong.action.Meld.RelativeSource;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;

/**
 * 手で組み立てる、四人打ちのテスト用の局面（東 1 局、親は席 0）。持つ部品は Engine と同じで、牌の表記は Scenes と同じ（0 は赤の 5）。
 * 物理牌はまだ使っていないものから選び、赤と書かない牌には赤でない牌を使う。面子は純手牌より先に作る（面子を 3 枚と数えて、手牌は
 * 14 枚まで）。牌山は全ての位置を伏せたままにし、置けるのは裏ドラ表示牌だけ。
 */
final class Scene {

  static final Variant VARIANT = Variant.TENHOU_4P;

  final Wall wall = new Wall(VARIANT);
  final PublicState state = new PublicState(VARIANT);
  final Hand[] hands = new Hand[VARIANT.seats()];
  final ChanceLog chance = new ChanceLog(VARIANT);
  final Decisions decisions = new Decisions(VARIANT);
  final Resolution resolution = new Resolution(VARIANT);
  final Window window = new Window(VARIANT);
  final RuleWorkspace ws = new RuleWorkspace(VARIANT);
  final FullReader full;

  private final TileSet set = VARIANT.tiles();
  private final int[] wallTiles = new int[set.size()];
  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];
  private long nextId = 1;

  Scene(RoundStart start) {
    for (int seat = 0; seat < hands.length; seat++) {
      hands[seat] = new Hand(VARIANT);
    }
    TableState table = new TableState(VARIANT);
    table.set(start);
    state.start(table);
    Arrays.fill(wallTiles, Wall.UNKNOWN);
    wall.load(wallTiles);
    full = new FullReader(state, hands, decisions, chance);
  }

  /** 全員の持ち点が配給原点の局面。 */
  static Scene first() {
    return new Scene(RoundStart.first(VARIANT));
  }

  /** 持ち点を席の順に指定した局面（合計は配給原点の 4 倍）。 */
  static Scene withScores(int... scores) {
    return new Scene(new RoundStart(VARIANT, 0, 0, 0, scores));
  }

  long nextId() {
    return nextId;
  }

  /** 席の純手牌に牌を足す。 */
  Scene hand(int seat, String tiles) {
    for (Face face : Scenes.faces(tiles)) {
      hands[seat].add(take(face));
    }
    return this;
  }

  /**
   * ほかの席が打った牌をポンして、その面子を足す（手牌から使う 2 枚は赤でない牌）。打牌とポンは公開のイベントに記録する。
   *
   * @param seat ポンする席
   * @param from 牌を打った席
   * @param tile ポンする牌の表記
   */
  Scene pon(int seat, int from, String tile) {
    int called = take(Scenes.faces(tile)[0]);
    int kind = Tiles.kind(called);
    take(Face.of(kind, false));
    take(Face.of(kind, false));
    state.discard(from, called, false, false);
    AkaSource aka = set.red(called) ? AkaSource.CALLED_TILE : AkaSource.NONE;
    int meld = Meld.pon(kind, RelativeSource.of(seat, from, hands.length), aka).id();
    state.call(seat, meld);
    hands[seat].addMeld(meld);
    return this;
  }

  /** 席の暗槓を足す（同じ種類の 4 枚。5 なら赤を含む）。 */
  Scene ankan(int seat, String tile) {
    int kind = Scenes.faces(tile)[0].kind();
    boolean red = false;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(kind, copy);
      mark(physicalId);
      red |= set.red(physicalId);
    }
    int meld = Meld.ankan(kind, red).id();
    hands[seat].addMeld(meld);
    state.ankan(seat, meld);
    return this;
  }

  /** 席が、純手牌にあるポンの 4 枚目で加槓する。 */
  Scene kakan(int seat, String tile) {
    Face face = Scenes.faces(tile)[0];
    int kind = face.kind();
    Hand hand = hands[seat];
    int added = hand.pick(kind, face.red());
    hand.take(added);
    state.kakan(seat, hand.upgrade(kind, set.red(added)));
    return this;
  }

  /**
   * 直前の打牌を、席が純手牌の 2 枚でポンする。
   *
   * @param seat ポンする席
   * @param useRed 手牌の赤を使うか
   */
  Scene ponDiscard(int seat, boolean useRed) {
    TurnEvent discard = state.last();
    int kind = discard.tile();
    Hand hand = hands[seat];
    hand.take(hand.pick(kind, useRed));
    hand.take(hand.pick(kind, false));
    AkaSource aka = discard.red() ? AkaSource.CALLED_TILE : useRed ? AkaSource.CONSUMED_HAND_TILE : AkaSource.NONE;
    int meld = Meld.pon(kind, RelativeSource.of(seat, discard.seat(), hands.length), aka).id();
    hand.addMeld(meld);
    state.call(seat, meld);
    return this;
  }

  /**
   * 上家の直前の打牌を、席が純手牌の 2 枚でチーする（赤でない牌を先に使う）。
   *
   * @param seat チーする席
   * @param run できる順子の表記
   */
  Scene chi(int seat, String run) {
    TurnEvent discard = state.last();
    int called = discard.tile();
    int base = Scenes.faces(run)[0].kind();
    Hand hand = hands[seat];
    boolean handRed = false;
    for (int kind = base; kind < base + Meld.Type.CHI.tiles(); kind++) {
      if (kind != called) {
        boolean red = Tiles.contains(hand.redFives(), kind);
        int physicalId = hand.pick(kind, red && hand.count(kind) == 1);
        handRed |= set.red(physicalId);
        hand.take(physicalId);
      }
    }
    AkaSource aka = discard.red() ? AkaSource.CALLED_TILE : handRed ? AkaSource.CONSUMED_HAND_TILE : AkaSource.NONE;
    int meld = Meld.chi(base, called, aka).id();
    hand.addMeld(meld);
    state.call(seat, meld);
    return this;
  }

  /** 席の河に牌を打つ（手牌は変えない）。 */
  Scene river(int seat, String tiles) {
    for (Face face : Scenes.faces(tiles)) {
      state.discard(seat, take(face), false, false);
    }
    return this;
  }

  /** 席が立直を宣言して牌を打つ（手牌は変えない）。 */
  Scene riichiDiscard(int seat, String tile) {
    state.discard(seat, take(Scenes.faces(tile)[0]), false, true);
    return this;
  }

  Scene draw(int seat, String tile) {
    return draw(seat, tile, false);
  }

  /** 嶺上のツモ。 */
  Scene rinshan(int seat, String tile) {
    return draw(seat, tile, true);
  }

  /** 牌を見ずに、山を指定した枚数だけ減らす。 */
  Scene burn(int draws) {
    for (int draw = 0; draw < draws; draw++) {
      state.draw(0, TurnEvent.NONE, false);
    }
    return this;
  }

  Scene indicator(String tile) {
    state.reveal(take(Scenes.faces(tile)[0]));
    return this;
  }

  /**
   * 裏ドラ表示牌を牌山に置く。i 番目の表示牌に対応する裏ドラ表示牌は、牌山の長さを L として位置 L − 5 − 2i にある。
   *
   * @param index 何番目の表示牌に対応する裏ドラ表示牌か（0 始まり）
   * @param tile 置く牌の表記
   */
  Scene ura(int index, String tile) {
    wallTiles[Wall.uraPosition(VARIANT, index)] = take(Scenes.faces(tile)[0]);
    wall.load(wallTiles);
    return this;
  }

  /** 宣言牌を打たずに立直を成立させる。 */
  Scene riichi(int seat) {
    state.riichi(seat, false);
    return this;
  }

  /** 直前にツモった席の、手番の判断を作る。 */
  Scene turn() {
    nextId = Legal.turn(state, hands[state.last().seat()], wall, decisions, nextId, ws);
    return this;
  }

  /** 直前の打牌か加槓への、他家の応答を求める。 */
  Scene respond() {
    nextId = window.open(state, hands, wall, decisions, nextId, ws, true);
    return this;
  }

  /** 直前のチー・ポンの後の、打牌の判断を作る。 */
  Scene postCall(int seat) {
    nextId = Legal.postCall(hands[seat], seat, decisions, nextId);
    return this;
  }

  /** 席の合法手の名前（Action の文字列）を返す。その席の判断が無ければ空。 */
  List<String> legal(int seat) {
    List<String> names = new ArrayList<>();
    int slot = decisions.slotOf(seat);
    for (int index = 0; slot >= 0 && index < decisions.legalCount(slot); index++) {
      names.add(Action.of(decisions.legal(slot, index)).toString());
    }
    return names;
  }

  /** 判断の順に、合法手の名前で手を選んで他家の応答を締める。 */
  Window.Outcome close(String... choices) {
    return window.close(state, hands, decisions, choices(choices), resolution);
  }

  /** 判断の順に並べた合法手の名前を、合法手の何番目かに直す。 */
  int[] choices(String... names) {
    Assert.assertEquals(names.length, decisions.size(), "one choice per slot");
    int[] choices = new int[names.length];
    for (int slot = 0; slot < names.length; slot++) {
      choices[slot] = legal(decisions.seat(slot)).indexOf(names[slot]);
      Assert.assertTrue(choices[slot] >= 0, names[slot] + " is not legal for seat " + decisions.seat(slot));
    }
    return choices;
  }

  void copyFrom(Scene other) {
    wall.copyFrom(other.wall);
    state.copyFrom(other.state);
    for (int seat = 0; seat < hands.length; seat++) {
      hands[seat].copyFrom(other.hands[seat]);
    }
    chance.copyFrom(other.chance);
    decisions.copyFrom(other.decisions);
    resolution.copyFrom(other.resolution);
    window.copyFrom(other.window);
    nextId = other.nextId;
  }

  private Scene draw(int seat, String tile, boolean rinshan) {
    int physicalId = take(Scenes.faces(tile)[0]);
    hands[seat].add(physicalId);
    state.draw(seat, physicalId, rinshan);
    return this;
  }

  /** 表記の 1 枚に当たる、まだ使っていない物理牌を取る。 */
  private int take(Face face) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(face.kind(), copy);
      if (!used[physicalId] && set.red(physicalId) == face.red()) {
        return mark(physicalId);
      }
    }
    throw new IllegalStateException("No free copy of " + face.mjai());
  }

  private int mark(int physicalId) {
    if (used[physicalId]) {
      throw new IllegalStateException("Tile " + physicalId + " is used twice");
    }
    used[physicalId] = true;
    return physicalId;
  }
}
