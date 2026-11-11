package com.epsilon.classic.rulebot;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.testng.Assert;

/**
 * テスト用の場面。手で並べた牌山で四人打ちの東 1 局（親は席 0）をエンジンに打たせ、判断の時点の記録を作る。配牌、山からのツモ（局の
 * 最初から全員を通して引く順）、最初のドラ表示を置く。残りの位置を決まったシードで並べ替えた牌で埋めてから、牌譜の牌山として局を始める。
 *
 * <p>牌は数字の後に色の文字（m・p・s・z）を付けて書き、0 は赤の 5。0 と書かない 5 には赤でない牌を使う。選ぶ手は "席:手の名前"
 * （{@code Action.toString}）で書き、書かない席は見送る。
 */
public final class RuleScene {

  public static final Variant VARIANT = Variant.TENHOU_4P;

  /** 残りの位置を埋める牌を並べ替えるシード。 */
  private static final long FILL_SEED = 31;

  /** 配牌で 1 度に取る枚数と、その枚数ずつ取る回数。13 枚目は最後に 1 枚だけ取る。 */
  private static final int PACKET = 4;

  private static final int PACKETS = 3;

  /** 最初のドラ表示の位置（牌山の末尾から数える）。 */
  private static final int FIRST_INDICATOR_FROM_END = 6;

  private final Engine engine = new Engine(VARIANT);
  private final RuleWorkspace ws = new RuleWorkspace(VARIANT);
  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation observation = new Observation(VARIANT);
  private final TileSet set = VARIANT.tiles();
  private final int seats = VARIANT.seats();
  private final int[] wall = new int[set.size()];
  private final boolean[] used = new boolean[Tiles.PHYSICAL_IDS];
  private Boundary boundary;

  public RuleScene() {
    Arrays.fill(wall, -1);
  }

  /** 席の配牌 13 枚を置く。 */
  public RuleScene hand(int seat, String tiles) {
    int[] faces = faces(tiles);
    Assert.assertEquals(faces.length, Variant.HAND, "a deal is 13 tiles: " + tiles);
    for (int index = 0; index < faces.length; index++) {
      int packet = index / PACKET;
      int position =
          packet < PACKETS
              ? (packet * seats + seat) * PACKET + index % PACKET
              : PACKETS * PACKET * seats + seat;
      place(position, faces[index]);
    }
    return this;
  }

  /** 局の最初からの山のツモを、全員を通して引く順に置く。 */
  public RuleScene draws(String tiles) {
    int[] faces = faces(tiles);
    for (int index = 0; index < faces.length; index++) {
      place(Variant.HAND * seats + index, faces[index]);
    }
    return this;
  }

  /** 最初のドラ表示を置く。 */
  public RuleScene indicator(String tile) {
    place(set.size() - FIRST_INDICATOR_FROM_END, faces(tile)[0]);
    return this;
  }

  /** 残りの位置を埋めて局を始め、親の最初の手番まで進める。 */
  public RuleScene start() {
    List<Integer> rest = new ArrayList<>();
    for (int index = 0; index < set.size(); index++) {
      if (!used[set.physicalId(index)]) {
        rest.add(set.physicalId(index));
      }
    }
    Collections.shuffle(rest, new Random(FILL_SEED));
    int next = 0;
    for (int position = 0; position < wall.length; position++) {
      if (wall[position] < 0) {
        wall[position] = rest.get(next++);
      }
    }
    engine.startRecordedRound(RoundStart.first(VARIANT), wall);
    boundary = engine.advance(ws);
    return this;
  }

  /** 判断する席ごとに、"席:手の名前" で書いた手を選ぶ。書かない席は見送る。書く手はどれも、いま判断する席のものでなければならない。 */
  public RuleScene play(String... picks) {
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
    for (String pick : picks) {
      Assert.assertTrue(decisions.slotOf(pick.charAt(0) - '0') >= 0, pick + " has no decision at this boundary");
    }
    boundary = engine.commit(choices, ws);
    return this;
  }

  /** 席の判断の記録を返す。記録は次に進めるまで有効。その席に判断が無ければテストを失敗させる。 */
  public Observation observe(int seat) {
    Assert.assertEquals(boundary, Boundary.DECISIONS, "no decision is pending");
    Decisions decisions = engine.decisions();
    int slot = decisions.slotOf(seat);
    Assert.assertTrue(slot >= 0, "seat " + seat + " has no decision");
    batch.clear();
    return observation.bind(batch, writer.capture(engine.view(seat), decisions, slot, batch));
  }

  /** 席の合法手の名前を並べて返す（その席に判断が無ければ空）。 */
  public List<String> legal(int seat) {
    Decisions decisions = engine.decisions();
    int slot = decisions.slotOf(seat);
    List<String> names = new ArrayList<>();
    for (int index = 0; slot >= 0 && index < decisions.legalCount(slot); index++) {
      names.add(Action.of(decisions.legal(slot, index)).toString());
    }
    return names;
  }

  /** 牌の表記を、書いた順に番号の列にして返す。番号は牌の種類 × 2 で、赤なら 1 を足す。 */
  private static int[] faces(String text) {
    int[] faces = new int[text.length()];
    int size = 0;
    int digits = 0;
    for (int at = 0; at < text.length(); at++) {
      char c = text.charAt(at);
      if (Character.isDigit(c)) {
        digits++;
        continue;
      }
      int first =
          switch (c) {
            case 'm' -> Tiles.M1;
            case 'p' -> Tiles.P1;
            case 's' -> Tiles.S1;
            case 'z' -> Tiles.TON;
            default -> throw new IllegalArgumentException("Unknown suit " + c + " in " + text);
          };
      for (int digit = at - digits; digit < at; digit++) {
        int number = text.charAt(digit) - '0';
        faces[size++] = number == 0 ? (first + Tiles.FIVE) * 2 + 1 : (first + number - 1) * 2;
      }
      digits = 0;
    }
    return Arrays.copyOf(faces, size);
  }

  /** 番号で表した牌に当たる物理牌のうち、まだ使っていないものを 1 枚選んで、牌山の位置に置く。 */
  private void place(int position, int face) {
    Assert.assertEquals(wall[position], -1, "wall position " + position + " is set twice");
    int kind = face / 2;
    boolean red = face % 2 == 1;
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int physicalId = Tiles.physicalId(kind, copy);
      if (!used[physicalId] && set.red(physicalId) == red) {
        used[physicalId] = true;
        wall[position] = physicalId;
        return;
      }
    }
    throw new IllegalStateException("No free " + (red ? "red " : "") + "copy of kind " + kind);
  }
}
