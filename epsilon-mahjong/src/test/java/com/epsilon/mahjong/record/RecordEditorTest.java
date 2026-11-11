package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.engine.SeatTable;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 1 つの席から見た、判断の時点までの記録を書き換える {@code RecordEditor} のテスト。書き換えない記録を受けた {@code SeatTable} は、
 * エンジンと同じ判断に行き着く。書き換えは替えた所だけを変え、元の牌に戻すと元の記録に戻る。
 */
public class RecordEditorTest {

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /**
   * 自己対局の記録の全ての判断で、判断する席ごとの記録は、判断の時点までのイベントから他家のツモを伏せた並びになる。それを受けた
   * {@code SeatTable} の合法手は、エンジンの合法手と同じになる。
   */
  @Test(dataProvider = "variants")
  public void theSeatsRecordReachesEveryDecision(Variant variant) {
    List<RecordedGame> games = new ArrayList<>();
    for (int seed = 0; seed < 10; seed++) {
      games.add(RecordedGames.play(variant, seed, "self-" + seed));
    }
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    int compared = 0;
    for (RecordedGame game : games) {
      for (RecordedRound round : game.rounds()) {
        Boundary boundary = driver.start(round, ws);
        while (boundary == Boundary.DECISIONS) {
          Decisions decisions = driver.engine().decisions();
          int cursor = driver.cursor();
          for (int slot = 0; slot < decisions.size(); slot++) {
            int seat = decisions.seat(slot);
            String where = game.meta().id() + " " + round.start() + " seat " + seat + " at " + cursor;
            RecordEditor editor = new RecordEditor(round, cursor, seat);
            Assert.assertEquals(editor.events(), shown(round.events().subList(0, cursor), seat), where);
            SeatTable table = table(editor, ws);
            Assert.assertEquals(legal(table.decisions(), 0), legal(decisions, slot), where);
            compared++;
          }
          boundary = driver.step(ws);
        }
      }
    }
    Assert.assertTrue(compared > 5_000, compared + " decisions");
  }

  /**
   * 自分の打牌を替えると、その牌が手に入った所（配牌かツモ）も同じ牌に替わる。他家の打牌、手にある配牌とツモ、表示牌、局の始まりも、
   * 替えた所だけが変わる。どれも元の牌に戻すと、元の記録に戻る。
   */
  @Test
  public void rewritesTouchOnlyTheirPlacesAndComeBack() {
    int checked = 0;
    for (int seed = 0; seed < 5; seed++) {
      RecordedGame game = RecordedGames.play(Variant.TENHOU_4P, seed, "self-" + seed);
      for (RecordedRound round : game.rounds()) {
        List<RecordEvent> events = round.events();
        int seat = round.start().variant().dealer(round.start().round());
        RecordEditor original = new RecordEditor(round, events.size(), seat);
        RecordEditor editor = new RecordEditor(round, events.size(), seat);
        int spare = spares(round, 1)[0];
        for (int position = 0; position < events.size(); position++) {
          if (events.get(position) instanceof RecordEvent.Discard discard && callOf(events, position) == null) {
            List<RecordEvent> before = editor.events();
            int[] hand = editor.hand();
            editor.replaceDiscarded(position, spare);
            List<RecordEvent> after = editor.events();
            Assert.assertEquals(
                after.get(position),
                new RecordEvent.Discard(discard.seat(), spare, discard.tsumogiri(), discard.riichi()));
            int changed = changes(before, after) + changes(hand, editor.hand());
            Assert.assertEquals(changed, discard.seat() == seat ? 2 : 1, game.meta().id() + " " + position);
            editor.replaceDiscarded(position, discard.tile());
            Assert.assertEquals(editor.events(), before);
            Assert.assertEquals(editor.hand(), hand);
            checked++;
          }
        }
        for (int index = 0; index < Variant.HAND; index++) {
          int dealt = editor.hand()[index];
          if (discarded(events, seat, dealt)) {
            int at = index;
            Assert.assertThrows(IllegalArgumentException.class, () -> editor.replaceDealt(at, spare));
          } else {
            editor.replaceDealt(index, spare);
            Assert.assertEquals(editor.hand()[index], spare);
            Assert.assertEquals(editor.events(), original.events());
            editor.replaceDealt(index, dealt);
          }
        }
        editor.replaceIndicator(0, spare);
        Assert.assertEquals(editor.indicator(), spare);
        editor.replaceIndicator(0, round.indicator());
        int reveals = (int) events.stream().filter(RecordEvent.DoraReveal.class::isInstance).count();
        Assert.assertThrows(IllegalArgumentException.class, () -> editor.replaceIndicator(reveals + 1, spare));
        int[] scores = round.start().scores();
        int[] swapped = {scores[1], scores[0], scores[3], scores[2]};
        editor.replaceStart(round.start().honba() + 1, round.start().kyotaku() + 2, swapped);
        Assert.assertEquals(editor.scores(), swapped);
        Assert.assertEquals(editor.honba(), round.start().honba() + 1);
        Assert.assertEquals(editor.kyotaku(), round.start().kyotaku() + 2);
        editor.replaceStart(round.start().honba(), round.start().kyotaku(), scores);
        assertSame(editor, original);
      }
    }
    Assert.assertTrue(checked > 1_000, checked + " discards");
  }

  /**
   * もしもの局面を作る。全員がツモ切りで流局まで打つ局の 6 回目の手番の判断で、手にある配牌の 1 枚を誰にも見えない牌に替え、他家の
   * 最後の打牌を別の牌に替えると、{@code SeatTable} の手牌と河と合法手がそれに従う。最初の表示牌を替えるとドラが替わる。局の始まりの
   * 本場・供託・点は、点と供託の和が配給原点の和と合わない値に替えても卓に載る。席から見える本場・供託・点はその値になり、合法手は
   * 替える前と同じ。
   */
  @Test
  public void aRewrittenRecordPlaysOnTheSeatTable() {
    RecordedRound round = RecordedGames.scene(Scenes.first(Variant.TENHOU_4P), "tsumogiri").rounds().get(0);
    Variant variant = round.start().variant();
    RuleWorkspace ws = new RuleWorkspace(variant);
    ReplayDriver driver = new ReplayDriver(variant);
    Boundary boundary = driver.start(round, ws);
    int turns = 0;
    while (boundary == Boundary.DECISIONS) {
      Decisions decisions = driver.engine().decisions();
      if (decisions.kind(0) == EngineDecisionKind.TURN && ++turns == 6) {
        break;
      }
      boundary = driver.step(ws);
    }
    Assert.assertEquals(boundary, Boundary.DECISIONS);
    int seat = driver.engine().decisions().seat(0);
    int cursor = driver.cursor();
    RecordEditor editor = new RecordEditor(round, cursor, seat);
    int index = heldDealt(editor, round.events().subList(0, cursor));
    int[] unseen = spares(round, 3);
    editor.replaceDealt(index, unseen[0]);
    int other = lastDiscardOfAnother(editor.events(), seat);
    editor.replaceDiscarded(other, unseen[1]);
    editor.replaceIndicator(0, unseen[2]);
    SeatTable table = table(editor, ws);
    Assert.assertTrue(table.view().holds(unseen[0]));
    Assert.assertFalse(table.view().holds(round.deal(seat, index)));
    int discarder = ((RecordEvent.Discard) editor.events().get(other)).seat();
    Assert.assertTrue(Tiles.contains(table.view().riverKinds(discarder), Tiles.kind(unseen[1])));
    Assert.assertEquals(table.view().indicator(0), unseen[2]);
    List<String> moves = legal(table.decisions(), 0);
    Assert.assertTrue(moves.stream().anyMatch(code -> code.startsWith("DAHAI(")));

    int honba = round.start().honba() + 1;
    int kyotaku = round.start().kyotaku() + 2;
    int[] scores = {40_000, 30_000, 20_000, 10_000};
    editor.replaceStart(honba, kyotaku, scores);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new RoundStart(variant, editor.round(), honba, kyotaku, scores));
    SeatTable moved = table(editor, ws);
    Assert.assertEquals(moved.view().honba(), honba);
    Assert.assertEquals(moved.view().kyotaku(), kyotaku);
    for (int player = 0; player < scores.length; player++) {
      Assert.assertEquals(moved.view().score(player), scores[player], "seat " + player);
    }
    Assert.assertEquals(legal(moved.decisions(), 0), moves);
  }

  /**
   * 他家のツモと、打牌でないイベントは替えられない。ルールに無い牌にも替えられない。鳴かれた打牌は、面子と同じ種類で、赤かどうかも同じ
   * 牌にだけ替えられる。
   */
  @Test
  public void impossibleRewritesAreRejected() {
    RecordedRound round = ReplayDriverTest.ponOverChi().round(ReplayDriverTest.ponOverChiResult());
    List<RecordEvent> events = round.events();
    RecordEditor editor = new RecordEditor(round, events.size(), 1);
    Assert.assertThrows(IllegalArgumentException.class, () -> editor.replaceDrawn(0, Tiles.physicalId(Tiles.M1, 3)));
    Assert.assertThrows(IllegalArgumentException.class, () -> editor.replaceDiscarded(2, Tiles.physicalId(Tiles.M1, 3)));
    Assert.assertThrows(IllegalArgumentException.class, () -> editor.replaceDiscarded(1, Tiles.physicalId(Tiles.P4, 3)));
    RecordEvent.Discard called = (RecordEvent.Discard) events.get(1);
    int otherCopy = Tiles.physicalId(Tiles.P3, Tiles.copy(called.tile()) == 1 ? 2 : 1);
    editor.replaceDiscarded(1, otherCopy);
    Assert.assertEquals(((RecordEvent.Discard) editor.events().get(1)).tile(), otherCopy);
    Assert.assertEquals(editor.events().get(0), new RecordEvent.Draw(3, TurnEvent.NONE));
    Assert.assertThrows(IndexOutOfBoundsException.class, () -> new RecordEditor(round, events.size() + 1, 1));

    RecordedRound sanma = RecordedGames.play(Variant.TENHOU_3P, 0, "self-0").rounds().get(0);
    RecordEditor three = new RecordEditor(sanma, 0, 0);
    Assert.assertThrows(IllegalArgumentException.class, () -> three.replaceDealt(0, Tiles.physicalId(Tiles.M2, 0)));
    Assert.assertThrows(IllegalArgumentException.class, () -> three.replaceStart(0, 0, new int[4]));
  }

  /**
   * 書き換えた記録から局の始まり・自分の配牌・最初の表示牌を取ってもしもの局を始め（局の始まりは値の範囲だけを確かめる）、記録の
   * イベントを受けさせた {@code SeatTable} を返す。
   */
  private static SeatTable table(RecordEditor editor, RuleWorkspace ws) {
    SeatTable table = new SeatTable(editor.variant(), editor.seat());
    table.startScenario(
        editor.round(), editor.honba(), editor.kyotaku(), editor.scores(), editor.hand(), editor.indicator());
    for (RecordEvent event : editor.events()) {
      event.applyTo(table, ws);
    }
    return table;
  }

  /**
   * その席から見たイベントの並びを返す。他家のツモの牌は伏せる。
   *
   * @param seat 見る席
   */
  private static List<RecordEvent> shown(List<RecordEvent> events, int seat) {
    List<RecordEvent> shown = new ArrayList<>();
    for (RecordEvent event : events) {
      shown.add(
          event instanceof RecordEvent.Draw draw && draw.seat() != seat
              ? new RecordEvent.Draw(draw.seat(), TurnEvent.NONE)
              : event);
    }
    return shown;
  }

  /**
   * 判断する席の合法手の名前を返す。
   *
   * @param slot 判断する席の何番目か
   */
  private static List<String> legal(Decisions decisions, int slot) {
    List<String> names = new ArrayList<>();
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      names.add(Action.of(decisions.legal(slot, index)).toString());
    }
    return names;
  }

  /**
   * 局の記録に出てこない物理牌（配牌・ツモ・表示牌・裏ドラ表示牌のどれでもない牌）を、小さい順に選ぶ。
   *
   * @param count 選ぶ枚数
   */
  private static int[] spares(RecordedRound round, int count) {
    boolean[] seen = new boolean[Tiles.PHYSICAL_IDS];
    for (int seat = 0; seat < round.start().variant().seats(); seat++) {
      for (int index = 0; index < Variant.HAND; index++) {
        seen[round.deal(seat, index)] = true;
      }
    }
    seen[round.indicator()] = true;
    for (RecordEvent event : round.events()) {
      switch (event) {
        case RecordEvent.Draw draw -> seen[draw.tile()] = true;
        case RecordEvent.DoraReveal reveal -> seen[reveal.tile()] = true;
        default -> {}
      }
    }
    for (int ura : round.result().uraIndicators()) {
      seen[ura] = true;
    }
    int[] spares = new int[count];
    int found = 0;
    for (int tile = 0; tile < Tiles.PHYSICAL_IDS && found < count; tile++) {
      if (!seen[tile] && round.start().variant().tiles().contains(tile)) {
        spares[found++] = tile;
      }
    }
    Assert.assertEquals(found, count, "unseen tiles in " + round.start());
    return spares;
  }

  /**
   * 打牌を鳴いたイベントを返す。
   *
   * @param position 打牌の位置
   * @return 鳴いたイベント。鳴かれていなければ null
   */
  private static RecordEvent.Call callOf(List<RecordEvent> events, int position) {
    for (int at = position + 1; at < events.size(); at++) {
      RecordEvent event = events.get(at);
      if (event instanceof RecordEvent.Call call) {
        return call;
      }
      if (!(event instanceof RecordEvent.DoraReveal) && !(event instanceof RecordEvent.RiichiAccepted)) {
        return null;
      }
    }
    return null;
  }

  /**
   * 席がその物理牌を打ったか抜いたかを返す。
   *
   * @param seat 見る席
   * @param tile 物理 id
   */
  private static boolean discarded(List<RecordEvent> events, int seat, int tile) {
    return events.stream()
        .anyMatch(
            event -> event instanceof RecordEvent.Discard discard && discard.seat() == seat && discard.tile() == tile
                || event instanceof RecordEvent.Nuki nuki && nuki.seat() == seat && nuki.tile() == tile);
  }

  /**
   * 判断の時点でまだ手にある配牌を 1 枚選び、配牌の何番目かを返す。選ぶのは、判断の時点までに打っても抜いてもおらず、鳴いた面子の
   * 種類でもない配牌。
   *
   * @param events 判断の時点までのイベント
   */
  private static int heldDealt(RecordEditor editor, List<RecordEvent> events) {
    long melded = 0;
    for (RecordEvent event : events) {
      switch (event) {
        case RecordEvent.Call call when call.seat() == editor.seat() -> {
          Meld meld = Meld.of(call.meld());
          for (int index = 0; index < meld.size(); index++) {
            melded |= Tiles.bit(meld.tile(index));
          }
        }
        case RecordEvent.ClosedKan kan when kan.seat() == editor.seat() -> melded |= Tiles.bit(kan.kind());
        case RecordEvent.AddedKan kan when kan.seat() == editor.seat() -> melded |= Tiles.bit(kan.kind());
        default -> {}
      }
    }
    int[] hand = editor.hand();
    for (int index = 0; index < hand.length; index++) {
      if (!discarded(events, editor.seat(), hand[index]) && !Tiles.contains(melded, Tiles.kind(hand[index]))) {
        return index;
      }
    }
    throw new AssertionError("Seat " + editor.seat() + " holds no dealt tile");
  }

  /**
   * ほかの席の打牌のうち、鳴かれていない最後のものの位置を返す。
   *
   * @param seat 除く席
   */
  private static int lastDiscardOfAnother(List<RecordEvent> events, int seat) {
    for (int position = events.size() - 1; position >= 0; position--) {
      if (events.get(position) instanceof RecordEvent.Discard discard
          && discard.seat() != seat
          && callOf(events, position) == null) {
        return position;
      }
    }
    throw new AssertionError("No discard of another seat");
  }

  private static int changes(List<RecordEvent> before, List<RecordEvent> after) {
    int changes = 0;
    for (int index = 0; index < before.size(); index++) {
      changes += before.get(index).equals(after.get(index)) ? 0 : 1;
    }
    return changes;
  }

  private static int changes(int[] before, int[] after) {
    int changes = 0;
    for (int index = 0; index < before.length; index++) {
      changes += before[index] == after[index] ? 0 : 1;
    }
    return changes;
  }

  private static void assertSame(RecordEditor editor, RecordEditor original) {
    Assert.assertEquals(editor.events(), original.events());
    Assert.assertEquals(editor.hand(), original.hand());
    Assert.assertEquals(editor.indicator(), original.indicator());
    Assert.assertEquals(editor.scores(), original.scores());
    Assert.assertEquals(editor.honba(), original.honba());
    Assert.assertEquals(editor.kyotaku(), original.kyotaku());
  }
}
