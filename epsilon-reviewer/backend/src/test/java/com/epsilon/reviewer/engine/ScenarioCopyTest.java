package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.dora;
import static com.epsilon.reviewer.engine.TestEdits.hand;
import static com.epsilon.reviewer.engine.TestEdits.merge;
import static com.epsilon.reviewer.engine.TestEdits.rivers;

import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordEvent.Discard;
import com.epsilon.mahjong.record.RecordEvent.Draw;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.LocatedDecision;
import com.epsilon.reviewer.engine.scenario.RewrittenRound;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.RoundRewriter;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.IntUnaryOperator;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * もしもの変更で置く牌の枚数のテスト。変えた局面で見える枚数が、その牌の写しの数（赤の 5 は 1 枚、赤でない 5 は 3 枚、ほかは 4 枚）
 * 以内なら変更は通る。ある変更が外した牌の写しは、変更の並び順によらず、ほかの変更が使える。通った変更には物理牌を割り当て、判断まで
 * 再生できる。判断より前のイベントで判断する席に見える牌どうしは、物理牌が重ならない。新しく置く牌には、空いている写しのうち番号の
 * 最も小さいものを割り当てる。判断する席から見えない写しも、空いている写しに数える。
 */
public class ScenarioCopyTest {
  private static final int SEAT = 0;

  /** 配牌。席 2 の赤の 5p は判断まで伏せたまま。席 3 は W を 2 枚（116・117）伏せている。 */
  private static final int[][] DEAL = {
    {0, 4, 8, 53, 54, 72, 76, 80, 108, 109, 28, 29, 120},
    {30, 55, 36, 40, 44, 48, 56, 60, 68, 84, 92, 96, 104},
    {31, 52, 12, 20, 24, 32, 64, 100, 112, 113, 124, 128, 132},
    {88, 1, 5, 9, 13, 21, 37, 41, 45, 116, 117, 125, 129}
  };

  /** ドラ表示牌の 9s。 */
  private static final int INDICATOR = 105;

  /**
   * 打った順に並べた {席, ツモ, 打牌}。ツモと打牌が同じ牌ならツモ切り。判断は最後にある席 0（親）の打牌。その時点の席 0 の手牌は
   * 1m 2m 3m 8m 5p 5p 1s 2s 3s E E N N とツモの C で、河は 5m（ツモ切り）と 8m（手出し）。8m は 4 枚とも、赤でない 5p は 3 枚とも
   * 見えていて、赤の 5s は席 3 の河にある。
   */
  private static final int[][] TURNS = {
    {0, 17, 17}, {1, 33, 30}, {2, 65, 31}, {3, 14, 88},
    {0, 121, 28}, {1, 69, 55}, {2, 114, 114}, {3, 118, 118},
    {0, 133, 133}
  };

  /** 判断する席がツモ切りした 5m。 */
  private static final int DISCARDED_5M = 17;

  /** 判断する席が手出しした 8m。 */
  private static final int DISCARDED_8M = 28;

  /** 席 1 が打った 8m。 */
  private static final int OPPONENT_8M = 30;

  /** 判断する席の手牌の E のうち、最も後に入った 1 枚。 */
  private static final int NEWEST_E = 109;

  /** 席 3 が伏せている W のうち番号の小さい方。判断する席から見えないので、W に替えた牌に割り当てる。 */
  private static final int HIDDEN_W = 116;

  /** 牌山に残っている W。 */
  private static final int UNSEEN_W = 119;

  /** 判断が決まる記録のイベントの位置（最後の打牌）。 */
  private static final int POSITION = 2 * TURNS.length - 1;

  private static final LoadedRecord ROUND = round(Map.of());
  private static final int DECISION = ROUND.step(0, POSITION);
  private static final LocatedDecision LOCATED = Scenarios.find(ROUND, 0, DECISION, SEAT);

  /** 判断する席の河で、ツモ切りの 5m と手出しの 8m を入れ替える。変更をどちらの順に並べても、5m と 8m を逆の順に打った局と同じになる。 */
  @Test
  public void swappingOwnDiscardsReusesTheCopiesTheyFree() {
    var eight = new RiverEdit(SEAT, 0, "8m", true);
    var five = new RiverEdit(SEAT, 1, "5m", false);
    Map<Integer, Integer> swapped = Map.of(DISCARDED_5M, DISCARDED_8M, DISCARDED_8M, DISCARDED_5M);
    assertSameAs(swapped, rivers(eight, five));
    assertSameAs(swapped, rivers(five, eight));
  }

  /**
   * 判断する席が手出しした 8m を W に替えると、8m の写しが 1 つ空き、手牌の E を 8m に替える変更がそれを使う。局を作り直すときは、
   * 席 3 が伏せている W を河の W に回し、席 3 には牌山に残っている W を持たせる。
   */
  @Test
  public void handEditReusesTheCopyAnOwnDiscardEditFrees() {
    assertSameAs(
        Map.of(NEWEST_E, DISCARDED_8M, DISCARDED_8M, HIDDEN_W, HIDDEN_W, UNSEEN_W),
        merge(
            rivers(new RiverEdit(SEAT, 1, "W", false)), hand(new HandEdit(handIndex("E"), "8m"))));
  }

  /** 席 1 の 8m を W に替えて空いた写しを、手牌の E を 8m に替える変更が使う。 */
  @Test
  public void handEditReusesTheCopyAnOpponentDiscardEditFrees() {
    assertSameAs(
        Map.of(NEWEST_E, OPPONENT_8M, OPPONENT_8M, HIDDEN_W, HIDDEN_W, UNSEEN_W),
        merge(rivers(new RiverEdit(1, 0, "W", false)), hand(new HandEdit(handIndex("E"), "8m"))));
  }

  /** 写しの数を超えたときは、その牌を置いた最初の変更を名指す。ツモ切りの印だけを変えた河の牌は、牌を置いたことにならない。 */
  @Test
  public void theFirstEditThatPlacesTheTileIsNamed() {
    rejected(
        merge(rivers(new RiverEdit(2, 1, "5pr", true)), dora(new DoraEdit(0, "5pr"))),
        "edits.rivers[0]",
        "5pr");
    rejected(
        merge(rivers(new RiverEdit(1, 0, "8m", true)), hand(new HandEdit(handIndex("E"), "8m"))),
        "edits.hand[0]",
        "8m");
  }

  /**
   * 判断の局面で変えられる全ての所について、37 種の牌のどれかに替える変更と、2 つの所の牌を入れ替える変更を試す。見える枚数が写しの数を
   * 超える変更だけを断り、ほかは物理牌を割り当てて判断まで再生できる。
   */
  @Test
  public void everyEditWithinTheCopiesIsAssigned() {
    Map<String, Integer> visible = ScenarioEditTest.visible(LOCATED.table(), SEAT);
    List<Place> places = places(LOCATED.table());
    for (Place place : places)
      for (String tile : ScenarioEditTest.TILES) {
        ScenarioEdits edits = place.editTo().apply(tile);
        if (!tile.equals(place.tile())
            && visible.getOrDefault(tile, 0) >= ScenarioEditTest.copies(tile))
          rejected(edits, place.field() + "[0]", tile);
        else accepted(edits);
      }
    for (int i = 0; i < places.size(); i++)
      for (int j = i + 1; j < places.size(); j++) {
        Place a = places.get(i);
        Place b = places.get(j);
        if (!a.tile().equals(b.tile()))
          accepted(merge(a.editTo().apply(b.tile()), b.editTo().apply(a.tile())));
      }
  }

  /**
   * 変更したもしもが、物理牌を {@code relabel} で付け替えた局（変えた牌を実際に打った局）の判断と同じ O と局面になることを確かめる。
   * 判断より前のイベントで判断する席に見える牌どうしの物理牌が、重ならないことも確かめる。
   */
  private static void assertSameAs(Map<Integer, Integer> relabel, ScenarioEdits edits) {
    LoadedRecord played = round(relabel);
    Assert.assertEquals(input(ROUND, edits), input(played, TestEdits.NONE));
    RewrittenRound rewritten = rewrite(edits);
    Assert.assertEquals(rewritten.table(), Scenarios.find(played, 0, DECISION, SEAT).table());
    assertDistinctTiles(rewritten);
  }

  private static void accepted(ScenarioEdits edits) {
    input(ROUND, edits);
    assertDistinctTiles(rewrite(edits));
  }

  private static void rejected(ScenarioEdits edits, String field, String tile) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> input(ROUND, edits));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_EDIT, failure.getMessage());
    Assert.assertEquals(
        failure.getMessage(),
        field
            + ": more copies of "
            + tile
            + " would be visible than the "
            + ScenarioEditTest.copies(tile)
            + " in the set.");
  }

  /**
   * 判断より前のイベントで判断する席に見える牌（その席の配牌とツモ、他家の打牌、表示牌）の物理牌が重ならないことと、判断する席の打牌が
   * その席の手にある牌であることを確かめる。他家の配牌とツモは判断する席から見えないので比べない（この局に鳴きは無い）。
   */
  private static void assertDistinctTiles(RewrittenRound round) {
    Set<Integer> seen = new HashSet<>();
    Set<Integer> held = new HashSet<>();
    IntConsumer enter =
        id -> {
          Assert.assertTrue(seen.add(id), "tile " + id + " is seen twice");
          held.add(id);
        };
    Assert.assertTrue(seen.add(round.indicator()), "indicator " + round.indicator());
    Arrays.stream(round.hand()).forEach(enter);
    for (RecordEvent event : round.events())
      switch (event) {
        case Draw draw when draw.seat() == SEAT -> enter.accept(draw.tile());
        case Discard discard when discard.seat() == SEAT ->
            Assert.assertTrue(held.remove(discard.tile()), "not in the hand: " + event);
        case Discard discard ->
            Assert.assertTrue(seen.add(discard.tile()), "seen twice: " + event);
        default -> {}
      }
  }

  /** 判断で方策が受け取った O の写し。 */
  private static long[] input(LoadedRecord record, ScenarioEdits edits) {
    var policy = new RecordingPolicy();
    Scenarios.evaluate(record, new ScenarioQuery(0, DECISION, SEAT, edits, null), policy);
    Assert.assertEquals(policy.inputs.size(), 1);
    return policy.inputs.getFirst();
  }

  private static RewrittenRound rewrite(ScenarioEdits edits) {
    return RoundRewriter.apply(ROUND.rounds().getFirst(), LOCATED, edits);
  }

  /** 物理牌を {@code relabel} で付け替えた局。席から見えない所も同じように付け替えるので、牌譜として成り立つ。 */
  private static LoadedRecord round(Map<Integer, Integer> relabel) {
    IntUnaryOperator id = tile -> relabel.getOrDefault(tile, tile);
    int[][] deal =
        Arrays.stream(DEAL)
            .map(hand -> Arrays.stream(hand).map(id).toArray())
            .toArray(int[][]::new);
    List<RecordEvent> events = new ArrayList<>();
    for (int[] turn : TURNS) {
      events.add(new Draw(turn[0], id.applyAsInt(turn[1])));
      events.add(new Discard(turn[0], id.applyAsInt(turn[2]), turn[1] == turn[2], false));
    }
    return TestRounds.round(deal, INDICATOR, events.toArray(RecordEvent[]::new));
  }

  /** 判断の局面の手牌で、{@code tile} が最初に現れる位置。 */
  private static int handIndex(String tile) {
    return LOCATED.table().players().get(SEAT).hand().indexOf(tile);
  }

  /**
   * 判断の局面で変えられる所 1 つ。
   *
   * @param field 変更のフィールドの名前
   * @param tile 判断の時点でそこにある牌
   * @param editTo 牌を受け取り、そこをその牌に替える変更を作る関数（河の牌はツモ切りの印を保つ）
   */
  private record Place(String field, String tile, Function<String, ScenarioEdits> editTo) {}

  /** 変えられる所を並べる。判断する席の手牌とツモ牌、全ての河の牌（この局に鳴きは無い）、ドラ表示牌。 */
  private static List<Place> places(TableState table) {
    List<Place> places = new ArrayList<>();
    PlayerState own = table.players().get(SEAT);
    for (int i = 0; i <= own.hand().size(); i++) {
      int index = i;
      String tile = index < own.hand().size() ? own.hand().get(index) : own.draw();
      places.add(new Place("edits.hand", tile, code -> hand(new HandEdit(index, code))));
    }
    for (int seat = 0; seat < table.players().size(); seat++)
      for (int i = 0; i < table.players().get(seat).river().size(); i++) {
        int player = seat;
        int index = i;
        RiverTile old = table.players().get(seat).river().get(index);
        places.add(
            new Place(
                "edits.rivers",
                old.tile(),
                code -> rivers(new RiverEdit(player, index, code, old.tsumogiri()))));
      }
    places.add(
        new Place("edits.dora", table.dora().getFirst(), code -> dora(new DoraEdit(0, code))));
    return places;
  }
}
