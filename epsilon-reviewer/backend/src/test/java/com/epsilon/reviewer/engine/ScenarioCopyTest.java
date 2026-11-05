package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.dora;
import static com.epsilon.reviewer.engine.TestEdits.hand;
import static com.epsilon.reviewer.engine.TestEdits.merge;
import static com.epsilon.reviewer.engine.TestEdits.rivers;

import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.DecisionLocator;
import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.LocatedDecision;
import com.epsilon.reviewer.engine.scenario.RewrittenRound;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.RoundRewriter;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
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
 * もしもの変更が置く牌は、変えた局面に見える数が牌の写しの数（赤の 5 は 1 枚、ほかの 5 は 3 枚、ほかは 4 枚）までなら通り、変更が外した牌の
 * 写しはほかの変更が順に依らず使えることを確かめる。通った変更は物理牌を割り当てて判断まで再生でき、判断より前の出来事で判断の席に見える牌の
 * 物理牌は重ならない。
 */
public class ScenarioCopyTest {
  private static final int SEAT = 0;

  /** 配牌。席 2 の赤の 5p は判断まで伏せたまま。 */
  private static final int[][] DEAL = {
    {0, 4, 8, 53, 54, 72, 76, 80, 108, 109, 28, 29, 120},
    {30, 55, 36, 40, 44, 48, 56, 60, 68, 84, 92, 96, 104},
    {31, 52, 12, 20, 24, 32, 64, 100, 112, 113, 124, 128, 132},
    {88, 1, 5, 9, 13, 21, 37, 41, 45, 116, 117, 125, 129}
  };

  /**
   * 打牌の順の {席, ツモ, 打牌}（同じ牌ならツモ切り）。判断は最後の席 0（親）の打牌で、その時点の席 0 の手は 1m 2m 3m 8m 5p 5p 1s 2s 3s E E
   * N N とツモの C、河は 5m（ツモ切り）と 8m（手出し）。8m は 4 枚とも、赤でない 5p は 3 枚とも見え、赤の 5s は席 3 の河にある。ドラ表示は 9s。
   */
  private static final int[][] TURNS = {
    {0, 17, 17}, {1, 33, 30}, {2, 65, 31}, {3, 14, 88},
    {0, 121, 28}, {1, 69, 55}, {2, 114, 114}, {3, 118, 118},
    {0, 133, 133}
  };

  /** 判断の席がツモ切りした 5m。 */
  private static final int DISCARDED_5M = 17;

  /** 判断の席が手から打った 8m。 */
  private static final int DISCARDED_8M = 28;

  /** 席 1 が打った 8m。 */
  private static final int OPPONENT_8M = 30;

  /** 判断の席の手で最も後に手に入った E。 */
  private static final int NEWEST_E = 109;

  /** 牌山に残っている W。 */
  private static final int UNSEEN_W = 119;

  /** 判断の出来事の位置（対局の始めと局の始めの後に、巡ごとのツモと打牌が続く）。 */
  private static final int DECISION = 1 + 2 * TURNS.length;

  private static final LoadedRecord ROUND = round(Map.of());
  private static final LocatedDecision LOCATED = DecisionLocator.find(ROUND, 0, DECISION, SEAT);

  /** 判断の席の河のツモ切りの 5m と手出しの 8m を入れ替える。どちらの順でも、5m と 8m を逆に打った局と同じになる。 */
  @Test
  public void swappingOwnDiscardsReusesTheCopiesTheyFree() {
    var eight = new RiverEdit(SEAT, 0, "8m", true);
    var five = new RiverEdit(SEAT, 1, "5m", false);
    Map<Integer, Integer> swapped = Map.of(DISCARDED_5M, DISCARDED_8M, DISCARDED_8M, DISCARDED_5M);
    assertSameAs(swapped, rivers(eight, five));
    assertSameAs(swapped, rivers(five, eight));
  }

  /** 判断の席の手出しの 8m を W に替えて空いた写しを、手牌の E を 8m に替える変更が使う。 */
  @Test
  public void handEditReusesTheCopyAnOwnDiscardEditFrees() {
    assertSameAs(
        Map.of(NEWEST_E, DISCARDED_8M, DISCARDED_8M, UNSEEN_W),
        merge(
            rivers(new RiverEdit(SEAT, 1, "W", false)), hand(new HandEdit(handIndex("E"), "8m"))));
  }

  /** 席 1 の 8m を W に替えて空いた写しを、手牌の E を 8m に替える変更が使う。 */
  @Test
  public void handEditReusesTheCopyAnOpponentDiscardEditFrees() {
    assertSameAs(
        Map.of(NEWEST_E, OPPONENT_8M, OPPONENT_8M, UNSEEN_W),
        merge(rivers(new RiverEdit(1, 0, "W", false)), hand(new HandEdit(handIndex("E"), "8m"))));
  }

  /** 写しの数を超えたら、その牌を置いた最初の変更を名指す。ツモ切りの印だけを変えた河の牌は、牌を置いていない。 */
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
   * 判断の局面の変えられる所の全てで、37 の牌のどれかに替える変更と、2 つの所の牌を入れ替える変更。見える数が写しの数を超える変更だけを断り、
   * ほかは物理牌を割り当てて判断まで再生できる。
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
   * 変更が、物理牌を relabel で付け替えた局（変えた牌をそう打った局）の判断と同じ方策の入力と局面になり、判断より前の出来事で判断の席に見える
   * 牌の物理牌が重ならないこと。
   */
  private static void assertSameAs(Map<Integer, Integer> relabel, ScenarioEdits edits) {
    LoadedRecord played = round(relabel);
    Assert.assertNull(PolicyInput.difference(input(played, TestEdits.NONE), input(ROUND, edits)));
    RewrittenRound rewritten = rewrite(edits);
    Assert.assertEquals(rewritten.table(), DecisionLocator.find(played, 0, DECISION, SEAT).table());
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
   * 判断より前の出来事で、判断の席に見える牌（その席の配牌とツモ、他家の打牌）の物理牌が重ならず、判断の席の打牌がその席の手にある牌であること。
   * 他家の配牌とツモは判断の席に見えないので比べない（この局に鳴きは無い）。
   */
  private static void assertDistinctTiles(RewrittenRound round) {
    Set<Integer> seen = new HashSet<>();
    Set<Integer> held = new HashSet<>();
    IntConsumer enter =
        id -> {
          Assert.assertTrue(seen.add(id), "tile " + id + " is seen twice");
          held.add(id);
        };
    for (ReplayEvent event : round.events().subList(0, LOCATED.target()))
      switch (event) {
        case StartKyoku start ->
            Arrays.stream(start.initialHandPhysicalTileIds()[SEAT]).forEach(enter);
        case Tsumo draw when draw.actor() == SEAT -> enter.accept(draw.physicalTileId());
        case Dahai discard when discard.actor() == SEAT ->
            Assert.assertTrue(held.remove(discard.physicalTileId()), "not in the hand: " + event);
        case Dahai discard ->
            Assert.assertTrue(seen.add(discard.physicalTileId()), "seen twice: " + event);
        default -> {}
      }
  }

  private static PolicyInput input(LoadedRecord record, ScenarioEdits edits) {
    var policy = new RecordingPolicy();
    ScenarioEngine.evaluate(record, new ScenarioQuery(0, DECISION, SEAT, edits, null), policy);
    Assert.assertEquals(policy.inputs.size(), 1);
    return policy.inputs.getFirst();
  }

  private static RewrittenRound rewrite(ScenarioEdits edits) {
    return RoundRewriter.apply(ROUND.replay().events(), LOCATED, edits);
  }

  /** 物理牌を relabel で付け替えた局（席の目に見えない所も同じに付け替えるので、牌譜として成り立つ）。 */
  private static LoadedRecord round(Map<Integer, Integer> relabel) {
    IntUnaryOperator id = tile -> relabel.getOrDefault(tile, tile);
    int[][] deal =
        Arrays.stream(DEAL)
            .map(hand -> Arrays.stream(hand).map(id).toArray())
            .toArray(int[][]::new);
    int[] scores = {25000, 25000, 25000, 25000};
    List<ReplayEvent> events = new ArrayList<>();
    events.add(new StartGame());
    events.add(new StartKyoku(Tile.TON, 1, 0, 0, 0, Tile.S9, deal, scores));
    for (int[] turn : TURNS) {
      events.add(new Tsumo(turn[0], id.applyAsInt(turn[1])));
      events.add(new Dahai(turn[0], id.applyAsInt(turn[2]), turn[1] == turn[2]));
    }
    return ScenarioReplayTest.synthetic(events);
  }

  /** 判断の局面の手牌での、牌 tile の最初の位置。 */
  private static int handIndex(String tile) {
    return LOCATED.table().players().get(SEAT).hand().indexOf(tile);
  }

  /**
   * 判断の局面の変えられる所。
   *
   * @param field 変更の欄の名前
   * @param tile 判断の時点の牌
   * @param editTo 牌を替える変更（河の牌はツモ切りの印を保つ）
   */
  private record Place(String field, String tile, Function<String, ScenarioEdits> editTo) {}

  /** 判断の席の手牌とツモ牌、全ての河の牌（この局に鳴きは無い）、ドラ表示。 */
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
