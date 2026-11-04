package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.DecisionLocator;
import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.RoundRewriter;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * もしもの変更の種類ごとに、方策の入力の該当する値だけが変わり、ほかの値と変更を写した局面のほかの部分が変わらないこと、局面に照らして成り
 * 立たない変更を断ることを確かめる。判断は天鳳の牌譜の、他家のリーチの後の打牌の判断を使う。
 */
public class ScenarioEditTest {
  private static final String[] RARE_TILES = {"E", "S", "W", "N", "P", "F", "C", "1m", "9m", "1p"};

  /** リーチしていない席の打牌の判断で、宣言牌の後にも打牌した（棒を出した）他家の河に、鳴かれていない牌が 3 枚以上ある。 */
  private static final Decision AFTER_RIICHI =
      Decision.find(
          ScenarioEditTest::discards,
          (table, step) ->
              table.players().get(step.actor()).riichi()
                  ? -1
                  : opponent(
                      table,
                      step,
                      player ->
                          player.riichi()
                              && !player.river().getLast().riichi()
                              && uncalled(player).size() >= 3));

  /** 打牌の判断で、他家の河に鳴かれた牌がある。 */
  private static final Decision AFTER_CALL =
      Decision.find(
          ScenarioEditTest::discards,
          (table, step) -> opponent(table, step, player -> calledIndex(player) >= 0));

  /** ポンできる他家の打牌への応答の判断。変える他家は打牌した席。 */
  private static final Decision PON_OFFER =
      Decision.find(step -> step.offers(Action.Type.PON), (table, step) -> table.activeSeat());

  @Test
  public void editingTheAnsweredDiscardRegeneratesTheResponses() {
    Decision decision = PON_OFFER;
    int index = decision.player().river().size() - 1;
    RiverTile old = decision.player().river().get(index);
    PlayerState own = decision.table().players().get(decision.step().actor());
    int[] visible = visible(decision.table(), decision.step().actor());
    String tile = null;
    for (String honor : List.of("E", "S", "W", "N", "P", "F", "C"))
      if (tile == null
          && !own.hand().contains(honor)
          && !honor.equals(old.tile())
          && visible[TileCode.type(honor)] < Tile.TILES_PER_TYPE) tile = honor;
    var edit = new RiverEdit(decision.opponent(), index, tile, old.tsumogiri());
    var policy = new RecordingPolicy();
    ScenarioAnswer answer =
        ScenarioEngine.evaluate(
            decision.record(),
            new ScenarioQuery(
                decision.round(),
                decision.step().eventIndex(),
                decision.step().actor(),
                rivers(edit),
                null),
            policy);
    boolean passed = decision.step().chosenType().equals("PASS");
    Assert.assertEquals(answer.decision().size(), 1);
    Assert.assertEquals(
        Action.fromIndex(answer.decision().actionIds()[0]).type(), Action.Type.PASS);
    Assert.assertEquals(answer.decision().chosen() == 0, passed);
    Assert.assertEquals(
        policy.inputs.getFirst().turn(),
        "discard " + decision.opponent() + " " + TileCode.type(tile) + " false");
  }

  @Test
  public void riverTileEditChangesOnlyThatDiscard() {
    Decision decision = AFTER_RIICHI;
    int index = uncalled(decision.player()).getFirst();
    RiverTile old = decision.player().river().get(index);
    String tile = rareTile(decision.table(), decision.step().actor(), old.tile());
    var edit = new RiverEdit(decision.opponent(), index, tile, old.tsumogiri());
    Answer base = decision.answer(TestEdits.NONE);
    Answer edited = decision.answer(rivers(edit));
    String event = onlyChangedEvent(base.input(), edited.input());
    Assert.assertEquals(field(event, 2), Integer.toString(TileCode.type(tile)));
    assertUnchanged(
        base.input(),
        edited.input(),
        PolicyInput::hand,
        PolicyInput::melds,
        PolicyInput::seats,
        PolicyInput::scores,
        PolicyInput::honba,
        PolicyInput::kyotaku,
        PolicyInput::indicators,
        PolicyInput::dora,
        PolicyInput::turn,
        PolicyInput::legal,
        PolicyInput::round);
    Assert.assertEquals(
        edited.table(), withRiverTile(base.table(), edit, old.riichi()), "table outside the edit");
  }

  /** 配牌から来た牌もツモから来た牌も、入った所ごと替わるので、判断の手は変わらない。 */
  @Test
  public void ownRiverTileEditChangesOnlyThatDiscardAndKeepsTheHand() {
    Decision decision = AFTER_RIICHI;
    int seat = decision.step().actor();
    PlayerState own = decision.table().players().get(seat);
    Answer base = decision.answer(TestEdits.NONE);
    List<Integer> indices = uncalled(own);
    Assert.assertTrue(
        indices.stream().anyMatch(index -> own.river().get(index).tsumogiri())
            && indices.stream().anyMatch(index -> !own.river().get(index).tsumogiri()),
        "the river has both kinds of discards");
    for (int index : indices) {
      RiverTile old = own.river().get(index);
      var edit =
          new RiverEdit(seat, index, rareTile(decision.table(), seat, old.tile()), old.tsumogiri());
      Answer edited = decision.answer(rivers(edit));
      String event = onlyChangedEvent(base.input(), edited.input());
      Assert.assertEquals(field(event, 1), Integer.toString(seat));
      Assert.assertEquals(field(event, 2), Integer.toString(TileCode.type(edit.tile())));
      assertUnchanged(
          base.input(),
          edited.input(),
          PolicyInput::hand,
          PolicyInput::melds,
          PolicyInput::scores,
          PolicyInput::indicators,
          PolicyInput::dora,
          PolicyInput::turn,
          PolicyInput::legal,
          PolicyInput::round);
      Assert.assertEquals(edited.table(), withRiverTile(base.table(), edit, old.riichi()));
    }
  }

  @Test
  public void tsumogiriEditChangesOnlyTheFlagOfThatDiscard() {
    Decision decision = AFTER_RIICHI;
    int index = uncalled(decision.player()).getFirst();
    RiverTile old = decision.player().river().get(index);
    var edit = new RiverEdit(decision.opponent(), index, old.tile(), !old.tsumogiri());
    Answer base = decision.answer(TestEdits.NONE);
    Answer edited = decision.answer(rivers(edit));
    String event = onlyChangedEvent(base.input(), edited.input());
    Assert.assertEquals(field(event, 4), Boolean.toString(!old.tsumogiri()));
    assertUnchanged(
        base.input(),
        edited.input(),
        PolicyInput::hand,
        PolicyInput::melds,
        PolicyInput::seats,
        PolicyInput::scores,
        PolicyInput::indicators,
        PolicyInput::dora,
        PolicyInput::turn,
        PolicyInput::legal,
        PolicyInput::round,
        PolicyInput::furiten);
    Assert.assertEquals(edited.table(), withRiverTile(base.table(), edit, old.riichi()));
  }

  @Test
  public void doraEditChangesOnlyTheIndicator() {
    Decision decision = AFTER_RIICHI;
    String tile =
        rareTile(decision.table(), decision.step().actor(), decision.table().dora().getFirst());
    Answer base = decision.answer(TestEdits.NONE);
    Answer edited =
        decision.answer(
            new ScenarioEdits(null, List.of(new DoraEdit(0, tile)), null, null, null, null));
    List<Integer> indicators = new ArrayList<>(base.input().indicators());
    indicators.set(0, TileCode.type(tile));
    Assert.assertEquals(edited.input().indicators(), indicators);
    assertUnchanged(
        base.input(),
        edited.input(),
        PolicyInput::events,
        PolicyInput::rivers,
        PolicyInput::hand,
        PolicyInput::melds,
        PolicyInput::seats,
        PolicyInput::scores,
        PolicyInput::honba,
        PolicyInput::kyotaku,
        PolicyInput::turn,
        PolicyInput::legal,
        PolicyInput::round);
    List<String> dora = new ArrayList<>(base.table().dora());
    dora.set(0, tile);
    TableState table = base.table();
    Assert.assertEquals(
        edited.table(),
        new TableState(
            table.players(),
            dora,
            table.honba(),
            table.kyotaku(),
            table.remaining(),
            table.activeSeat()));
  }

  @Test
  public void scoreHonbaAndKyotakuEditsReachThePolicyAsShownAtTheDecision() {
    Decision decision = AFTER_RIICHI;
    Answer base = decision.answer(TestEdits.NONE);
    List<Integer> scores = new ArrayList<>(base.input().scores());
    for (int seat = 0; seat < scores.size(); seat++)
      scores.set(seat, scores.get(seat) + (seat == decision.step().actor() ? 3000 : -1000));
    int honba = base.input().honba() + 2;
    int kyotaku = base.input().kyotaku() + 3;
    Answer edited =
        decision.answer(new ScenarioEdits(null, null, null, List.copyOf(scores), honba, kyotaku));
    Assert.assertEquals(edited.input().scores(), scores);
    Assert.assertEquals(edited.input().honba(), honba);
    Assert.assertEquals(edited.input().kyotaku(), kyotaku);
    assertUnchanged(
        base.input(),
        edited.input(),
        PolicyInput::events,
        PolicyInput::rivers,
        PolicyInput::hand,
        PolicyInput::melds,
        PolicyInput::seats,
        PolicyInput::indicators,
        PolicyInput::dora,
        PolicyInput::turn,
        PolicyInput::legal,
        PolicyInput::round,
        PolicyInput::furiten);
    TableState table = base.table();
    List<PlayerState> players = new ArrayList<>();
    for (int seat = 0; seat < table.players().size(); seat++)
      players.add(table.players().get(seat).withScore(scores.get(seat)));
    Assert.assertEquals(
        edited.table(),
        new TableState(
            players, table.dora(), honba, kyotaku, table.remaining(), table.activeSeat()));
    Assert.assertTrue(decision.deposited() > 0, "the decision follows a deposited riichi stick");
    Assert.assertEquals(
        decision
            .answer(new ScenarioEdits(null, null, null, null, null, decision.deposited()))
            .input()
            .kyotaku(),
        decision.deposited());
  }

  @Test
  public void editsThatContradictThePositionAreRejected() {
    Decision decision = AFTER_RIICHI;
    int seat = decision.step().actor();
    int opponent = decision.opponent();
    PlayerState player = decision.player();
    RiverTile own = decision.table().players().get(seat).river().getFirst();
    rejected(
        decision,
        rivers(new RiverEdit(seat, 0, own.tile(), !own.tsumogiri())),
        "edits.rivers[0]",
        "recorded tsumogiri");
    rejected(
        decision,
        rivers(new RiverEdit(opponent, player.river().size(), "E", false)),
        "edits.rivers[0]",
        "no tile at this index");
    int first = uncalled(player).getFirst();
    rejected(
        decision,
        rivers(
            new RiverEdit(opponent, first, "E", false), new RiverEdit(opponent, first, "S", true)),
        "edits.rivers[1]",
        "edited twice");
    for (String code : List.of("0p", "5z", "6sr", "1z", "5", "E5", "5mR"))
      rejected(
          decision,
          rivers(new RiverEdit(opponent, first, code, false)),
          "edits.rivers[0]",
          "invalid tile code");
    var hiddenDora = new DoraEdit(decision.table().dora().size(), "E");
    rejected(
        decision,
        new ScenarioEdits(null, List.of(hiddenDora), null, null, null, null),
        "edits.dora[0]",
        "no dora indicator");
    rejected(
        decision,
        new ScenarioEdits(
            null, List.of(new DoraEdit(0, "E"), new DoraEdit(0, "S")), null, null, null, null),
        "edits.dora[1]",
        "edited twice");
    List<Integer> scores = decision.table().players().stream().map(PlayerState::score).toList();
    rejected(decision, scores(scores, 0, 25050), "edits.scores[0]", "multiples of 100");
    rejected(decision, scores(scores, 2, 4_000_000), "edits.scores[2]", "out of range");
    rejected(
        decision,
        new ScenarioEdits(null, null, null, null, 256, null),
        "edits.honba",
        "honba");
    rejected(
        decision, new ScenarioEdits(null, null, null, null, -1, null), "edits.honba", "honba");
    rejected(
        decision,
        new ScenarioEdits(null, null, null, null, null, 101),
        "edits.kyotaku",
        "kyotaku");
    rejected(
        decision,
        new ScenarioEdits(null, null, null, null, null, decision.deposited() - 1),
        "edits.kyotaku",
        "deposited");
  }

  @Test
  public void editsThatShowTooManyCopiesOrRedFivesAreRejected() {
    Decision decision = AFTER_RIICHI;
    int seat = decision.step().actor();
    int[] visible = visible(decision.table(), seat);
    int kind = 0;
    for (int type = 0; type < Tile.NUM_TILE_TYPES; type++)
      if (visible[type] > visible[kind] && !Tile.canBeAka(type)) kind = type;
    String tile = TileCode.of(kind, false);
    List<RiverEdit> copies = new ArrayList<>();
    for (RiverEdit slot : opponentSlots(decision.table(), seat)) {
      if (copies.size() == Tile.TILES_PER_TYPE + 1 - visible[kind]) break;
      String old = decision.table().players().get(slot.seat()).river().get(slot.index()).tile();
      if (!old.equals(tile)) copies.add(new RiverEdit(slot.seat(), slot.index(), tile, false));
    }
    rejected(
        decision,
        new ScenarioEdits(List.copyOf(copies), null, null, null, null, null),
        "edits.rivers[0]",
        "more than 4 copies of " + tile);
    int five = Tile.M5;
    for (int type : new int[] {Tile.P5, Tile.S5}) if (visible[type] < visible[five]) five = type;
    String plain = TileCode.of(five, false);
    List<RiverEdit> reds = new ArrayList<>();
    for (RiverEdit slot : opponentSlots(decision.table(), seat)) {
      if (reds.size() == 2 - visibleReds(decision.table(), seat, five)) break;
      String old = decision.table().players().get(slot.seat()).river().get(slot.index()).tile();
      if (!old.startsWith(plain))
        reds.add(new RiverEdit(slot.seat(), slot.index(), TileCode.of(five, true), false));
    }
    rejected(
        decision,
        new ScenarioEdits(List.copyOf(reds), null, null, null, null, null),
        "edits.rivers[0]",
        "more than one red " + plain);
  }

  @Test
  public void calledTilesCannotBeEdited() {
    Decision decision = AFTER_CALL;
    int index = calledIndex(decision.player());
    rejected(
        decision,
        rivers(new RiverEdit(decision.opponent(), index, "E", false)),
        "edits.rivers[0]",
        "called tile");
  }

  static void rejected(Decision decision, ScenarioEdits edits, String name, String reason) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> decision.answer(edits));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_EDIT, failure.getMessage());
    Assert.assertEquals(failure.code().status(), 400, failure.getMessage());
    Assert.assertTrue(failure.getMessage().startsWith(name + ":"), failure.getMessage());
    Assert.assertTrue(failure.getMessage().contains(reason), failure.getMessage());
  }

  private static ScenarioEdits rivers(RiverEdit... edits) {
    return new ScenarioEdits(List.of(edits), null, null, null, null, null);
  }

  private static ScenarioEdits scores(List<Integer> shown, int seat, int score) {
    List<Integer> scores = new ArrayList<>(shown);
    scores.set(seat, score);
    return new ScenarioEdits(null, null, null, List.copyOf(scores), null, null);
  }

  /** 変更で変わった出来事がただ 1 つであることを確かめ、その出来事を返す。 */
  private static String onlyChangedEvent(PolicyInput base, PolicyInput edited) {
    Assert.assertEquals(edited.events().size(), base.events().size());
    List<String> changed = new ArrayList<>();
    for (int event = 0; event < base.events().size(); event++)
      if (!base.events().get(event).equals(edited.events().get(event)))
        changed.add(edited.events().get(event));
    Assert.assertEquals(changed.size(), 1, changed.toString());
    return changed.getFirst();
  }

  @SafeVarargs
  static void assertUnchanged(
      PolicyInput base, PolicyInput edited, Function<PolicyInput, ?>... values) {
    for (Function<PolicyInput, ?> value : values)
      Assert.assertEquals(value.apply(edited), value.apply(base));
  }

  static String field(String event, int index) {
    return event.split(" ")[index];
  }

  private static TableState withRiverTile(TableState table, RiverEdit edit, boolean riichi) {
    List<PlayerState> players = new ArrayList<>(table.players());
    players.set(
        edit.seat(),
        players
            .get(edit.seat())
            .withRiverTile(
                edit.index(), new RiverTile(edit.tile(), edit.tsumogiri(), riichi, false)));
    return new TableState(
        players,
        table.dora(),
        table.honba(),
        table.kyotaku(),
        table.remaining(),
        table.activeSeat());
  }

  /** 判断の席に見える牌の数が少なく、元の牌と違う種類の牌。 */
  static String rareTile(TableState table, int seat, String old) {
    int[] visible = visible(table, seat);
    for (String tile : RARE_TILES)
      if (visible[TileCode.type(tile)] <= 2 && TileCode.type(tile) != TileCode.type(old))
        return tile;
    throw new AssertionError("No rare tile in the fixture position.");
  }

  static int[] visible(TableState table, int seat) {
    int[] copies = new int[Tile.NUM_TILE_TYPES];
    for (String tile : visibleTiles(table, seat)) copies[TileCode.type(tile)]++;
    return copies;
  }

  private static int visibleReds(TableState table, int seat, int type) {
    return (int)
        visibleTiles(table, seat).stream()
            .filter(tile -> TileCode.red(tile) && TileCode.type(tile) == type)
            .count();
  }

  private static List<String> visibleTiles(TableState table, int seat) {
    List<String> tiles = new ArrayList<>(table.players().get(seat).hand());
    if (table.players().get(seat).draw() != null) tiles.add(table.players().get(seat).draw());
    for (PlayerState player : table.players()) {
      for (RiverTile tile : player.river()) if (!tile.called()) tiles.add(tile.tile());
      for (Meld meld : player.melds()) tiles.addAll(meld.tiles());
    }
    tiles.addAll(table.dora());
    return tiles;
  }

  /** 他家の鳴かれていない河の牌の位置。 */
  private static List<RiverEdit> opponentSlots(TableState table, int seat) {
    List<RiverEdit> slots = new ArrayList<>();
    for (int opponent = 0; opponent < table.players().size(); opponent++)
      if (opponent != seat)
        for (int index : uncalled(table.players().get(opponent)))
          slots.add(new RiverEdit(opponent, index, "", false));
    return slots;
  }

  private static List<Integer> uncalled(PlayerState player) {
    List<Integer> indices = new ArrayList<>();
    for (int index = 0; index < player.river().size(); index++)
      if (!player.river().get(index).called()) indices.add(index);
    return indices;
  }

  private static int calledIndex(PlayerState player) {
    for (int index = 0; index < player.river().size(); index++)
      if (player.river().get(index).called()) return index;
    return -1;
  }

  /** 記録された手が打牌の判断（手番か鳴いた後）。 */
  static boolean discards(Format6.Step step) {
    return step.chosenType().equals("DAHAI");
  }

  private static int opponent(TableState table, Format6.Step step, Predicate<PlayerState> wanted) {
    for (int seat = 0; seat < table.players().size(); seat++)
      if (seat != step.actor() && wanted.test(table.players().get(seat))) return seat;
    return -1;
  }

  record Answer(PolicyInput input, TableState table) {}

  /**
   * 試験の判断と、変える他家。
   *
   * @param deposited 判断までにこの局で出たリーチの棒
   */
  record Decision(
      LoadedRecord record,
      int round,
      Format6.Step step,
      TableState table,
      int opponent,
      int deposited) {
    PlayerState player() {
      return table.players().get(opponent);
    }

    /** 変更を当てた方策の入力と、変更を写した判断の局面。 */
    Answer answer(ScenarioEdits edits) {
      var policy = new RecordingPolicy();
      ScenarioEngine.evaluate(
          record, new ScenarioQuery(round, step.eventIndex(), step.actor(), edits, null), policy);
      Assert.assertEquals(policy.inputs.size(), 1);
      return new Answer(policy.inputs.getFirst(), edited(edits));
    }

    /** RoundRewriter が写した局面（もしもの答えは鳴く時だけ局面を返すので、場所を探して写し直す）。 */
    private TableState edited(ScenarioEdits edits) {
      return RoundRewriter.apply(
              record.replay().events(),
              DecisionLocator.find(record, round, step.eventIndex(), step.actor()),
              edits)
          .table();
    }

    /**
     * 牌譜 kakan-a の判断のうち、変える他家が見つかる最初の判断。
     *
     * @param wanted 判断の種類
     * @param opponent 判断の局面から変える他家を選ぶ（いなければ -1）
     */
    static Decision find(
        Predicate<Format6.Step> wanted, ToIntBiFunction<TableState, Format6.Step> opponent) {
      LoadedRecord record = GoldenRecords.read("kakan-a");
      Format6.Document document = Format6.analyze(record, new RecordingPolicy());
      List<StartKyoku> starts =
          record.replay().events().stream()
              .filter(StartKyoku.class::isInstance)
              .map(StartKyoku.class::cast)
              .toList();
      for (Format6.Located located : document.decisions()) {
        Format6.Step step = located.step();
        if (!wanted.test(step)) continue;
        TableState table = step.table();
        int seat = opponent.applyAsInt(table, step);
        if (seat >= 0)
          return new Decision(
              record,
              located.round(),
              step,
              table,
              seat,
              table.kyotaku() - starts.get(located.round()).kyotaku());
      }
      throw new AssertionError("No such decision in the fixture.");
    }
  }
}
