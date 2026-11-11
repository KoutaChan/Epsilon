package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.dora;
import static com.epsilon.reviewer.engine.TestEdits.rivers;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.record.TileCopies;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.RoundRewriter;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * もしもの変更の種類ごとに、O のうちその変更に当たる値だけが変わることを確かめる。ほかの値も、変更を当てた局面のほかの部分も変わらない。
 * 局面と合わない変更は、決まった順に確かめ、決まったメッセージで断る。判断には主に、天鳳の牌譜で他家が立直した後の打牌の判断を使う。
 */
public class ScenarioEditTest {
  /** 37 種の牌の記号。 */
  static final List<String> TILES = Arrays.stream(Face.values()).map(Face::mjai).toList();

  private static final String[] RARE_TILES = {"E", "S", "W", "N", "P", "F", "C", "1m", "9m", "1p"};

  /** 立直していない席の打牌の判断。立直した他家が宣言牌の後にも打牌していて（立直棒を出していて）、その河に鳴かれていない牌が 3 枚以上ある。 */
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

  /** 他家の打牌にポンできる応答の判断。変える他家は、その牌を打った席。 */
  private static final Decision PON_OFFER =
      Decision.find(step -> step.offers(Action.Type.PON), (table, step) -> table.activeSeat());

  /** 判断する席が応答する他家の打牌を、その席が鳴けない字牌に替える。候補は見送りだけになり、方策は呼ばれない。 */
  @Test
  public void editingTheAnsweredDiscardRegeneratesTheResponses() {
    Decision decision = PON_OFFER;
    int index = decision.player().river().size() - 1;
    RiverTile old = decision.player().river().get(index);
    PlayerState own = decision.table().players().get(decision.step().actor());
    Map<String, Integer> visible = visible(decision.table(), decision.step().actor());
    String tile = null;
    for (String honor : List.of("E", "S", "W", "N", "P", "F", "C"))
      if (tile == null
          && !own.hand().contains(honor)
          && !honor.equals(old.tile())
          && visible.getOrDefault(honor, 0) < copies(honor)) tile = honor;
    var edit = new RiverEdit(decision.opponent(), index, tile, old.tsumogiri());
    var policy = new RecordingPolicy();
    ScenarioAnswer answer =
        Scenarios.evaluate(
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
    Assert.assertEquals(Action.of(answer.decision().actionIds()[0]).type(), Action.Type.PASS);
    Assert.assertEquals(answer.decision().chosen() == 0, passed);
    Assert.assertEquals(answer.decision().values()[0], 0f);
    Assert.assertTrue(policy.inputs.isEmpty(), "a pass-only response does not call the policy");
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
    Assert.assertEquals(field(event, 1), Integer.toString(decision.opponent()));
    Assert.assertEquals(field(event, 2), Integer.toString(TileCode.kind(tile)));
    assertUnchanged(
        base.input(),
        edited.input(),
        ObservedInput::hand,
        ObservedInput::riichi,
        ObservedInput::scores,
        ObservedInput::honba,
        ObservedInput::kyotaku,
        ObservedInput::indicators,
        ObservedInput::legal);
    Assert.assertEquals(
        edited.table(), withRiverTile(base.table(), edit, old.riichi()), "table outside the edit");
  }

  /** 自分の河の牌を替えると、その牌が手に入った所（配牌かツモ）で替わる。そのため、判断の時点の手牌は変わらない。 */
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
      Assert.assertEquals(field(event, 2), Integer.toString(TileCode.kind(edit.tile())));
      assertUnchanged(
          base.input(),
          edited.input(),
          ObservedInput::hand,
          ObservedInput::handAka,
          ObservedInput::scores,
          ObservedInput::indicators,
          ObservedInput::legal);
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
        ObservedInput::hand,
        ObservedInput::riichi,
        ObservedInput::scores,
        ObservedInput::indicators,
        ObservedInput::legal,
        ObservedInput::furiten);
    Assert.assertEquals(edited.table(), withRiverTile(base.table(), edit, old.riichi()));
  }

  @Test
  public void doraEditChangesOnlyTheIndicator() {
    Decision decision = AFTER_RIICHI;
    String tile =
        rareTile(decision.table(), decision.step().actor(), decision.table().dora().getFirst());
    Answer base = decision.answer(TestEdits.NONE);
    Answer edited = decision.answer(dora(new DoraEdit(0, tile)));
    List<Integer> indicators = new ArrayList<>(base.input().indicators());
    indicators.set(0, TileCode.kind(tile));
    Assert.assertEquals(edited.input().indicators(), indicators);
    assertUnchanged(
        base.input(),
        edited.input(),
        ObservedInput::events,
        ObservedInput::hand,
        ObservedInput::riichi,
        ObservedInput::scores,
        ObservedInput::honba,
        ObservedInput::kyotaku,
        ObservedInput::legal);
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
        ObservedInput::events,
        ObservedInput::hand,
        ObservedInput::riichi,
        ObservedInput::indicators,
        ObservedInput::legal,
        ObservedInput::furiten);
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

  /** 確かめる順は河 → ドラ → 手牌 → 点 → 本場 → 供託。前のフィールドの誤りを先に名指す。 */
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
        rivers(new RiverEdit(4, 0, "E", false)),
        "edits.rivers[0]",
        "seat must be between 0 and 3.");
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
    rejected(
        decision,
        dora(new DoraEdit(decision.table().dora().size(), "E")),
        "edits.dora[0]",
        "no dora indicator");
    rejected(
        decision,
        dora(new DoraEdit(0, "E"), new DoraEdit(0, "S")),
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
        "and 100.");
    rejected(
        decision,
        new ScenarioEdits(null, null, null, null, null, decision.deposited() - 1),
        "edits.kyotaku",
        "deposited");
    // 河の誤りは、後のフィールド（ドラ・点）の誤りより先に名指される。
    rejected(
        decision,
        new ScenarioEdits(
            List.of(new RiverEdit(opponent, player.river().size(), "E", false)),
            List.of(new DoraEdit(decision.table().dora().size(), "E")),
            null,
            scores(scores, 0, 25050).scores(),
            null,
            null),
        "edits.rivers[0]",
        "no tile at this index");
    ReviewerException count =
        Assert.expectThrows(
            ReviewerException.class,
            () -> decision.answer(new ScenarioEdits(null, null, null, scores.subList(0, 3), null, null)));
    Assert.assertEquals(count.code(), ErrorCode.INVALID_REQUEST);
    Assert.assertEquals(count.getMessage(), "edits.scores must contain four scores.");
  }

  /**
   * 写しの数（4・3・1 枚）ごとに、その数の牌のうち最も多く見えている牌を選び、写しの数より 1 枚多く見えるまで他家の河に置く。この変更は、
   * 最初の変更を名指して断られる。
   */
  @Test
  public void editsThatShowMoreCopiesThanTheSetAreRejected() {
    Decision decision = AFTER_RIICHI;
    int seat = decision.step().actor();
    Map<String, Integer> visible = visible(decision.table(), seat);
    for (int count : new int[] {4, 3, 1}) {
      String tile =
          TILES.stream()
              .filter(code -> copies(code) == count)
              .max(Comparator.comparingInt(code -> visible.getOrDefault(code, 0)))
              .orElseThrow();
      int needed = count + 1 - visible.getOrDefault(tile, 0);
      List<RiverEdit> edits = new ArrayList<>();
      for (RiverEdit slot : opponentSlots(decision.table(), seat)) {
        if (edits.size() == needed) break;
        String old = decision.table().players().get(slot.seat()).river().get(slot.index()).tile();
        if (!old.equals(tile)) edits.add(new RiverEdit(slot.seat(), slot.index(), tile, false));
      }
      Assert.assertEquals(edits.size(), needed, "opponent river tiles to change to " + tile);
      rejected(
          decision,
          new ScenarioEdits(edits, null, null, null, null, null),
          "edits.rivers[0]",
          "more copies of " + tile + " would be visible than the " + count + " in the set.");
    }
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

  private static ScenarioEdits scores(List<Integer> shown, int seat, int score) {
    List<Integer> scores = new ArrayList<>(shown);
    scores.set(seat, score);
    return new ScenarioEdits(null, null, null, List.copyOf(scores), null, null);
  }

  /** 変更で変わったイベントがただ 1 つであることを確かめ、そのイベントを返す。 */
  private static String onlyChangedEvent(ObservedInput base, ObservedInput edited) {
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
      ObservedInput base, ObservedInput edited, Function<ObservedInput, ?>... values) {
    for (Function<ObservedInput, ?> value : values)
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

  /** 判断する席に見えている枚数が少なく、元の牌と種類が違う牌。 */
  static String rareTile(TableState table, int seat, String old) {
    Map<String, Integer> visible = visible(table, seat);
    for (String tile : RARE_TILES)
      if (visible.getOrDefault(tile, 0) <= 2 && TileCode.kind(tile) != TileCode.kind(old))
        return tile;
    throw new AssertionError("No rare tile in the fixture position.");
  }

  /** 牌の記号ごとの写しの数（赤の 5 は 1 枚、赤でない 5 は 3 枚、ほかは 4 枚）。 */
  static int copies(String tile) {
    return TileCopies.copies(Variant.TENHOU_4P, TileCode.face(tile));
  }

  /** 判断する席に見える牌を、牌の記号ごとに数える。 */
  static Map<String, Integer> visible(TableState table, int seat) {
    Map<String, Integer> counts = new HashMap<>();
    for (String tile : visibleTiles(table, seat)) counts.merge(tile, 1, Integer::sum);
    return counts;
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

  /** 他家の河で、鳴かれていない牌の位置。 */
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

  /** 記録された手が打牌か。手番の打牌と、鳴いた後の打牌の両方を数える。 */
  static boolean discards(Format6.Step step) {
    return step.chosenType().equals("DAHAI");
  }

  private static int opponent(TableState table, Format6.Step step, Predicate<PlayerState> wanted) {
    for (int seat = 0; seat < table.players().size(); seat++)
      if (seat != step.actor() && wanted.test(table.players().get(seat))) return seat;
    return -1;
  }

  record Answer(ObservedInput input, TableState table) {}

  /**
   * テストに使う判断と、変える他家。
   *
   * @param deposited 判断までにこの局で出た立直棒の本数
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

    /** 変更を当てたときに方策が受け取る O と、変更を当てた判断の局面。 */
    Answer answer(ScenarioEdits edits) {
      var policy = new RecordingPolicy();
      Scenarios.evaluate(
          record, new ScenarioQuery(round, step.eventIndex(), step.actor(), edits, null), policy);
      Assert.assertEquals(policy.inputs.size(), 1);
      return new Answer(policy.input(0), edited(edits));
    }

    /** RoundRewriter で変更を当てた局面。もしもの答えが局面を返すのは鳴くときだけなので、判断の場所を探して変更を当て直す。 */
    private TableState edited(ScenarioEdits edits) {
      return RoundRewriter.apply(
              record.rounds().get(round),
              Scenarios.find(record, round, step.eventIndex(), step.actor()),
              edits)
          .table();
    }

    /**
     * エンジンで打った四人打ちの半荘（種 0）の判断のうち、変える他家が見つかる最初の判断を返す。
     *
     * @param wanted 使う判断を選ぶ条件
     * @param opponent 判断の局面から変える他家を選ぶ関数（いなければ -1 を返す）
     */
    static Decision find(
        Predicate<Format6.Step> wanted, ToIntBiFunction<TableState, Format6.Step> opponent) {
      LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
      Format6.Document document = Format6.analyze(record, new RecordingPolicy());
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
              table.kyotaku() - record.rounds().get(located.round()).start().kyotaku());
      }
      throw new AssertionError("No such decision in the fixture.");
    }
  }
}
