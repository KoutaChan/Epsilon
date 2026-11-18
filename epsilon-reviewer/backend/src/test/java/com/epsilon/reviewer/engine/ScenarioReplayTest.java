package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.dora;
import static com.epsilon.reviewer.engine.TestEdits.hand;
import static com.epsilon.reviewer.engine.TestEdits.merge;
import static com.epsilon.reviewer.engine.TestEdits.rivers;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.record.RecordEvent.Discard;
import com.epsilon.mahjong.record.RecordEvent.Draw;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.TileCopies;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.DoraEdit;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.LocatedDecision;
import com.epsilon.reviewer.engine.scenario.RewrittenRound;
import com.epsilon.reviewer.engine.scenario.RiverEdit;
import com.epsilon.reviewer.engine.scenario.RoundRewriter;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.MeldView;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableView;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.function.IntUnaryOperator;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * もしもの変更の種類ごとに、O のうちその変更に当たる値だけが変わることを確かめる。ほかの値も、変更を当てた局面のほかの部分も変わらない。
 * 局面と合わない変更は、決まった順に確かめ、決まったメッセージで断る。判断には主に、天鳳の牌譜で他家が立直した後の打牌の判断を使う。
 */
public class ScenarioReplayTest {
  /** 37 種の牌の記号。 */
  static final List<String> TILES = Arrays.stream(Face.values()).map(Face::mjai).toList();

  private static final String[] RARE_TILES = {"E", "S", "W", "N", "P", "F", "C", "1m", "9m", "1p"};

  /** 立直していない席の打牌の判断。立直した他家が宣言牌の後にも打牌していて（立直棒を出していて）、その河に鳴かれていない牌が 3 枚以上ある。 */
  private static final Decision AFTER_RIICHI =
      Decision.find(
          ScenarioReplayTest::discards,
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
          ScenarioReplayTest::discards,
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
    TableView table = base.table();
    Assert.assertEquals(
        edited.table(),
        new TableView(
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
    TableView table = base.table();
    List<PlayerState> players = new ArrayList<>();
    for (int seat = 0; seat < table.players().size(); seat++)
      players.add(table.players().get(seat).withScore(scores.get(seat)));
    Assert.assertEquals(
        edited.table(),
        new TableView(
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

  private static TableView withRiverTile(TableView table, RiverEdit edit, boolean riichi) {
    List<PlayerState> players = new ArrayList<>(table.players());
    players.set(
        edit.seat(),
        players
            .get(edit.seat())
            .withRiverTile(
                edit.index(), new RiverTile(edit.tile(), edit.tsumogiri(), riichi, false)));
    return new TableView(
        players,
        table.dora(),
        table.honba(),
        table.kyotaku(),
        table.remaining(),
        table.activeSeat());
  }

  /** 判断する席に見えている枚数が少なく、元の牌と種類が違う牌。 */
  static String rareTile(TableView table, int seat, String old) {
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
  static Map<String, Integer> visible(TableView table, int seat) {
    Map<String, Integer> counts = new HashMap<>();
    for (String tile : visibleTiles(table, seat)) counts.merge(tile, 1, Integer::sum);
    return counts;
  }

  private static List<String> visibleTiles(TableView table, int seat) {
    List<String> tiles = new ArrayList<>(table.players().get(seat).hand());
    if (table.players().get(seat).draw() != null) tiles.add(table.players().get(seat).draw());
    for (PlayerState player : table.players()) {
      for (RiverTile tile : player.river()) if (!tile.called()) tiles.add(tile.tile());
      for (MeldView meld : player.melds()) tiles.addAll(meld.tiles());
      for (int index = 0; index < player.nuki(); index++) tiles.add("N");
    }
    tiles.addAll(table.dora());
    return tiles;
  }

  /** 他家の河で、鳴かれていない牌の位置。 */
  private static List<RiverEdit> opponentSlots(TableView table, int seat) {
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

  private static int opponent(TableView table, Format6.Step step, Predicate<PlayerState> wanted) {
    for (int seat = 0; seat < table.players().size(); seat++)
      if (seat != step.actor() && wanted.test(table.players().get(seat))) return seat;
    return -1;
  }

  record Answer(ObservedInput input, TableView table) {}

  /**
   * テストに使う判断と、変える他家。
   *
   * @param deposited 判断までにこの局で出た立直棒の本数
   */
  record Decision(
      LoadedRecord record,
      int round,
      Format6.Step step,
      TableView table,
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
    private TableView edited(ScenarioEdits edits) {
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
        Predicate<Format6.Step> wanted, ToIntBiFunction<TableView, Format6.Step> opponent) {
      LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
      Format6.Document document = Format6.analyze(record, new RecordingPolicy());
      for (Format6.Located located : document.decisions()) {
        Format6.Step step = located.step();
        if (!wanted.test(step)) continue;
        TableView table = step.table();
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

  /** 立直の宣言牌にチーかポンができる応答の判断がある半荘の、乱数の種。 */
  private static final long RIICHI_TILE_SEED = 11;

  @Test
  public void recordedCallMatchesTheReviewDiscardAfterTheCall() {
    int compared = 0;
    for (Map.Entry<String, LoadedRecord> entry : PlayedRecords.fourPlayer().entrySet()) {
      compared += compareRecordedCalls(entry.getKey(), entry.getValue());
      compared += compareRecordedCalls(entry.getKey() + " mjai", PlayedRecords.mjai(entry.getValue()));
    }
    Assert.assertTrue(compared > 0, "The fixtures have no recorded chi or pon.");
  }

  @Test
  public void callIsOnlyAChiOrPonAtAResponse() {
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
    List<CallDecision> decisions = decisions(record);
    CallDecision turn = first(decisions, EngineDecisionKind.TURN);
    CallDecision response = first(decisions, EngineDecisionKind.RESPONSE);
    int pass = Action.code(Action.Type.PASS);
    assertRefused(record, turn, turn.step().actionId(0), "call: the decision is not a response");
    assertRefused(record, response, pass, "call: only a chi or a pon");
    assertRefused(record, response, Integer.MAX_VALUE, "call: the call is not a legal response");
  }

  /**
   * 立直の宣言牌を鳴くと、鳴いた時に立直が成立し、供託が 1 本増えて、宣言した席の点が 1000 点減る。鳴いた後の供託か点が O に書ける範囲
   * （供託は 100 本まで、点は -3,276,800 点から）を外れる変更は断る。範囲に収まる変更は、鳴いた後の値で評価する。
   */
  @Test
  public void callingARiichiTileRefusesEditsTheDepositPutsOutOfRange() {
    int maxKyotaku = 100;
    int minScore = -3_276_800;
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, RIICHI_TILE_SEED);
    for (CallDecision response : decisions(record)) {
      TableView shown = response.step().table();
      int declarer = shown.activeSeat();
      List<RiverTile> river = shown.players().get(declarer).river();
      int call = call(response.step());
      if (response.kind() != EngineDecisionKind.RESPONSE || call < 0 || !river.getLast().riichi())
        continue;
      var policy = new RecordingPolicy();
      ScenarioAnswer answer =
          Scenarios.evaluate(record, query(response, kyotaku(maxKyotaku - 1), call), policy);
      Assert.assertEquals(answer.table().kyotaku(), maxKyotaku);
      Assert.assertEquals(policy.input(0).kyotaku(), maxKyotaku);
      assertRefused(
          record, response, kyotaku(maxKyotaku), call, "call: the riichi deposit at the call");
      Scenarios.evaluate(record, query(response, kyotaku(maxKyotaku), null), new RecordingPolicy());
      List<Integer> scores =
          new ArrayList<>(shown.players().stream().map(PlayerState::score).toList());
      scores.set(declarer, minScore + 1000);
      policy = new RecordingPolicy();
      answer = Scenarios.evaluate(record, query(response, scores(scores), call), policy);
      Assert.assertEquals(answer.table().players().get(declarer).score(), minScore);
      Assert.assertEquals(policy.input(0).scores().get(declarer).intValue(), minScore);
      scores.set(declarer, minScore);
      assertRefused(
          record, response, scores(scores), call, "call: the riichi deposit at the call");
      Scenarios.evaluate(record, query(response, scores(scores), null), new RecordingPolicy());
      return;
    }
    Assert.fail("The fixture has no chi or pon offered on a riichi declaration tile.");
  }

  /** 牌譜で鳴いた応答の判断ごとに、同じ鳴きをしたもしもと、通常の解析での次の判断（鳴いた後の打牌）を比べる。 */
  private static int compareRecordedCalls(String name, LoadedRecord record) {
    List<CallDecision> decisions = decisions(record);
    int compared = 0;
    for (int i = 0; i < decisions.size(); i++) {
      CallDecision response = decisions.get(i);
      if (response.kind() != EngineDecisionKind.RESPONSE) continue;
      String recorded = response.step().chosenType();
      if (!recorded.equals("CHI") && !recorded.equals("PON")) continue;
      // 他家のロンが優先されると鳴きは成り立たないので、次の判断が鳴いた後の打牌のときだけ比べる。
      CallDecision discard = next(decisions, i);
      if (discard == null || discard.kind() != EngineDecisionKind.POST_CALL_DAHAI) continue;
      String where =
          name
              + " round "
              + response.round()
              + " event "
              + response.step().eventIndex()
              + " seat "
              + response.step().actor();
      var policy = new RecordingPolicy();
      int call = response.step().actionId(response.step().chosen());
      ScenarioAnswer answer = Scenarios.evaluate(record, query(response, call), policy);
      Assert.assertEquals(policy.inputs.size(), 1, where);
      Assert.assertEquals(policy.input(0), discard.input(), where);
      JsonObject expected = discard.step().candidateSet();
      expected.remove("chosen");
      Assert.assertEquals(Format6.candidateSet(answer), expected, where);
      Assert.assertEquals(answer.table(), discard.step().table(), where);
      compared++;
    }
    return compared;
  }

  /** 同じ局で同じ席が次に行う判断。無ければ null。 */
  private static CallDecision next(List<CallDecision> decisions, int index) {
    CallDecision from = decisions.get(index);
    for (int i = index + 1; i < decisions.size(); i++) {
      CallDecision later = decisions.get(i);
      if (later.round() != from.round()) return null;
      if (later.step().actor() == from.step().actor()) return later;
    }
    return null;
  }

  private static void assertRefused(
      LoadedRecord record, CallDecision decision, int call, String message) {
    assertRefused(record, decision, TestEdits.NONE, call, message);
  }

  private static void assertRefused(
      LoadedRecord record, CallDecision decision, ScenarioEdits edits, int call, String message) {
    ReviewerException refused =
        Assert.expectThrows(
            ReviewerException.class,
            () -> Scenarios.evaluate(record, query(decision, edits, call), new RecordingPolicy()));
    Assert.assertEquals(refused.code(), ErrorCode.INVALID_EDIT);
    Assert.assertTrue(refused.getMessage().startsWith(message), refused.getMessage());
  }

  private static ScenarioQuery query(CallDecision decision, int call) {
    return query(decision, TestEdits.NONE, call);
  }

  private static ScenarioQuery query(CallDecision decision, ScenarioEdits edits, Integer call) {
    return new ScenarioQuery(
        decision.round(), decision.step().eventIndex(), decision.step().actor(), edits, call);
  }

  private static ScenarioEdits kyotaku(int kyotaku) {
    return new ScenarioEdits(null, null, null, null, null, kyotaku);
  }

  private static ScenarioEdits scores(List<Integer> scores) {
    return new ScenarioEdits(null, null, null, List.copyOf(scores), null, null);
  }

  /** 候補にある最初のチーかポンの行動の番号。無ければ -1。 */
  private static int call(Format6.Step step) {
    for (int i = 0; i < step.candidates().size(); i++)
      if (step.type(i) == Action.Type.CHI || step.type(i) == Action.Type.PON)
        return step.actionId(i);
    return -1;
  }

  private static CallDecision first(List<CallDecision> decisions, EngineDecisionKind kind) {
    return decisions.stream().filter(d -> d.kind() == kind).findFirst().orElseThrow();
  }

  /** 通常の解析の判断を、方策が受け取った O のフィールドと組にして、順に並べる。 */
  private static List<CallDecision> decisions(LoadedRecord record) {
    var policy = new RecordingPolicy();
    Format6.Document document = Format6.analyze(record, policy);
    List<CallDecision> decisions = new ArrayList<>();
    for (Format6.Located located : document.decisions())
      decisions.add(
          new CallDecision(located.round(), located.step(), policy.input(decisions.size())));
    Assert.assertEquals(decisions.size(), policy.inputs.size());
    return decisions;
  }

  private record CallDecision(int round, Format6.Step step, ObservedInput input) {
    EngineDecisionKind kind() {
      return input.kind();
    }
  }

  @Test
  public void unchangedPositionGivesThePolicyTheReviewInputAtEveryDecision() {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    PlayedRecords.fourPlayer()
        .forEach(
            (name, record) -> {
              records.put(name, record);
              records.put(name + " mjai", PlayedRecords.mjai(record));
            });
    records.put("chankan round", TestRounds.chankanRound());
    Set<EngineDecisionKind> kinds = EnumSet.noneOf(EngineDecisionKind.class);
    records.forEach((name, record) -> compareEveryDecision(name, record, kinds));
    Assert.assertEquals(
        kinds,
        EnumSet.of(
            EngineDecisionKind.TURN,
            EngineDecisionKind.RESPONSE,
            EngineDecisionKind.CHANKAN,
            EngineDecisionKind.POST_CALL_DAHAI));
  }

  private static void compareEveryDecision(
      String name, LoadedRecord record, Set<EngineDecisionKind> kinds) {
    var review = new RecordingPolicy();
    Format6.Document document = Format6.analyze(record, review);
    int decision = 0;
    for (Format6.Located located : document.decisions()) {
      Format6.Step step = located.step();
      String where =
          name
              + " round "
              + located.round()
              + " event "
              + step.eventIndex()
              + " seat "
              + step.actor();
      var scenario = new RecordingPolicy();
      ScenarioAnswer answer =
          Scenarios.evaluate(
              record,
              new ScenarioQuery(
                  located.round(), step.eventIndex(), step.actor(), TestEdits.NONE, null),
              scenario);
      Assert.assertEquals(scenario.inputs.size(), 1, where);
      Assert.assertEquals(
          scenario.input(0), review.input(decision), where + ": the fields of the O");
      Assert.assertEquals(scenario.inputs.getFirst(), review.inputs.get(decision), where);
      Assert.assertEquals(Format6.candidateSet(answer), step.candidateSet(), where);
      Assert.assertNull(answer.table(), where);
      Assert.assertEquals(
          Scenarios.find(record, located.round(), step.eventIndex(), step.actor()).table(),
          step.table(),
          where);
      kinds.add(review.input(decision).kind());
      decision++;
    }
    Assert.assertEquals(decision, review.inputs.size(), name);
  }

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
    Map<String, Integer> visible = visible(LOCATED.table(), SEAT);
    List<Place> places = places(LOCATED.table());
    for (Place place : places)
      for (String tile : TILES) {
        ScenarioEdits edits = place.editTo().apply(tile);
        if (!tile.equals(place.tile())
            && visible.getOrDefault(tile, 0) >= copies(tile))
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
            + copies(tile)
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
  private static List<Place> places(TableView table) {
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
