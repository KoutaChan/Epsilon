package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.GameState;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.reviewer.engine.scenario.VisibleTileCount.Placement;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * もしもの変更を確かめ、判断のある局の出来事と判断の表示の局面へ写す。河 → ドラ → 手牌 → 点 → 本場 → 供託 → 見える牌の数の順に確かめる。
 *
 * <p>変更は判断の時点の表示で指す。河はその席の打牌の順、ドラは開いた順（0 は局の始め、1 からは槓ドラ）、手牌は hand[] の位置（hand.length
 * はツモ牌）。点と供託は判断の時点の値で受け取り、判断までのリーチの支払いを保つよう局の始めの値を同じ差だけ動かす。
 *
 * <p>牌を替えた河と手牌の物理牌は、見える牌の数を確かめてから割り当てる。替えた牌の物理牌を全て空け、新しい牌それぞれにその記号の空いている
 * 写しを割り当てて、元の物理牌を書いた所（打牌と、判断の席の牌が手に入った配牌の位置かツモ）へ書く。判断より前の出来事で判断の席に見える牌
 * （その席の配牌とツモ、全ての打牌、副露と槓の牌）の物理牌は、どれも判断の時点の表示に見える牌なので（ドラ表示は物理牌を持たない）、見える牌の
 * 数が通れば写しは足りる。
 */
public final class RoundRewriter {
  private final LocatedDecision decision;
  private final int seat;
  private final List<ReplayEvent> events;
  private final StartKyoku start;

  /** 配牌の写し。判断の席の行を書き換える。 */
  private final int[][] deal;

  private final List<PlayerState> players;
  private final List<String> dora;

  /** 判断の席の手牌。判断の hand[] の位置のまま書き換え、最後に並べる。 */
  private final List<String> hand;

  private String draw;

  /** 判断の席の伏せた手の物理牌（手に入った順）。手牌の変更が替えた牌は外す。 */
  private final List<Integer> concealed;

  /** 判断より前の出来事で判断の席に見える牌の物理牌（他家の伏せた配牌とツモは数えない）。 */
  private final TileCopies copies = new TileCopies();

  /** 判断の席の手に入った物理牌ごとの、入った所。 */
  private final Map<Integer, Slot> entries = new HashMap<>();

  /** 席ごとの、判断より前の打牌の出来事の位置（河の順）。 */
  private final List<List<Integer>> discards = new ArrayList<>();

  /** 判断より前に槓ドラを開いた出来事の位置（開いた順）。 */
  private final List<Integer> reveals = new ArrayList<>();

  /** 牌を替えた変更（当てた順）。 */
  private final List<Placement> placements = new ArrayList<>();

  private final List<Replacement> replacements = new ArrayList<>();

  private RoundRewriter(List<ReplayEvent> record, LocatedDecision decision) {
    this.decision = decision;
    seat = decision.seat();
    events = new ArrayList<>(record.subList(decision.start(), decision.end()));
    start = (StartKyoku) events.getFirst();
    deal = start.initialHandPhysicalTileIds().clone();
    deal[seat] = deal[seat].clone();
    players = new ArrayList<>(decision.table().players());
    dora = new ArrayList<>(decision.table().dora());
    hand = new ArrayList<>(players.get(seat).hand());
    draw = players.get(seat).draw();
    concealed = new ArrayList<>(decision.concealed());
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) discards.add(new ArrayList<>());
    for (int position = 0; position < decision.target(); position++) scan(position);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++)
      if (discards.get(player).size() != players.get(player).river().size())
        throw new IllegalStateException("The river of seat " + player + " differs from the round");
  }

  /** @param record 牌譜の全ての出来事 */
  public static RewrittenRound apply(
      List<ReplayEvent> record, LocatedDecision decision, ScenarioEdits edits) {
    var round = new RoundRewriter(record, decision);
    each(edits.rivers(), "edits.rivers", edit -> List.of(edit.seat(), edit.index()), round::river);
    each(edits.dora(), "edits.dora", DoraEdit::index, round::dora);
    each(edits.hand(), "edits.hand", HandEdit::index, round::hand);
    return round.finish(edits);
  }

  /** 変更を順に当てる。同じ所の 2 つ目の変更は断る。 */
  private static <E> void each(
      List<E> edits, String field, Function<E, Object> place, BiConsumer<E, String> apply) {
    Set<Object> places = new HashSet<>();
    for (int i = 0; i < edits.size(); i++) {
      E edit = edits.get(i);
      String name = field + "[" + i + "]";
      if (!places.add(place.apply(edit)))
        throw invalidEdit(name + ": the same tile is edited twice.");
      apply.accept(edit, name);
    }
  }

  /**
   * 判断より前の出来事を読み、打牌と槓ドラの位置、判断の席の牌が手に入った所、判断の席に見える牌の物理牌を覚える。鳴いた牌は打牌の物理牌なので、
   * 副露は手から出した牌だけを数える。
   */
  private void scan(int position) {
    switch (events.get(position)) {
      case StartKyoku ignored -> {
        for (int index = 0; index < deal[seat].length; index++)
          enter(deal[seat][index], new Slot.Dealt(index));
      }
      case Tsumo drawn when drawn.actor() == seat ->
          enter(drawn.physicalTileId(), new Slot.Drawn(position));
      case Dahai discard -> {
        discards.get(discard.actor()).add(position);
        copies.use(discard.physicalTileId());
      }
      case Dora ignored -> reveals.add(position);
      case Chi call -> copies.use(call.consumedPhysicalTileIds());
      case Pon call -> copies.use(call.consumedPhysicalTileIds());
      case Daiminkan call -> copies.use(call.consumedPhysicalTileIds());
      case Ankan kan -> copies.use(kan.consumedPhysicalTileIds());
      case Kakan kan -> copies.use(kan.addedPhysicalTileId());
      default -> {}
    }
  }

  private void enter(int physicalTileId, Slot slot) {
    copies.use(physicalTileId);
    entries.put(physicalTileId, slot);
  }

  /**
   * 河の 1 枚（鳴かれた牌は変えない）。他家の打牌は牌とツモ切りの印を替える。判断の席の打牌は牌だけを替え、その牌が手に入った所も同じ物理牌に
   * する（打牌は判断より前に手を出たので、判断の手は変わらない）。
   */
  private void river(RiverEdit edit, String name) {
    if (edit.seat() < 0 || edit.seat() >= GameState.NUM_PLAYERS)
      throw invalidEdit(name + ": seat must be between 0 and 3.");
    PlayerState player = players.get(edit.seat());
    if (edit.index() < 0 || edit.index() >= player.river().size())
      throw invalidEdit(name + ": the river has no tile at this index at the decision.");
    RiverTile old = player.river().get(edit.index());
    if (old.called()) throw invalidEdit(name + ": a called tile cannot be edited.");
    requireTile(edit.tile(), name);
    boolean own = edit.seat() == seat;
    if (own && edit.tsumogiri() != old.tsumogiri())
      throw invalidEdit(name + ": a discard of the deciding seat keeps its recorded tsumogiri.");
    int position = discards.get(edit.seat()).get(edit.index());
    var discard = (Dahai) events.get(position);
    int id = discard.physicalTileId();
    if (edit.tsumogiri() != old.tsumogiri())
      events.set(position, new Dahai(edit.seat(), id, edit.tsumogiri(), discard.inferred()));
    if (!edit.tile().equals(old.tile())) {
      Slot discarded = new Slot.Discarded(position);
      replace(name, id, edit.tile(), own ? List.of(discarded, entry(id)) : List.of(discarded));
    }
    players.set(
        edit.seat(),
        player.withRiverTile(edit.index(), old.withTile(edit.tile(), edit.tsumogiri())));
  }

  /** 開いているドラ表示牌の 1 枚。局の始めの表示は局の始めの出来事に、槓ドラはそれを開いた出来事に写す。 */
  private void dora(DoraEdit edit, String name) {
    if (edit.index() < 0 || edit.index() >= dora.size())
      throw invalidEdit(name + ": no dora indicator is revealed at this index at the decision.");
    requireTile(edit.tile(), name);
    if (edit.tile().equals(dora.get(edit.index()))) return;
    placements.add(new Placement(name, edit.tile()));
    dora.set(edit.index(), edit.tile());
    if (edit.index() > 0)
      events.set(
          reveals.get(edit.index() - 1),
          new Dora(TileCode.type(edit.tile()), TileCode.red(edit.tile())));
  }

  /**
   * 判断の席の手牌かツモ牌の 1 枚。同じ牌が手に複数あれば、どの位置を指しても最も後に手に入った 1 枚（同じ組の次の変更は次に新しい 1 枚）を
   * 替える。
   */
  private void hand(HandEdit edit, String name) {
    if (decision.riichi())
      throw invalidEdit(name + ": the hand cannot be edited after a riichi declaration.");
    PlayerState shown = decision.table().players().get(seat);
    int size = shown.hand().size();
    boolean drawn = edit.index() == size;
    if (edit.index() < 0 || edit.index() > size || drawn && shown.draw() == null)
      throw invalidEdit(name + ": the hand has no tile at this index at the decision.");
    requireTile(edit.tile(), name);
    String old = drawn ? shown.draw() : shown.hand().get(edit.index());
    if (edit.tile().equals(old)) return;
    int id = drawn ? decision.drawn() : takeNewest(old);
    replace(name, id, edit.tile(), List.of(entry(id)));
    if (drawn) draw = edit.tile();
    else hand.set(edit.index(), edit.tile());
  }

  /** 河か手牌の牌を替えた変更。物理牌 removed を書いた所 slots へ、新しい牌の物理牌を後で書く。 */
  private void replace(String name, int removed, String tile, List<Slot> slots) {
    placements.add(new Placement(name, tile));
    replacements.add(new Replacement(removed, tile, slots));
  }

  /** 伏せた手の牌 code のうち、最も後に手に入った物理牌を取り出す。 */
  private int takeNewest(String code) {
    for (int i = concealed.size() - 1; i >= 0; i--)
      if (TileCopies.isCopy(concealed.get(i), code)) return concealed.remove(i);
    throw new IllegalStateException("The hand of seat " + seat + " does not hold " + code);
  }

  /** 判断の席の手に物理牌が入った所（配牌の位置か、判断より前のその席のツモ）。 */
  private Slot entry(int physicalTileId) {
    Slot slot = entries.get(physicalTileId);
    if (slot == null)
      throw new IllegalStateException(
          "Tile " + physicalTileId + " never entered the hand of seat " + seat);
    return slot;
  }

  /** 判断の席の手を並べ、点・本場・供託を写し、見える牌の数を確かめてから物理牌を割り当て、局の始めを作り直す。 */
  private RewrittenRound finish(ScenarioEdits edits) {
    hand.sort(Comparator.comparingInt(TileCode::order));
    players.set(seat, players.get(seat).withHand(List.copyOf(hand), draw));
    TableState shown = decision.table();
    int[] scores = scores(edits.scores());
    int honba = edits.honba() == null ? start.honba() : requireHonba(edits.honba());
    int deposited = shown.kyotaku() - start.kyotaku();
    int kyotaku =
        edits.kyotaku() == null ? shown.kyotaku() : requireKyotaku(edits.kyotaku(), deposited);
    var table =
        new TableState(players, dora, honba, kyotaku, shown.remaining(), shown.activeSeat());
    VisibleTileCount.require(table, seat, placements);
    assign();
    String indicator = dora.getFirst();
    events.set(
        0,
        new StartKyoku(
            start.wind(),
            start.kyoku(),
            honba,
            kyotaku - deposited,
            start.oya(),
            TileCode.type(indicator),
            deal,
            scores,
            TileCode.red(indicator)));
    return new RewrittenRound(events, table);
  }

  /**
   * 替えた牌の物理牌を全て空けてから、新しい牌それぞれに空いている写しを割り当てて書く。結果が変更の順に依らないよう、空ける物理牌の順に
   * 割り当てる。
   */
  private void assign() {
    replacements.sort(Comparator.comparingInt(Replacement::removed));
    for (Replacement replacement : replacements) copies.free(replacement.removed());
    for (Replacement replacement : replacements) {
      int id = copies.take(replacement.tile());
      for (Slot slot : replacement.slots()) write(slot, id);
    }
  }

  /** 物理牌を書く。出来事はほかの値を保つ。 */
  private void write(Slot slot, int physicalTileId) {
    switch (slot) {
      case Slot.Dealt(int index) -> deal[seat][index] = physicalTileId;
      case Slot.Drawn(int position) -> {
        var drawn = (Tsumo) events.get(position);
        events.set(position, new Tsumo(drawn.actor(), physicalTileId, drawn.inferred()));
      }
      case Slot.Discarded(int position) -> {
        var discard = (Dahai) events.get(position);
        events.set(
            position,
            new Dahai(discard.actor(), physicalTileId, discard.tsumogiri(), discard.inferred()));
      }
    }
  }

  /** 判断の時点の点を受け取り、局の始めの点を同じ差だけ動かす。 */
  private int[] scores(List<Integer> edits) {
    int[] scores = start.scores().clone();
    for (int player = 0; player < edits.size(); player++) {
      int score = edits.get(player);
      String name = "edits.scores[" + player + "]";
      if (score % ScenarioLimits.POINT_UNIT != 0)
        throw invalidEdit(name + ": scores must be multiples of 100.");
      if (score < ScenarioLimits.MIN_SCORE || score > ScenarioLimits.MAX_SCORE)
        throw invalidEdit(name + ": the score is out of range.");
      scores[player] += score - players.get(player).score();
      players.set(player, players.get(player).withScore(score));
    }
    return scores;
  }

  private static int requireHonba(int honba) {
    if (honba < 0 || honba > ScenarioLimits.MAX_HONBA)
      throw invalidEdit(
          "edits.honba: honba must be between 0 and " + ScenarioLimits.MAX_HONBA + ".");
    return honba;
  }

  /** 判断の時点の供託は、判断までにこの局で出たリーチの棒より少なくできない。 */
  private static int requireKyotaku(int kyotaku, int deposited) {
    if (kyotaku < deposited || kyotaku > ScenarioLimits.MAX_KYOTAKU)
      throw invalidEdit(
          "edits.kyotaku: kyotaku must be between "
              + deposited
              + " (riichi sticks deposited in this round) and "
              + ScenarioLimits.MAX_KYOTAKU
              + ".");
    return kyotaku;
  }

  private static void requireTile(String tile, String name) {
    if (!TileCode.isValid(tile)) throw invalidEdit(name + ": invalid tile code " + tile + ".");
  }

  private static ReviewerException invalidEdit(String message) {
    return new ReviewerException(ErrorCode.INVALID_EDIT, message);
  }

  /** 物理牌を書く所。 */
  private sealed interface Slot {
    /** 判断の席の配牌の位置。 */
    record Dealt(int index) implements Slot {}

    /** 判断の席のツモの出来事の位置。 */
    record Drawn(int position) implements Slot {}

    /** 打牌の出来事の位置。 */
    record Discarded(int position) implements Slot {}
  }

  /**
   * 物理牌を替える牌。
   *
   * @param removed 局から外す物理牌
   * @param tile 新しい牌
   * @param slots removed を書いた所。新しい牌の物理牌に書き換える
   */
  private record Replacement(int removed, String tile, List<Slot> slots) {}
}
