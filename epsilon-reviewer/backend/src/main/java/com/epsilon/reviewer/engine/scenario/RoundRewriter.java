package com.epsilon.reviewer.engine.scenario;

import com.epsilon.mahjong.record.RecordEditor;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.rules.Variant;
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
import java.util.TreeMap;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * もしもの変更を確かめて、判断のある局に反映する。反映する先は、判断する席から見た局の記録と、判断の時点の表示用の局面。確かめる順は
 * 河 → ドラ → 手牌 → 点 → 本場 → 供託 → 見える牌の数。
 *
 * <p>変更する牌は、判断の時点の表示での位置で指す。河はその席の打牌の順、ドラはめくった順（0 は局の始め、1 から先は槓ドラ）、手牌は
 * {@code hand[]} の位置（{@code hand.length} はツモ牌）で指す。点と供託は判断の時点の値で受け取る。判断までの立直の支払いが変わらない
 * よう、局の始めの値を同じ差だけ動かす。
 *
 * <p>河・ドラ・手牌で牌を替えたときの物理牌は、見える牌の数を確かめてから割り当てる。まず替えた牌の物理牌を全て空ける。次に新しい牌
 * ごとに、その牌の空いている写しのうち番号の最も小さいものを割り当て、元の物理牌があった所へ書く。書く所は、打牌、ドラ表示牌、または
 * 判断する席にその牌が入った所（配牌の位置かツモ）。判断する席に見える物理牌（その席の配牌とツモ、全ての打牌、ドラ表示牌）は、どれも
 * 判断の時点の表示に見える牌なので、見える牌の数の確かめを通れば写しは足りる。鳴きの記録は手から出した物理牌を持たないので、他家の
 * 面子の物理牌は数えない（席から見える卓も読まない）。
 */
public final class RoundRewriter {
  private final Variant variant;
  private final int seats;
  private final LocatedDecision decision;
  private final int seat;
  private final RecordedRound recorded;
  private final RecordEditor editor;
  private final List<PlayerState> players;
  private final List<String> dora;

  /** 判断する席の手牌。判断の局面の {@code hand[]} の位置のまま書き換え、最後に並べ直す。 */
  private final List<String> hand;

  private String draw;

  /** 判断する席の伏せた手牌の物理牌（手に入った順）。手牌の変更で替えた牌は外す。 */
  private final List<Integer> concealed;

  /** 判断する席に見える物理牌。 */
  private final TileCopies copies;

  /** 判断する席の手に入った物理牌ごとに、どこから入ったか。 */
  private final Map<Integer, Entry> entries = new HashMap<>();

  /** 席ごとに、判断より前の打牌のイベントの位置（河の順）。 */
  private final List<List<Integer>> discards = new ArrayList<>();

  /** めくったドラ表示牌の物理牌（めくった順。0 は局の始め）。 */
  private final List<Integer> indicators = new ArrayList<>();

  /** ツモ切りの印を替えた他家の打牌の位置と、新しい印。 */
  private final Map<Integer, Boolean> tsumogiri = new TreeMap<>();

  /** 牌を替えた変更（適用した順）。 */
  private final List<Placement> placements = new ArrayList<>();

  private final List<Replacement> replacements = new ArrayList<>();

  private RoundRewriter(RecordedRound recorded, LocatedDecision decision) {
    variant = recorded.start().variant();
    seats = variant.seats();
    this.decision = decision;
    seat = decision.seat();
    this.recorded = recorded;
    editor = new RecordEditor(recorded, decision.position(), seat);
    copies = new TileCopies(variant);
    players = new ArrayList<>(decision.table().players());
    dora = new ArrayList<>(decision.table().dora());
    hand = new ArrayList<>(players.get(seat).hand());
    draw = players.get(seat).draw();
    concealed = new ArrayList<>(decision.concealed());
    for (int player = 0; player < seats; player++) discards.add(new ArrayList<>());
    for (int index = 0; index < Variant.HAND; index++)
      enter(recorded.deal(seat, index), new Entry.Dealt(index));
    reveal(recorded.indicator());
    List<RecordEvent> events = recorded.events();
    for (int position = 0; position < decision.position(); position++) {
      switch (events.get(position)) {
        case RecordEvent.Draw drawn when drawn.seat() == seat ->
            enter(drawn.tile(), new Entry.Drawn(position));
        case RecordEvent.Discard discard -> {
          discards.get(discard.seat()).add(position);
          copies.use(discard.tile());
        }
        case RecordEvent.DoraReveal revealed -> reveal(revealed.tile());
        default -> {}
      }
    }
    for (int player = 0; player < seats; player++)
      if (discards.get(player).size() != players.get(player).river().size())
        throw new IllegalStateException("The river of seat " + player + " differs from the round");
  }

  /**
   * @param recorded 判断のある局
   * @throws ReviewerException invalid_request（点の数が席の数と違う）、または invalid_edit（変更が正しくない）
   */
  public static RewrittenRound apply(
      RecordedRound recorded, LocatedDecision decision, ScenarioEdits edits) {
    int seats = recorded.start().variant().seats();
    if (!edits.scores().isEmpty() && edits.scores().size() != seats)
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST,
          "edits.scores must contain " + (seats == 3 ? "three" : "four") + " scores.");
    var round = new RoundRewriter(recorded, decision);
    each(edits.rivers(), "edits.rivers", edit -> List.of(edit.seat(), edit.index()), round::river);
    each(edits.dora(), "edits.dora", DoraEdit::index, round::dora);
    each(edits.hand(), "edits.hand", HandEdit::index, round::hand);
    return round.finish(edits);
  }

  /** 変更を順に適用する。同じ所を 2 回変えようとしたら断る。 */
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

  private void enter(int physicalId, Entry entry) {
    copies.use(physicalId);
    entries.put(physicalId, entry);
  }

  private void reveal(int physicalId) {
    copies.use(physicalId);
    indicators.add(physicalId);
  }

  /**
   * 河の牌を 1 枚替える（鳴かれた牌は替えない）。他家の打牌なら、牌とツモ切りの印を替える。判断する席の打牌なら牌だけを替え、その牌が
   * 手に入った所も同じ物理牌にする。打牌は判断より前に手から出ているので、判断の時点の手牌は変わらない。
   */
  private void river(RiverEdit edit, String name) {
    if (edit.seat() < 0 || edit.seat() >= seats)
      throw invalidEdit(name + ": seat must be between 0 and " + (seats - 1) + ".");
    PlayerState player = players.get(edit.seat());
    if (edit.index() < 0 || edit.index() >= player.river().size())
      throw invalidEdit(name + ": the river has no tile at this index at the decision.");
    RiverTile old = player.river().get(edit.index());
    if (old.called()) throw invalidEdit(name + ": a called tile cannot be edited.");
    requireTile(edit.tile(), name);
    if (edit.seat() == seat && edit.tsumogiri() != old.tsumogiri())
      throw invalidEdit(name + ": a discard of the deciding seat keeps its recorded tsumogiri.");
    int position = discards.get(edit.seat()).get(edit.index());
    var discard = (RecordEvent.Discard) recorded.events().get(position);
    if (edit.tsumogiri() != old.tsumogiri()) tsumogiri.put(position, edit.tsumogiri());
    if (!edit.tile().equals(old.tile()))
      replace(name, discard.tile(), edit.tile(), new Slot.Discarded(position));
    players.set(
        edit.seat(),
        player.withRiverTile(edit.index(), old.withTile(edit.tile(), edit.tsumogiri())));
  }

  /** めくられているドラ表示牌を 1 枚替える。 */
  private void dora(DoraEdit edit, String name) {
    if (edit.index() < 0 || edit.index() >= dora.size())
      throw invalidEdit(name + ": no dora indicator is revealed at this index at the decision.");
    requireTile(edit.tile(), name);
    if (edit.tile().equals(dora.get(edit.index()))) return;
    replace(name, indicators.get(edit.index()), edit.tile(), new Slot.Indicator(edit.index()));
    dora.set(edit.index(), edit.tile());
  }

  /**
   * 判断する席の手牌かツモ牌を 1 枚替える。同じ牌が手に何枚かあれば、どの位置を指しても、最も後に手に入った 1 枚を替える。同じ牌を
   * もう 1 枚替えると、その次に新しい 1 枚を替える。
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
    replace(name, id, edit.tile(), new Slot.Held(id));
    if (drawn) draw = edit.tile();
    else hand.set(edit.index(), edit.tile());
  }

  /** 河・ドラ・手牌で牌を替える変更を控えておく。新しい牌の物理牌は、外す物理牌があった所へ後で書く。 */
  private void replace(String name, int removed, String tile, Slot slot) {
    placements.add(new Placement(name, tile));
    replacements.add(new Replacement(removed, tile, slot));
  }

  /** 伏せた手牌にある指定の牌のうち、最も後に手に入った物理牌を取り出す。 */
  private int takeNewest(String code) {
    for (int i = concealed.size() - 1; i >= 0; i--)
      if (copies.isCopy(concealed.get(i), code)) return concealed.remove(i);
    throw new IllegalStateException("The hand of seat " + seat + " does not hold " + code);
  }

  /** 判断する席の手牌を並べ直し、点・本場・供託を反映する。見える牌の数を確かめてから物理牌を割り当て、局の始まりを作り直す。 */
  private RewrittenRound finish(ScenarioEdits edits) {
    hand.sort(Comparator.comparingInt(TileCode::order));
    players.set(seat, players.get(seat).withHand(List.copyOf(hand), draw));
    TableState shown = decision.table();
    int[] scores = scores(edits.scores());
    int honba = edits.honba() == null ? recorded.start().honba() : requireHonba(edits.honba());
    int deposited = shown.kyotaku() - recorded.start().kyotaku();
    int kyotaku =
        edits.kyotaku() == null ? shown.kyotaku() : requireKyotaku(edits.kyotaku(), deposited);
    var table =
        new TableState(
            List.copyOf(players),
            List.copyOf(dora),
            honba,
            kyotaku,
            shown.remaining(),
            shown.activeSeat());
    VisibleTileCount.require(variant, table, seat, placements);
    assign();
    editor.replaceStart(honba, kyotaku - deposited, scores);
    List<RecordEvent> events = new ArrayList<>(editor.events());
    tsumogiri.forEach(
        (position, value) -> {
          var discard = (RecordEvent.Discard) events.get(position);
          events.set(
              position,
              new RecordEvent.Discard(discard.seat(), discard.tile(), value, discard.riichi()));
        });
    return new RewrittenRound(
        editor.round(),
        editor.honba(),
        editor.kyotaku(),
        editor.scores(),
        editor.hand(),
        editor.indicator(),
        List.copyOf(events),
        table);
  }

  /**
   * 替えた牌の物理牌を全て空けてから、新しい牌ごとに空いている写しを割り当てて書く。結果が変更の順に左右されないよう、空ける物理牌の
   * 番号の順に割り当てる。
   */
  private void assign() {
    replacements.sort(Comparator.comparingInt(Replacement::removed));
    for (Replacement replacement : replacements) copies.free(replacement.removed());
    for (Replacement replacement : replacements) {
      int id = copies.take(replacement.tile());
      switch (replacement.slot()) {
        case Slot.Discarded(int position) -> editor.replaceDiscarded(position, id);
        case Slot.Indicator(int index) -> editor.replaceIndicator(index, id);
        case Slot.Held(int removed) -> {
          switch (entry(removed)) {
            case Entry.Dealt(int index) -> editor.replaceDealt(index, id);
            case Entry.Drawn(int position) -> editor.replaceDrawn(position, id);
          }
        }
      }
    }
  }

  /** 判断する席の手にその物理牌が入った所（配牌の位置か、判断より前のその席のツモ）を返す。 */
  private Entry entry(int physicalId) {
    Entry entry = entries.get(physicalId);
    if (entry == null)
      throw new IllegalStateException(
          "Tile " + physicalId + " never entered the hand of seat " + seat);
    return entry;
  }

  /** 判断の時点の持ち点を受け取り、局の始まりの持ち点を同じ差だけ動かす。 */
  private int[] scores(List<Integer> edits) {
    int[] scores = recorded.start().scores();
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

  /** 判断の時点の供託は、この局で判断までに出た立直棒の数より少なくできない。 */
  private int requireKyotaku(int kyotaku, int deposited) {
    int max = ScenarioLimits.maxKyotaku(variant);
    if (kyotaku < deposited || kyotaku > max)
      throw invalidEdit(
          "edits.kyotaku: kyotaku must be between "
              + deposited
              + " (riichi sticks deposited in this round) and "
              + max
              + ".");
    return kyotaku;
  }

  private static void requireTile(String tile, String name) {
    if (!TileCode.isValid(tile)) throw invalidEdit(name + ": invalid tile code " + tile + ".");
  }

  private static ReviewerException invalidEdit(String message) {
    return new ReviewerException(ErrorCode.INVALID_EDIT, message);
  }

  /** 新しい物理牌を書く所。 */
  private sealed interface Slot {
    /** 打牌のイベントの位置。 */
    record Discarded(int position) implements Slot {}

    /** ドラ表示牌の番号（0 は局の始め）。 */
    record Indicator(int index) implements Slot {}

    /** 判断する席の手にある物理牌（その牌が手に入った所へ書く）。 */
    record Held(int physicalId) implements Slot {}
  }

  /** 判断する席の手に物理牌が入った所。 */
  private sealed interface Entry {
    /** 配牌の位置。 */
    record Dealt(int index) implements Entry {}

    /** ツモのイベントの位置。 */
    record Drawn(int position) implements Entry {}
  }

  /**
   * 物理牌を替える 1 枚。
   *
   * @param removed 局から外す物理牌
   * @param tile 新しい牌
   * @param slot 新しい牌の物理牌を書く所
   */
  private record Replacement(int removed, String tile, Slot slot) {}
}
