package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
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
import com.epsilon.replay.TileNotation;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.Function;

/**
 * もしもの変更を確かめ、判断のある局の出来事と判断の表示の局面へ写す。河 → ドラ → 手牌 → 点 → 本場 → 供託 → 見える牌の数の順に確かめる。
 *
 * <p>変更は判断の時点の表示で指す。河はその席の打牌の順、ドラは開いた順（0 は局の始め、1 からは槓ドラ）、手牌は hand[] の位置（hand.length
 * はツモ牌）。点と供託は判断の時点の値で受け取り、判断までのリーチの支払いを保つよう局の始めの値を同じ差だけ動かす。
 *
 * <p>判断の席の牌（手牌と河）は、その牌が入った所（配牌かツモ）の物理牌を新しい牌の使っていない物理牌に替え、席の目の再生が判断の手を作り
 * 直す。他家の打牌は牌の種類だけが見えるので、代表の物理牌に替える。
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

  /** 使えない物理牌（判断より前に局に現れた牌と、変更が使った牌）。 */
  private final Set<Integer> used = new HashSet<>();

  /** 席ごとの、判断より前の打牌の出来事の位置（河の順）。 */
  private final List<List<Integer>> discards = new ArrayList<>();

  /** 判断より前に槓ドラを開いた出来事の位置（開いた順）。 */
  private final List<Integer> reveals = new ArrayList<>();

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

  /** 判断より前の出来事を読み、打牌と槓ドラの位置、局に現れた物理牌を覚える（他家のツモは判断の席に見えない）。 */
  private void scan(int position) {
    switch (events.get(position)) {
      case StartKyoku begin -> use(begin.initialHandPhysicalTileIds()[seat]);
      case Tsumo drawn when drawn.actor() == seat -> use(drawn.physicalTileId());
      case Dahai discard -> {
        discards.get(discard.actor()).add(position);
        use(discard.physicalTileId());
      }
      case Dora ignored -> reveals.add(position);
      case Chi call -> useCall(call.calledPhysicalTileId(), call.consumedPhysicalTileIds());
      case Pon call -> useCall(call.calledPhysicalTileId(), call.consumedPhysicalTileIds());
      case Daiminkan call -> useCall(call.calledPhysicalTileId(), call.consumedPhysicalTileIds());
      case Ankan kan -> use(kan.consumedPhysicalTileIds());
      case Kakan kan -> use(kan.addedPhysicalTileId());
      default -> {}
    }
  }

  private void use(int... physicalTileIds) {
    for (int id : physicalTileIds) used.add(id);
  }

  private void useCall(int called, int[] consumed) {
    use(called);
    use(consumed);
  }

  /**
   * 河の 1 枚（鳴かれた牌は変えない）。他家の打牌は牌とツモ切りの印を替える。判断の席の打牌は牌だけを替え、その牌が入った所と打牌を同じ
   * 新しい物理牌にする（打牌は判断より前に手を出たので、判断の手は変わらない）。
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
    if (edit.tile().equals(old.tile()) && edit.tsumogiri() == old.tsumogiri()) return;
    int position = discards.get(edit.seat()).get(edit.index());
    var discard = (Dahai) events.get(position);
    int id =
        own
            ? replaceInHand(discard.physicalTileId(), edit.tile(), name)
            : representative(edit.tile());
    events.set(position, new Dahai(edit.seat(), id, edit.tsumogiri(), discard.inferred()));
    players.set(
        edit.seat(),
        player.withRiverTile(edit.index(), old.withTile(edit.tile(), edit.tsumogiri())));
  }

  /** 開いているドラ表示牌の 1 枚。局の始めの表示は局の始めの出来事に、槓ドラはそれを開いた出来事に写す。 */
  private void dora(DoraEdit edit, String name) {
    if (edit.index() < 0 || edit.index() >= dora.size())
      throw invalidEdit(name + ": no dora indicator is revealed at this index at the decision.");
    requireTile(edit.tile(), name);
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
    replaceInHand(drawn ? decision.drawn() : takeNewest(old), edit.tile(), name);
    if (drawn) draw = edit.tile();
    else hand.set(edit.index(), edit.tile());
  }

  /** 伏せた手の牌 code のうち、最も後に手に入った物理牌を取り出す。 */
  private int takeNewest(String code) {
    for (int i = concealed.size() - 1; i >= 0; i--) {
      int id = concealed.get(i);
      if (Tile.typeOf(id) == TileCode.type(code) && Tile.isAka(id) == TileCode.red(code))
        return concealed.remove(i);
    }
    throw new IllegalStateException("The hand of seat " + seat + " does not hold " + code);
  }

  /**
   * 判断の席の手に物理牌 old が入った所（配牌の位置か、判断より前のその席のツモ）を、牌 code の使っていない物理牌に替える。
   *
   * @return 替えた物理牌
   */
  private int replaceInHand(int old, String code, String name) {
    int id = unusedCopy(code, name);
    int[] dealt = deal[seat];
    for (int slot = 0; slot < dealt.length; slot++)
      if (dealt[slot] == old) {
        dealt[slot] = id;
        return id;
      }
    for (int position = 1; position < decision.target(); position++)
      if (events.get(position) instanceof Tsumo drawn
          && drawn.actor() == seat
          && drawn.physicalTileId() == old) {
        events.set(position, new Tsumo(seat, id, drawn.inferred()));
        return id;
      }
    throw new IllegalStateException("Tile " + old + " never entered the hand of seat " + seat);
  }

  /** 牌 code の写しのうち、局でまだ使っていない一番小さい物理牌を使う。 */
  private int unusedCopy(String code, String name) {
    int type = TileCode.type(code);
    boolean red = TileCode.red(code);
    for (int id = type * Tile.TILES_PER_TYPE; id < (type + 1) * Tile.TILES_PER_TYPE; id++)
      if (Tile.isAka(id) == red && used.add(id)) return id;
    throw invalidEdit(name + ": no unused copy of " + code + " is left in this round.");
  }

  /** 他家の打牌の物理牌。牌の種類と赤だけが見えるので、代表の写しを使う。 */
  private int representative(String code) {
    int id = TileNotation.representativeTileId(code);
    used.add(id);
    return id;
  }

  /** 判断の席の手を並べ、点・本場・供託を写して局の始めを作り直し、見える牌の数を確かめる。 */
  private RewrittenRound finish(ScenarioEdits edits) {
    hand.sort(Comparator.comparingInt(TileCode::order));
    players.set(seat, players.get(seat).withHand(List.copyOf(hand), draw));
    TableState shown = decision.table();
    int[] scores = scores(edits.scores());
    int honba = edits.honba() == null ? start.honba() : requireHonba(edits.honba());
    int deposited = shown.kyotaku() - start.kyotaku();
    int kyotaku =
        edits.kyotaku() == null ? shown.kyotaku() : requireKyotaku(edits.kyotaku(), deposited);
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
    var table =
        new TableState(players, dora, honba, kyotaku, shown.remaining(), shown.activeSeat());
    VisibleTileCount.require(table, seat, edits);
    return new RewrittenRound(events, table);
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
}
