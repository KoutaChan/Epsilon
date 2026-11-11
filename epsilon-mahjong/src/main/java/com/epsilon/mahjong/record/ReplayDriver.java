package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.Wall;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜の局を、自己対局と同じ {@code Engine} で、記録の手のとおりに進める。始めに、記録の配牌・表示牌・ツモ・裏ドラ表示牌をエンジンの
 * 牌山の位置（{@code Wall} の位置の式）に置き、残りの位置を記録に無い牌で埋める。判断の時点ごとに、記録の手が合法手にあるかを確かめる。
 * 打牌と北抜きでは記録の物理牌をエンジンの手から出すので、同じ種類の牌を何枚も持つ手でも、エンジンの手の物理牌は記録と同じになる
 * （チー・ポンで手から出す牌は記録に無いので、エンジンが選ぶ）。局末ごとに、局の結果・和了の点・点の移動が記録と合うかを確かめる。
 * 半荘を再生するときは、局末の後の卓が次局の始まり（最後の局
 * なら終局の点）と合うかも確かめる。記録が内訳を持たない結果（点を持たない和了、終わり方を持たない流局）は、和了者・放銃者と点の
 * 移動だけを照合し、内訳はエンジンの局末から読む。
 *
 * <p>他家の応答では、記録にある鳴きとロンをそのまま選び、記録に無い応答は見送りにする。ただし、通った手より優先の低い手（ロンに対する
 * 鳴き、ポン・大明槓に対するチー）を持つ席は、その手を選んだのに消されたのかどうかが記録から分からない。その席は
 * {@link #UNOBSERVED} とし、エンジンには見送りを渡す。加槓と抜いた北への応答はロンと見送りだけなので、分からない席は出ない。
 *
 * <p>{@code replay} と、判断の時点ごとに進めるメソッド（{@code start}・{@code step}）は、記録と食い違うと IllegalStateException
 * を投げる。判断の時点ごとに進めるときは、局をまたいだ卓のつながりは確かめない。
 *
 * <p>牌山と選んだ手の配列は使い回し、イベントごとにメモリを確保しない。同時に使えるのは 1 つのスレッドだけ。
 */
public final class ReplayDriver {

  /** 記録から分からない応答（優先の高い手に消されたかもしれない）。 */
  public static final int UNOBSERVED = -1;

  private static final int PASS_CODE = Action.code(Type.PASS);
  private static final int RON_CODE = Action.code(Type.RON_AGARI);
  private static final int TSUMO_CODE = Action.code(Type.TSUMO_AGARI);
  private static final int KYUSHU_CODE = Action.code(Type.KYUSHU_KYUHAI);

  /** 記録のイベントが手番の手でないことを表す。 */
  private static final int NO_ACTION = -1;

  /** 手が合法手に無いことを表す。 */
  private static final int ABSENT = -1;

  /** 応答の手の優先順位。後の定数ほど優先が高い。通った手より低い手は、選ばれていても消える。 */
  private enum Precedence {
    NONE,
    CHI,
    /** ポン・大明槓。 */
    MELD,
    RON;

    static Precedence of(Type type) {
      return switch (type) {
        case RON_AGARI -> RON;
        case PON, DAIMINKAN -> MELD;
        case CHI -> CHI;
        default -> NONE;
      };
    }
  }

  private final Variant variant;
  private final TileSet set;
  private final int seats;
  private final Engine engine;

  /** 組んだ牌山（位置 → 物理 id）。 */
  private final int[] wall;

  /** 物理 id ごとに、記録の牌として牌山に置いたか。 */
  private final boolean[] placed = new boolean[Tiles.PHYSICAL_IDS];

  /** 判断する席ごとに、{@code commit} に渡す手（合法手の何番目か）。 */
  private final int[] choices;

  /** 判断する席ごとに、記録が選んだ手（合法手の何番目か。記録から分からなければ {@link #UNOBSERVED}）。 */
  private final int[] recorded;

  /**
   * 手番の判断で、記録の打牌か北抜きが手から出す物理牌。{@code commit} に渡し、エンジンに同じ物理牌を出させる。ほかの判断では
   * {@link TurnEvent#NONE}。
   */
  private int played = TurnEvent.NONE;

  private RecordedRound round;

  /** 次に照合する記録のイベントの位置。判断の時点では、その判断で選ぶ記録の手はもう読み終えている。 */
  private int read;

  /** エンジンが止まっている所（判断の時点・局末・終局）。 */
  private Boundary boundary;

  /** 再生している牌譜の id。判断の時点ごとに進めるときは null。 */
  private String id;

  public ReplayDriver(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    set = variant.tiles();
    seats = variant.seats();
    engine = new Engine(variant);
    wall = new int[set.size()];
    choices = new int[seats - 1];
    recorded = new int[seats - 1];
  }

  /** 記録の手で進めるエンジン。判断の時点ごとに、判断・ビュー・局末をここから読む。 */
  public Engine engine() {
    return engine;
  }

  /** 局を始め、最初の判断の時点まで進める。記録と食い違えば IllegalStateException。 */
  public Boundary start(RecordedRound round, RuleWorkspace ws) {
    id = null;
    begin(round, ws);
    return boundary;
  }

  /**
   * 判断の時点で、記録が選んだ手。
   *
   * @param slot 何番目の判断する席か
   * @return 合法手の何番目か。記録から分からなければ {@link #UNOBSERVED}
   */
  public int choice(int slot) {
    return recorded[Objects.checkIndex(slot, engine.decisions().size())];
  }

  /** 記録が選んだ手をエンジンに渡し（記録から分からない席は見送る）、次の判断の時点まで進める。記録と食い違えば IllegalStateException。 */
  public Boundary step(RuleWorkspace ws) {
    arrive(engine.commit(choices, played, ws));
    return boundary;
  }

  /**
   * 局の記録のイベントのうち、エンジンがまだ行っていない最初のイベントの位置。それより前のイベントは、エンジンが全て行った。判断の
   * 時点では、この判断で選ぶ手が起こすイベントがここから始まる。値がイベントの数と同じなら、この判断の後に記録のイベントは無い
   * （和了か九種九牌か、局を終える見送り）。局末と終局ではイベントの数を返す。局を始める前は IllegalStateException。
   *
   * <p>記録のイベントは {@code RecordedRound} の決まりどおり、エンジンが起こす順に並ぶ。このため位置は、エンジンが行った公開のイベント、
   * ツモ、最初の表示牌の後にめくった表示牌、成立した立直の数を足したものに等しい。
   *
   * @return {@code RecordedRound} のイベントの列の何番目か
   */
  public int cursor() {
    if (round == null) {
      throw new IllegalStateException("No round has started");
    }
    if (boundary != Boundary.DECISIONS) {
      return round.events().size();
    }
    FullView full = engine.full();
    SeatView view = full.seat(0);
    int accepted = 0;
    for (int seat = 0; seat < seats; seat++) {
      accepted += view.riichi(seat) != RiichiState.NONE ? 1 : 0;
    }
    return view.events().size() + full.chance().size() + accepted;
  }

  /** 半荘の全ての局を再生する。記録と食い違えば IllegalStateException。 */
  public void replay(RecordedGame game, RuleWorkspace ws) {
    if (game.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + game.variant());
    }
    id = game.meta().id();
    List<RecordedRound> rounds = game.rounds();
    for (int index = 0; index < rounds.size(); index++) {
      begin(rounds.get(index), ws);
      while (boundary == Boundary.DECISIONS) {
        arrive(engine.commit(choices, played, ws));
      }
      checkNext(game, index);
    }
  }

  private void begin(RecordedRound round, RuleWorkspace ws) {
    if (round.start().variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + round.start().variant());
    }
    this.round = round;
    read = 0;
    buildWall();
    engine.startRecordedRound(round.start(), wall);
    arrive(engine.advance(ws));
  }

  /** エンジンが止まった所で呼ぶ。判断の時点なら記録が選んだ手を用意し、局末なら記録と照合する。 */
  private void arrive(Boundary reached) {
    boundary = reached;
    if (reached == Boundary.DECISIONS) {
      match();
    } else {
      checkEnd();
    }
  }

  private void match() {
    Decisions decisions = engine.decisions();
    switch (decisions.kind(0)) {
      case TURN -> {
        matchDraw(decisions.seat(0));
        matchTurn(decisions);
      }
      case POST_CALL_DAHAI -> matchTurn(decisions);
      case RESPONSE, CHANKAN, NUKI_RESPONSE -> matchResponses(decisions);
    }
  }

  /**
   * 手番を始めたツモが、記録の次のイベント（同じ席の、同じ種類の牌のツモ）かを確かめる。記録がそのツモより前に終わっていれば、
   * エンジンが他家の応答を求めなかったイベントへのロンで終わる記録なら合法でない手、ほかは局末の不一致とする。
   */
  private void matchDraw(int seat) {
    RecordEvent event = next();
    if (event == null) {
      rejectUnofferedRon();
      throw fault("The record ends before seat " + seat + " draws, with " + round.result());
    }
    SeatView view = engine.view(seat);
    if (!(event instanceof RecordEvent.Draw draw)
        || draw.seat() != seat
        || Tiles.kind(draw.tile()) != view.last().tile()) {
      throw fault("Seat " + seat + " draws a tile of kind " + view.last().tile() + ", but the record has " + event);
    }
    read++;
  }

  /**
   * 記録の局の結果が、エンジンの最後の公開のイベント（打牌・加槓・北抜き）へのロン（三家和を含む）なら、合法でない手として
   * IllegalStateException を投げる。エンジンがロンで局を終えていない所で呼ぶので、そうであれば、エンジンは誰もロンできないと判断して
   * 応答を求めなかったことになる。
   */
  private void rejectUnofferedRon() {
    RoundResult result = round.result();
    // 公開のイベントは、どの席から見ても同じ。
    EventLog events = engine.view(variant.dealer(round.start().round())).events();
    int last = events.size() - 1;
    if (last < 0 || events.kind(last) == EventKind.MELD) {
      return;
    }
    if (result.kind() == EndKind.RON && result.loser() == events.seat(last)) {
      throw fault("The record rons the " + events.kind(last) + " of seat " + result.loser() + ", which no seat can claim");
    }
    if (result.abortive() == AbortiveDraw.TRIPLE_RON) {
      throw fault("The record ends with three rons, which the engine does not offer");
    }
  }

  /**
   * 手番か鳴いた後の打牌の判断で、記録の次の手番の手を選ぶ。記録がここで終わるなら、局の結果のツモ和了か九種九牌を選ぶ。記録の打牌と
   * 北抜きは、記録の物理牌をエンジンにも出させる（同じ種類の牌のどれを出したかは、手のコードからは決まらない）。
   */
  private void matchTurn(Decisions decisions) {
    int seat = decisions.seat(0);
    RecordEvent event = next();
    int code;
    if (event == null) {
      code = endingCode(seat);
      if (code == NO_ACTION) {
        throw fault("The record ends at the turn of seat " + seat + " with " + round.result());
      }
    } else {
      code = turnCode(event, seat);
      if (code == NO_ACTION) {
        throw fault("Seat " + seat + " is on turn, but the record has " + event);
      }
      read++;
    }
    int index = indexOf(decisions, 0, code);
    if (index == ABSENT) {
      throw fault(Action.of(code) + " is not legal for seat " + seat);
    }
    played = playedTile(event);
    if (played != TurnEvent.NONE && !engine.view(seat).holds(played)) {
      throw fault("Seat " + seat + " does not hold tile " + played + " of " + event);
    }
    choices[0] = index;
    recorded[0] = index;
  }

  /**
   * 記録が手番で終わるときの手（その席のツモ和了か九種九牌）。手番で終わる流局は九種九牌だけなので、終わり方を持たない流局も
   * 九種九牌とする。
   *
   * @param seat 手番の席
   * @return 手のコード。どちらでもなければ {@link #NO_ACTION}
   */
  private int endingCode(int seat) {
    RoundResult result = round.result();
    if (result.kind() == EndKind.TSUMO && result.wins().get(0).seat() == seat) {
      return TSUMO_CODE;
    }
    boolean kyushu = result.abortive() == AbortiveDraw.NINE_TERMINALS || result.kind() == null;
    return kyushu ? KYUSHU_CODE : NO_ACTION;
  }

  /** 記録の手番のイベントが手から出す物理牌。打牌と北抜きのほかは {@link TurnEvent#NONE}。 */
  private static int playedTile(RecordEvent event) {
    return switch (event) {
      case RecordEvent.Discard discard -> discard.tile();
      case RecordEvent.Nuki nuki -> nuki.tile();
      case null, default -> TurnEvent.NONE;
    };
  }

  /** 手番のイベントを、手のコードにする。指定した席のイベントでないか、手番の手でなければ {@link #NO_ACTION}。 */
  private int turnCode(RecordEvent event, int seat) {
    return switch (event) {
      case RecordEvent.Discard discard when discard.seat() == seat ->
          Action.discard(Tiles.kind(discard.tile()), set.red(discard.tile()), discard.tsumogiri(), discard.riichi());
      case RecordEvent.ClosedKan kan when kan.seat() == seat ->
          Action.code(Type.ANKAN, kan.kind(), TileSelection.NOT_APPLICABLE);
      case RecordEvent.AddedKan kan when kan.seat() == seat ->
          Action.code(Type.KAKAN, kan.kind(), TileSelection.NOT_APPLICABLE);
      case RecordEvent.Nuki nuki when nuki.seat() == seat -> Action.code(Type.NUKI);
      default -> NO_ACTION;
    };
  }

  /**
   * 打牌・加槓・抜いた北への応答を選ぶ。記録の次のイベントがこの打牌への鳴きなら、鳴いた席はその鳴きを選ぶ。記録がここで終わり、
   * 局がこのイベントへのロンで終わるなら、和了者はロンを選ぶ。ほかの席は見送るが、通った手より優先の低い手を持つ席は
   * {@link #UNOBSERVED} にする。
   */
  private void matchResponses(Decisions decisions) {
    played = TurnEvent.NONE;
    int actor = engine.view(decisions.seat(0)).last().seat();
    for (int slot = 0; slot < decisions.size(); slot++) {
      recorded[slot] = indexOf(decisions, slot, PASS_CODE);
    }
    RecordEvent event = next();
    Precedence executed = Precedence.NONE;
    if (event instanceof RecordEvent.Call call) {
      matchCall(decisions, call, actor);
      executed = Precedence.of(Action.of(Action.call(Meld.of(call.meld()))).type());
    } else if (event == null && robs(actor, decisions)) {
      matchRons(decisions);
      executed = Precedence.RON;
    }
    for (int slot = 0; slot < decisions.size(); slot++) {
      int pass = indexOf(decisions, slot, PASS_CODE);
      if (recorded[slot] == pass && cutOff(decisions, slot, executed)) {
        recorded[slot] = UNOBSERVED;
      }
      choices[slot] = recorded[slot] == UNOBSERVED ? pass : recorded[slot];
    }
  }

  /**
   * 記録の鳴きを、鳴いた席が選ぶ手にする。鳴きが合うのは、応答を求めた打牌を鳴いていて、鳴いた牌の赤かどうかが打牌と同じときだけ。
   */
  private void matchCall(Decisions decisions, RecordEvent.Call call, int actor) {
    Meld meld = Meld.of(call.meld());
    int slot = decisions.slotOf(call.seat());
    boolean fits =
        slot >= 0
            && meld.source().seat(call.seat(), seats) == actor
            && (meld.aka() == Meld.AkaSource.CALLED_TILE) == engine.view(call.seat()).last().red();
    int index = fits ? indexOf(decisions, slot, Action.call(meld)) : ABSENT;
    if (index == ABSENT) {
      throw fault("Seat " + call.seat() + " cannot make " + meld);
    }
    recorded[slot] = index;
    read++;
  }

  /**
   * 記録の局の結果が、応答を求めたイベント（打牌・加槓・抜いた北）へのロン（三家和を含む）か。記録がここで終わり、結果が終わり方を
   * 持たない流局のときは、三家和のあるルールで、応答する全ての席がロンできるときだけ三家和とみなす。全員が見送って終わる流局とは
   * 記録から区別できないので、ロンできる席は全てロンしたとみる。
   *
   * @param actor 応答を求めたイベントの席
   * @param decisions 応答の判断
   */
  private boolean robs(int actor, Decisions decisions) {
    RoundResult result = round.result();
    if (result.kind() == EndKind.RON && result.loser() == actor || result.abortive() == AbortiveDraw.TRIPLE_RON) {
      return true;
    }
    if (result.kind() != null || !variant.aborts(AbortiveDraw.TRIPLE_RON) || decisions.size() != seats - 1) {
      return false;
    }
    for (int slot = 0; slot < decisions.size(); slot++) {
      if (indexOf(decisions, slot, RON_CODE) == ABSENT) {
        return false;
      }
    }
    return true;
  }

  /** 記録のロン（三家和なら応答する全ての席のロン）を、和了者が選ぶ手にする。 */
  private void matchRons(Decisions decisions) {
    RoundResult result = round.result();
    if (result.kind() == EndKind.RON) {
      for (int i = 0; i < result.wins().size(); i++) {
        matchRon(decisions, result.wins().get(i).seat());
      }
      return;
    }
    if (decisions.size() != seats - 1) {
      throw fault("Three rons need every other seat to claim the tile");
    }
    for (int slot = 0; slot < decisions.size(); slot++) {
      matchRon(decisions, decisions.seat(slot));
    }
  }

  private void matchRon(Decisions decisions, int winner) {
    int slot = decisions.slotOf(winner);
    int index = slot < 0 ? ABSENT : indexOf(decisions, slot, RON_CODE);
    if (index == ABSENT) {
      throw fault("Seat " + winner + " cannot ron");
    }
    recorded[slot] = index;
  }

  /**
   * エンジンが局を終えたときに、記録と照合する。記録にまだ手が残っているか、局の結果・和了の点・点の移動のどれかが記録と違えば
   * 食い違いとする。エンジンがロン以外で局を終えたのに、記録の結果が最後の打牌へのロンなら、合法でない手とする。エンジンは、最後の牌か
   * 途中流局になるその打牌に応答を求めなかったことになる。
   */
  private void checkEnd() {
    RecordEvent rest = next();
    if (rest != null) {
      throw fault("The engine ended the round before " + rest);
    }
    RoundEnd end = engine.roundEnd();
    RoundOutcome outcome = end.outcome();
    boolean ron = outcome.kind() == EndKind.RON || outcome.abortive() == AbortiveDraw.TRIPLE_RON;
    if (!ron) {
      rejectUnofferedRon();
    }
    if (!round.result().matches(end, round.riichiMask())) {
      throw fault("The engine ended with " + describe(end) + ", but the record has " + round.result());
    }
  }

  /** 局末の後の卓が、次局の始まり（最後の局なら終局の点）と同じかを確かめる。 */
  private void checkNext(RecordedGame game, int index) {
    TableState next = engine.roundEnd().next();
    List<RecordedRound> rounds = game.rounds();
    boolean last = index == rounds.size() - 1;
    boolean same = last ? next.endsWith(game.finalScores()) : next.startsAt(rounds.get(index + 1).start());
    if (!same) {
      String recorded =
          last ? "ends with " + Arrays.toString(game.finalScores()) : "goes on with " + rounds.get(index + 1).start();
      throw fault("The engine moves on to " + next + ", but the record " + recorded);
    }
  }

  /**
   * 記録の配牌・表示牌・ツモ・裏ドラ表示牌をエンジンの牌山の位置に置き、残りの位置を記録に無い牌（物理 id の昇順）で埋める。ツモは、
   * 暗槓・加槓・大明槓・北抜きの直後なら嶺上（補充）の位置に、ほかは山の位置に順に置く。置けない記録なら IllegalStateException を投げる。
   */
  private void buildWall() {
    Arrays.fill(wall, Wall.UNKNOWN);
    Arrays.fill(placed, false);
    int dealer = variant.dealer(round.start().round());
    for (int seat = 0; seat < seats; seat++) {
      int offset = Math.floorMod(seat - dealer, seats);
      for (int index = 0; index < Variant.HAND; index++) {
        place(Wall.dealtPosition(variant, offset, index), round.deal(seat, index));
      }
    }
    int indicators = 0;
    place(Wall.indicatorPosition(variant, indicators++), round.indicator());
    int live = 0;
    int replacements = 0;
    boolean replacement = false;
    List<RecordEvent> events = round.events();
    for (int index = 0; index < events.size(); index++) {
      switch (events.get(index)) {
        case RecordEvent.Draw draw -> {
          if (replacement ? replacements == variant.rinshanTiles() : live == variant.liveWall()) {
            throw fault("The record draws more tiles than the wall holds");
          }
          int position =
              replacement ? Wall.rinshanPosition(variant, ++replacements) : Wall.livePosition(variant, live++);
          place(position, draw.tile());
          replacement = false;
        }
        case RecordEvent.DoraReveal reveal -> {
          if (indicators == WinContext.MAX_INDICATORS) {
            throw fault("The record reveals more than " + WinContext.MAX_INDICATORS + " indicators");
          }
          place(Wall.indicatorPosition(variant, indicators++), reveal.tile());
        }
        case RecordEvent.Call call -> replacement = Meld.of(call.meld()).type() == Meld.Type.DAIMINKAN;
        case RecordEvent.ClosedKan ignored -> replacement = true;
        case RecordEvent.AddedKan ignored -> replacement = true;
        case RecordEvent.Nuki ignored -> replacement = true;
        case RecordEvent.Discard ignored -> replacement = false;
        case RecordEvent.RiichiAccepted ignored -> {}
      }
    }
    RoundResult result = round.result();
    if (result.uraCount() > indicators) {
      throw fault("The record has " + result.uraCount() + " ura indicators for " + indicators + " indicators");
    }
    for (int index = 0; index < result.uraCount(); index++) {
      place(Wall.uraPosition(variant, index), result.ura(index));
    }
    int next = 0;
    for (int position = 0; position < wall.length; position++) {
      if (wall[position] == Wall.UNKNOWN) {
        while (placed[set.physicalId(next)]) {
          next++;
        }
        wall[position] = set.physicalId(next++);
      }
    }
  }

  private void place(int position, int tile) {
    if (!set.contains(tile) || placed[tile] || wall[position] != Wall.UNKNOWN) {
      throw fault("Tile " + tile + " cannot sit at wall position " + position);
    }
    wall[position] = tile;
    placed[tile] = true;
  }

  /** 記録の次の手のイベント。表示牌と立直の成立は牌山と局の結果で照合するので、読み飛ばす。記録が尽きていれば null。 */
  private RecordEvent next() {
    List<RecordEvent> events = round.events();
    while (read < events.size()) {
      RecordEvent event = events.get(read);
      if (!(event instanceof RecordEvent.DoraReveal) && !(event instanceof RecordEvent.RiichiAccepted)) {
        return event;
      }
      read++;
    }
    return null;
  }

  /** 記録との食い違いを表す例外。メッセージの前に、牌譜 id（あれば）と局の始まりを付ける。 */
  private IllegalStateException fault(String message) {
    return new IllegalStateException((id == null ? "" : id + " ") + round.start() + ": " + message);
  }

  private String describe(RoundEnd end) {
    StringBuilder text = new StringBuilder().append(end.outcome());
    for (int i = 0; i < end.outcome().winnerCount(); i++) {
      text.append(' ').append(end.score(i));
    }
    text.append(" deltas=[");
    for (int seat = 0; seat < seats; seat++) {
      text.append(seat == 0 ? "" : ", ").append(end.delta(seat));
    }
    return text.append(']').toString();
  }

  private static int indexOf(Decisions decisions, int slot, int code) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      if (decisions.legal(slot, index) == code) {
        return index;
      }
    }
    return ABSENT;
  }

  /**
   * 判断する席に、通った手より優先の低い（消された）応答の手があるか。
   *
   * @param slot 何番目の判断する席か
   * @param executed 通った手の優先順位
   */
  private static boolean cutOff(Decisions decisions, int slot, Precedence executed) {
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Precedence precedence = Precedence.of(Action.of(decisions.legal(slot, index)).type());
      if (precedence != Precedence.NONE && precedence.compareTo(executed) < 0) {
        return true;
      }
    }
    return false;
  }
}
