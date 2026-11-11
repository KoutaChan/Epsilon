package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Random;

/**
 * テスト用の牌譜を作るヘルパー。手をランダムに選ぶ自己対局で半荘を打ち、記録する。打ち手は和了・九種九牌・槓・北抜き・鳴き・立直を
 * 確率で選び、どれも選ばなければ見送るか、向聴を最も下げる牌を打つ。北を抜くルールでは、局ごとにランダムに選んだ席が北を抜きも打ちも
 * せずに残し、北で待つ手を作る（抜いた北への応答を起こすため）。
 *
 * <p>イベントは RecordedRound が決めた並びで書く。判断の時点から次の判断の時点までは、他家の応答が出そろって成立した立直 → 公開の
 * イベント → 表示牌 → 応答を求めずに通った打牌で成立した立直 → ツモの順。打牌と北抜きの物理牌は、その席の手から消えた牌。
 */
public final class RecordedGames {

  private final Variant variant;
  private final int seats;
  private final TileSet set;
  private final Engine engine;
  private final RuleWorkspace ws;
  private final ShapeAnalyzer analyzer;
  private final HandShape shape = new HandShape();
  private final Random random;

  /** 手番の席が手を選ぶ前に持っていた物理牌（物理 id ごとの印）。 */
  private final boolean[] held = new boolean[Tiles.PHYSICAL_IDS];

  private final List<RecordEvent> events = new ArrayList<>();
  private final List<Integer> ura = new ArrayList<>();
  private int eventsSeen;
  private int chanceSeen;
  private int riichiSeen;

  /** 今の局で北を抜きも打ちもせずに残す席のビットマスク。北を抜かないルールでは 0。 */
  private int keepNorths;

  /** あらかじめ決めた手のうち、まだ選んでいないもの（"席:手の名前"）。ランダムに打つ半荘では null。 */
  private final Deque<String> picks;

  private RecordedGames(Variant variant, long seed, Deque<String> picks) {
    this.variant = variant;
    seats = variant.seats();
    set = variant.tiles();
    engine = new Engine(variant);
    ws = new RuleWorkspace(variant);
    analyzer = new ShapeAnalyzer(variant);
    random = new Random(seed);
    this.picks = picks;
  }

  /** 乱数の種から 1 半荘を打って記録する。 */
  public static RecordedGame play(Variant variant, long seed, String id) {
    return new RecordedGames(variant, seed, null).game(id);
  }

  /**
   * 手で置いた牌山の 1 局を打ち、1 局だけの半荘として記録する。判断の時点ごとに、決めた手の残りの先頭がその席の合法手ならそれを選ぶ
   * （ダブロンのように、1 つの時点で続けて選べる）。ほかの席は、手番ならツモ切り（できなければ最初の合法手）、応答なら見送る。終局の点は、
   * 局末の次の卓の点。
   *
   * @param picks 順に選ぶ手（"席:手の名前"）。局が終わっても残っていれば AssertionError を投げる
   */
  public static RecordedGame scene(Scenes scene, String id, String... picks) {
    Variant variant = scene.variant();
    RecordedGames recorder = new RecordedGames(variant, 0, new ArrayDeque<>(List.of(picks)));
    recorder.engine.startRecordedRound(scene.roundStart(), scene.wall());
    RecordedRound round = recorder.round(recorder.engine.advance(recorder.ws));
    if (!recorder.picks.isEmpty()) {
      throw new AssertionError("The round ended before " + recorder.picks);
    }
    int[] scores = new int[variant.seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = recorder.engine.roundEnd().next().score(seat);
    }
    return new RecordedGame(variant, List.of(round), scores, unnamed(id, variant));
  }

  /** テストで作る半荘の出どころの情報。天鳳のルールで打った、打ち手の名前の無い半荘とする。 */
  public static RecordedGame.Meta unnamed(String id, Variant variant) {
    return new RecordedGame.Meta(id, RecordedGame.Source.TENHOU, Collections.nCopies(variant.seats(), ""));
  }

  private RecordedGame game(String id) {
    engine.startGame(random.nextLong(), random.nextBoolean() ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
    List<RecordedRound> rounds = new ArrayList<>();
    while (true) {
      rounds.add(round(engine.advance(ws)));
      if (engine.roundEnd().next().gameOver()) {
        int[] scores = new int[seats];
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] = engine.roundEnd().next().score(seat);
        }
        return new RecordedGame(variant, rounds, scores, unnamed(id, variant));
      }
    }
  }

  private RecordedRound round(Boundary first) {
    FullView full = engine.full();
    SeatView table = full.seat(0);
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = table.score(seat);
    }
    RoundStart start = new RoundStart(variant, table.round(), table.honba(), table.kyotaku(), scores);
    int firstDraw = full.chance().tile(0);
    int[][] deal = new int[seats][];
    for (int seat = 0; seat < seats; seat++) {
      deal[seat] = new int[Variant.HAND];
      int count = 0;
      for (int index = 0; index < set.size(); index++) {
        int tile = set.physicalId(index);
        if (tile != firstDraw && full.seat(seat).holds(tile)) {
          deal[seat][count++] = tile;
        }
      }
    }
    int indicator = table.indicator(0);
    events.clear();
    ura.clear();
    eventsSeen = 0;
    chanceSeen = 0;
    riichiSeen = 0;
    keepNorths = variant.has(RuleFlag.NUKI) ? random.nextInt(1 << seats) : 0;
    emit(false, TurnEvent.NONE);
    Boundary boundary = first;
    while (boundary == Boundary.DECISIONS) {
      Decisions decisions = engine.decisions();
      EngineDecisionKind kind = decisions.kind(0);
      boolean window = kind != EngineDecisionKind.TURN && kind != EngineDecisionKind.POST_CALL_DAHAI;
      int actor = window ? TurnEvent.NONE : decisions.seat(0);
      if (!window) {
        for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
          held[tile] = engine.view(actor).holds(tile);
        }
      }
      int[] choices = new int[decisions.size()];
      for (int slot = 0; slot < decisions.size(); slot++) {
        choices[slot] = picks == null ? choose(decisions, slot) : scripted(decisions, slot);
      }
      boundary = engine.commit(choices, ws);
      emit(window, actor);
    }
    return new RecordedRound(start, deal, indicator, events, result());
  }

  /** 前回書いた後に起きた公開のイベント・表示牌・立直の成立・ツモを、RecordedRound が決めた並びで書く。裏ドラ表示牌は局の結果に回す。 */
  private void emit(boolean window, int actor) {
    FullView full = engine.full();
    SeatView table = full.seat(0);
    EventLog log = table.events();
    ChanceLog chance = full.chance();
    int riichi = 0;
    for (int seat = 0; seat < seats; seat++) {
      riichi |= table.riichi(seat) != RiichiState.NONE ? 1 << seat : 0;
    }
    int accepted = riichi & ~riichiSeen;
    riichiSeen = riichi;
    if (window) {
      accept(accepted);
    }
    for (int event = eventsSeen; event < log.size(); event++) {
      events.add(publicEvent(log, event, actor));
    }
    eventsSeen = log.size();
    for (int index = chanceSeen; index < chance.size(); index++) {
      if (chance.kind(index) == ChanceKind.INDICATOR) {
        events.add(new RecordEvent.DoraReveal(chance.tile(index)));
      }
    }
    if (!window) {
      accept(accepted);
    }
    for (int index = chanceSeen; index < chance.size(); index++) {
      switch (chance.kind(index)) {
        case DRAW, RINSHAN -> events.add(new RecordEvent.Draw(chance.seat(index), chance.tile(index)));
        case URA -> ura.add(chance.tile(index));
        case INDICATOR -> {}
      }
    }
    chanceSeen = chance.size();
  }

  private void accept(int riichiSeats) {
    for (int mask = riichiSeats; mask != 0; mask &= mask - 1) {
      events.add(new RecordEvent.RiichiAccepted(Integer.numberOfTrailingZeros(mask)));
    }
  }

  private RecordEvent publicEvent(EventLog log, int event, int actor) {
    int seat = log.seat(event);
    return switch (log.kind(event)) {
      case DISCARD -> new RecordEvent.Discard(seat, removed(seat, actor), log.tsumogiri(event), log.riichi(event));
      case MELD -> {
        Meld meld = Meld.of(log.meld(event));
        yield meld.type() == Meld.Type.ANKAN
            ? new RecordEvent.ClosedKan(seat, meld.base())
            : new RecordEvent.Call(seat, meld.id());
      }
      case KAKAN -> new RecordEvent.AddedKan(seat, Meld.of(log.meld(event)).base());
      case NUKI -> new RecordEvent.Nuki(seat, removed(seat, actor));
    };
  }

  /** 手番の席の手から消えた物理牌（打った牌か抜いた北）を返す。 */
  private int removed(int seat, int actor) {
    if (seat != actor) {
      throw new AssertionError("Seat " + seat + " gives up a tile outside its turn");
    }
    for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
      if (held[tile] && !engine.view(seat).holds(tile)) {
        return tile;
      }
    }
    throw new AssertionError("No tile left the hand of seat " + seat);
  }

  private RoundResult result() {
    RoundEnd end = engine.roundEnd();
    RoundOutcome outcome = end.outcome();
    List<RoundResult.Win> wins = new ArrayList<>();
    for (int i = 0; i < outcome.winnerCount(); i++) {
      WinScore score = end.score(i);
      wins.add(
          new RoundResult.Win(
              outcome.winner(i),
              outcome.base(i),
              score.yaku(),
              score.han(),
              score.fu(),
              score.dora() + score.nuki(),
              score.aka(),
              score.ura(),
              outcome.pao(i)));
    }
    int[] deltas = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      deltas[seat] = end.delta(seat);
    }
    int[] uraIndicators = new int[ura.size()];
    for (int index = 0; index < uraIndicators.length; index++) {
      uraIndicators[index] = ura.get(index);
    }
    return new RoundResult(
        outcome.kind(),
        outcome.abortive(),
        wins,
        outcome.loser(),
        outcome.tenpaiMask(),
        outcome.nagashiMask(),
        deltas,
        uraIndicators);
  }

  /**
   * ランダムに手を選ぶ。8 回に 1 回は合法手から一様に選ぶ。ほかは和了・九種九牌・槓・北抜き・鳴き・立直を確率で選び、どれも選ばなければ
   * 見送るか、向聴を最も下げる牌を打つ。
   */
  private int choose(Decisions decisions, int slot) {
    int count = decisions.legalCount(slot);
    if (random.nextInt(8) == 0) {
      return random.nextInt(count);
    }
    int pass = -1;
    for (int index = 0; index < count; index++) {
      boolean take =
          switch (Action.of(decisions.legal(slot, index)).type()) {
            case TSUMO_AGARI, RON_AGARI -> random.nextInt(8) > 0;
            case KYUSHU_KYUHAI -> random.nextBoolean();
            case ANKAN, KAKAN -> random.nextInt(3) == 0;
            case CHI, PON, DAIMINKAN -> random.nextInt(4) == 0;
            case RIICHI_DAHAI -> random.nextInt(4) == 0;
            case NUKI -> !keepsNorths(decisions.seat(slot)) && random.nextBoolean();
            case PASS -> {
              pass = index;
              yield false;
            }
            case DAHAI -> false;
          };
      if (take) {
        return index;
      }
    }
    return pass >= 0 ? pass : bestDiscard(decisions, slot);
  }

  /** あらかじめ決めた手を選ぶ。残りの先頭がこの席の合法手ならそれを選び、そうでなければ手番ならツモ切り（できなければ最初の合法手）、応答なら見送る。 */
  private int scripted(Decisions decisions, int slot) {
    String head = picks.peekFirst();
    String prefix = decisions.seat(slot) + ":";
    int fallback = 0;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (head != null && head.equals(prefix + action)) {
        picks.removeFirst();
        return index;
      }
      boolean tsumogiri = action.type() == Action.Type.DAHAI && action.selection().tsumogiri();
      if (tsumogiri || action.type() == Action.Type.PASS) {
        fallback = index;
      }
    }
    return fallback;
  }

  /** 向聴を最も下げる打牌を選ぶ（同じ向聴ならランダム）。北を残す席は北を打たない。ほかに打てる牌が無ければ最初の合法手を選ぶ。 */
  private int bestDiscard(Decisions decisions, int slot) {
    int seat = decisions.seat(slot);
    boolean keeper = keepsNorths(seat);
    SeatView view = engine.view(seat);
    shape.clear();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < view.count(kind); copy++) {
        shape.add(kind);
      }
    }
    for (int index = 0; index < view.meldCount(seat); index++) {
      shape.addMeld();
    }
    int best = Integer.MAX_VALUE;
    int chosen = 0;
    int ties = 0;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Action action = Action.of(decisions.legal(slot, index));
      if (action.type() != Action.Type.DAHAI || keeper && action.tile() == Tiles.PEI) {
        continue;
      }
      shape.remove(action.tile());
      int shanten = analyzer.shanten(shape);
      shape.add(action.tile());
      if (shanten < best) {
        best = shanten;
        chosen = index;
        ties = 1;
      } else if (shanten == best && random.nextInt(++ties) == 0) {
        chosen = index;
      }
    }
    return chosen;
  }

  private boolean keepsNorths(int seat) {
    return (keepNorths & 1 << seat) != 0;
  }
}
