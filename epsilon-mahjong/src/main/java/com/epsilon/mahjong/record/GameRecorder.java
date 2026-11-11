package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * エンジンで打った半荘を牌譜に書き取る。エンジンが止まるたび（局の最初の判断の時点と、{@code commit} の後の判断の時点か局末）に
 * {@link #advanced} を呼ぶ。局末には {@link #roundEnded} を、終局には {@link #gameEnded} を呼ぶ。どの打ち手が打っても使える。
 *
 * <p>止まるたびに、前に止まった時点より後に起きた公開のイベント・表示牌・立直の成立・ツモを足し、局末に RecordedRound の契約どおりの
 * 順に並べ直す（{@link RecordedRound#contractOrder}）。打牌と北抜きの物理牌は、その席の手から消えた牌で決める。局の結果と裏ドラ表示牌は
 * エンジンの局末の情報から作る。1 つの半荘は 1 つのスレッドで書き取る。
 */
public final class GameRecorder {

  private final Variant variant;
  private final int seats;
  private final TileSet set;

  /** 前に止まった時点で、席ごとに手にあった物理牌。 */
  private final boolean[][] held;

  private final List<RecordedRound> rounds = new ArrayList<>();
  private final List<RecordEvent> events = new ArrayList<>();
  private final List<Integer> ura = new ArrayList<>();
  private RoundStart start;
  private int[][] deal;
  private int indicator;

  /** 局を書き取っている途中なら true。 */
  private boolean open;

  private int eventsSeen;
  private int chanceSeen;
  private int riichiSeen;

  public GameRecorder(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    seats = variant.seats();
    set = variant.tiles();
    held = new boolean[seats][Tiles.PHYSICAL_IDS];
  }

  public Variant variant() {
    return variant;
  }

  /**
   * エンジンが止まったときに呼ぶ。局を書き取っていなければ局の始まり（配牌・最初の表示牌・親の最初のツモ）を読み、書き取っていれば
   * 前の時点からのイベントを足す。
   *
   * @throws IllegalArgumentException ほかのルールのビューを渡したとき
   * @throws IllegalStateException 打牌か北抜きの物理牌が手から決まらないとき
   */
  public void advanced(FullView full) {
    requireVariant(full);
    if (open) {
      emit(full);
    } else {
      begin(full);
    }
    remember(full);
  }

  /**
   * 局末に呼ぶ。残りのイベントを足し、その局の記録を仕上げる。
   *
   * @throws IllegalStateException 局を書き取っていないとき
   */
  public void roundEnded(FullView full, RoundEnd end) {
    requireVariant(full);
    if (!open) {
      throw new IllegalStateException("No round is being recorded");
    }
    emit(full);
    rounds.add(new RecordedRound(start, deal, indicator, RecordedRound.contractOrder(events), result(end)));
    open = false;
  }

  /**
   * 終局に呼ぶ。書き取った局をまとめて半荘の記録を返し、次の半荘に備えて書き取った局を消す。
   *
   * @param last 終局の卓（残りの供託をトップに渡した後の点）
   * @param meta 牌譜の出どころ
   * @throws IllegalStateException 局の途中のときか、局が 1 つも無いとき
   */
  public RecordedGame gameEnded(TableState last, RecordedGame.Meta meta) {
    if (open || rounds.isEmpty()) {
      throw new IllegalStateException(open ? "The last round has not ended" : "No round was recorded");
    }
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = last.score(seat);
    }
    RecordedGame game = new RecordedGame(variant, rounds, scores, meta);
    rounds.clear();
    return game;
  }

  private void begin(FullView full) {
    SeatView table = full.seat(0);
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = table.score(seat);
    }
    start = new RoundStart(variant, table.round(), table.honba(), table.kyotaku(), scores);
    int firstDraw = full.chance().tile(0);
    deal = new int[seats][];
    for (int seat = 0; seat < seats; seat++) {
      deal[seat] = new int[Variant.HAND];
      int count = 0;
      for (int index = 0; index < set.size(); index++) {
        int tile = set.physicalId(index);
        if (tile != firstDraw && full.seat(seat).holds(tile)) {
          deal[seat][count++] = tile;
        }
      }
      if (count != Variant.HAND) {
        throw new IllegalStateException("Seat " + seat + " was dealt " + count + " tiles");
      }
    }
    indicator = table.indicator(0);
    events.clear();
    ura.clear();
    eventsSeen = 0;
    chanceSeen = 0;
    riichiSeen = 0;
    open = true;
    emit(full);
  }

  /**
   * 前に止まった時点より後に起きたことを、公開のイベント、表示牌、成立した立直、ツモの順に足す。裏ドラ表示牌はイベントにせず、局の
   * 結果のために取っておく。
   */
  private void emit(FullView full) {
    SeatView table = full.seat(0);
    EventLog log = table.events();
    for (int event = eventsSeen; event < log.size(); event++) {
      events.add(publicEvent(full, log, event));
    }
    eventsSeen = log.size();
    ChanceLog chance = full.chance();
    for (int index = chanceSeen; index < chance.size(); index++) {
      if (chance.kind(index) == ChanceKind.INDICATOR) {
        events.add(new RecordEvent.DoraReveal(chance.tile(index)));
      }
    }
    int riichi = 0;
    for (int seat = 0; seat < seats; seat++) {
      riichi |= table.riichi(seat) != RiichiState.NONE ? 1 << seat : 0;
    }
    for (int accepted = riichi & ~riichiSeen; accepted != 0; accepted &= accepted - 1) {
      events.add(new RecordEvent.RiichiAccepted(Integer.numberOfTrailingZeros(accepted)));
    }
    riichiSeen = riichi;
    for (int index = chanceSeen; index < chance.size(); index++) {
      switch (chance.kind(index)) {
        case DRAW, RINSHAN -> events.add(new RecordEvent.Draw(chance.seat(index), chance.tile(index)));
        case URA -> ura.add(chance.tile(index));
        case INDICATOR -> {}
      }
    }
    chanceSeen = chance.size();
  }

  private RecordEvent publicEvent(FullView full, EventLog log, int event) {
    int seat = log.seat(event);
    return switch (log.kind(event)) {
      case DISCARD -> new RecordEvent.Discard(seat, removed(full, seat), log.tsumogiri(event), log.riichi(event));
      case MELD -> {
        Meld meld = Meld.of(log.meld(event));
        yield meld.type() == Meld.Type.ANKAN
            ? new RecordEvent.ClosedKan(seat, meld.base())
            : new RecordEvent.Call(seat, meld.id());
      }
      case KAKAN -> new RecordEvent.AddedKan(seat, Meld.of(log.meld(event)).base());
      case NUKI -> new RecordEvent.Nuki(seat, removed(full, seat));
    };
  }

  /**
   * 前に止まった時点では席の手にあり、いまは無い物理牌（打った牌か抜いた北）を返す。そういう牌がちょうど 1 枚でなければ
   * IllegalStateException を投げる。
   */
  private int removed(FullView full, int seat) {
    SeatView view = full.seat(seat);
    int removed = -1;
    for (int index = 0; index < set.size(); index++) {
      int tile = set.physicalId(index);
      if (held[seat][tile] && !view.holds(tile)) {
        if (removed >= 0) {
          throw new IllegalStateException("Seat " + seat + " gave up more than one tile in one step");
        }
        removed = tile;
      }
    }
    if (removed < 0) {
      throw new IllegalStateException("No tile left the hand of seat " + seat);
    }
    held[seat][removed] = false;
    return removed;
  }

  private void remember(FullView full) {
    for (int seat = 0; seat < seats; seat++) {
      SeatView view = full.seat(seat);
      for (int index = 0; index < set.size(); index++) {
        int tile = set.physicalId(index);
        held[seat][tile] = view.holds(tile);
      }
    }
  }

  private RoundResult result(RoundEnd end) {
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

  private void requireVariant(FullView full) {
    if (full.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + full.variant());
    }
  }
}
