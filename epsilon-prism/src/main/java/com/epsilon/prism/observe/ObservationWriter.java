package com.epsilon.prism.observe;

import static com.epsilon.prism.observe.ObservationSchema.CALL_DISCARD;
import static com.epsilon.prism.observe.ObservationSchema.HAND;
import static com.epsilon.prism.observe.ObservationSchema.HEADER;
import static com.epsilon.prism.observe.ObservationSchema.LEGAL;
import static com.epsilon.prism.observe.ObservationSchema.SEAT;
import static com.epsilon.prism.observe.ObservationSchema.WIN;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.EngineActionBuffer;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.PostCallDiscards;
import com.epsilon.engine.VisibleHandScoreBuffer;
import com.epsilon.engine.WinSettlementProjection;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.observe.ObservationSchema.CallDiscardBits;
import com.epsilon.prism.observe.ObservationSchema.HandBits;
import com.epsilon.prism.observe.ObservationSchema.HeaderBits;
import com.epsilon.prism.observe.ObservationSchema.LastEvent;
import com.epsilon.prism.observe.ObservationSchema.LegalBits;
import com.epsilon.prism.observe.ObservationSchema.SeatBits;
import com.epsilon.prism.observe.ObservationSchema.Section;
import com.epsilon.prism.observe.ObservationSchema.WinBits;
import java.util.Arrays;
import java.util.List;

/**
 * 公開の観測と合法手から記録 O を置き場に書く。スレッドに 1 つ持ち（作業域を持つのでスレッド安全でない）、収集・評価・Reviewer で同じ
 * 関数を通す。
 *
 * <p>解析はしない。規則の事実はエンジンの関数を読むだけにする: 判断の種類は呼ぶ側が渡し（live は判断点の {@code kind()}、牌譜は
 * {@link EngineDecisionKind#of}）、チー・ポンの後の打牌は {@link PostCallDiscards}、即時の和了は {@link WinSettlementProjection}（どちらも
 * {@link PublicObservation} が借用局面を渡す）。公開の出来事はエンジンの記録（{@link PublicObservation#events}）を {@link EventRecords}
 * でそのまま書く（鳴かれた打牌と副露の整合はエンジンが記録を書くときに確かめる）。規則の位置のない槓ドラの表示は例外にする。
 *
 * <p>置き場は 0 で始まり、0 のビットの欄は最小の値を読むので、最小の値は書かない。1 語に収まる記録（席・出来事・合法手）は欄を語の上で
 * 詰めてから丸ごと書く（{@link BitLayout#putRecord}）。置き場を伸ばすときを除いて割り当てをしない。作業域はこのクラスの中にだけ持つ。
 */
public final class ObservationWriter {

  /** チー・ポンを適用した後の自家の手。チー・ポンの手ごとに上書きし、その手の打牌を数える間だけ有効。 */
  private final ActionEffect.ProjectedHand afterCall = new ActionEffect.ProjectedHand();

  /** 1 つのチー・ポンの後の打牌（{@link PostCallDiscards} の出力）。手ごとに上書きし、番号を写すまでだけ有効。 */
  private final EngineActionBuffer discards = new EngineActionBuffer();

  /** 和了の射影の点数計算の作業域。このライターだけが使う。 */
  private final HandScoreEvaluator scoreEvaluator = new HandScoreEvaluator();

  /** 1 つのロン・ツモの公開の射影。和了の手ごとに上書きし、その手の欄を書くまでだけ有効。 */
  private final WinSettlementProjection projection = new WinSettlementProjection();

  /** 合法手ごとのチー・ポンの後の打牌の数（チー・ポンでない手は 0）。 */
  private final byte[] callDiscardCounts = new byte[Limits.MAX_LEGAL_ACTIONS];

  /** チー・ポンの後の打牌の手の番号（合法手の順に続けて並べる。O の CALL_DISCARD の節と同じ順）。 */
  private final short[] callDiscardActions =
      new short[Limits.MAX_LEGAL_ACTIONS * Limits.MAX_CALL_DISCARDS];

  private final Observation written = new Observation();
  private final long[] headerDraft = new long[HEADER.longs(1)];
  private final int[] start = new int[ObservationSchema.SECTIONS.length];
  private long[] words;
  private int base;
  private int observer;

  /**
   * O を置き場に書く。{@code o} と {@code legal} は呼び出しの間だけ借りる。
   *
   * @param o 判断する席の公開の観測
   * @param kind 判断の種類（live は判断点の {@code kind()}、牌譜は {@link EngineDecisionKind#of}。どちらも同じ値）
   * @param legal 合法手（エンジンの順、1 つ以上）
   * @param into 書く先
   * @return 置き場の中の O の先頭
   * @throws IllegalArgumentException 合法手がないか、値が O の欄の範囲（規則の上限）の外のとき
   * @throws IllegalStateException 観測が O の決まり（槓ドラの表示の位置）と合わないとき
   */
  public int capture(
      PublicObservation o, EngineDecisionKind kind, List<Action> legal, ObservationArena into) {
    if (legal.isEmpty()) {
      throw new IllegalArgumentException("no legal action");
    }
    if (legal.size() > Limits.MAX_LEGAL_ACTIONS) {
      throw new IllegalArgumentException("legal actions exceed the O field: " + legal.size());
    }
    observer = o.observer();
    int callDiscards = projectCalls(o, legal);
    writeHeader(o, kind, legal, callDiscards, countWins(legal));
    base = into.allocate(ObservationSchema.place(headerDraft, 0, start));
    words = into.words();
    System.arraycopy(headerDraft, 0, words, base, headerDraft.length);
    writeSeats(o);
    writeHand(o.hand(observer));
    EventRecords.writeObservation(o.events(), observer, words, start(Section.EVENT));
    writeLegal(legal);
    writeWins(o, legal);

    write(HEADER, base, 0, HeaderBits.TRANSITIONS, written.bind(words, base).countTransitions());
    for (int kan = 0; kan < written.kanDoraIndicators(); kan++) {
      written.kanDoraEvent(kan);
    }
    return base;
  }

  /** 頭を下書きに書く（記録の数を含む。遷移の数は合法手を書いた後に置き場へ書く）。 */
  private void writeHeader(
      PublicObservation o, EngineDecisionKind kind, List<Action> legal, int callDiscards, int wins) {
    Arrays.fill(headerDraft, 0L);
    TurnEvent last = o.turnEvent();
    header(HeaderBits.DECISION_KIND, kind.ordinal());
    header(HeaderBits.OBSERVER, observer);
    header(HeaderBits.DEALER, relative(o.dealer()));
    header(HeaderBits.KYOKU, o.roundIndex());
    header(HeaderBits.HONBA, o.honba());
    header(HeaderBits.KYOTAKU, o.riichiSticks());
    header(HeaderBits.WALL, o.remainingWallTiles());
    flag(HeaderBits.FURITEN_TEMPORARY, o.isTemporaryFuriten(observer));
    writeLastEvent(last);
    header(HeaderBits.HAND_AKA, o.hand(observer).concealedAkaMask());
    PublicObservation.VisibleDora dora = o.doraState();
    int indicators = dora.indicatorCount();
    header(HeaderBits.INDICATORS, indicators);
    for (int index = 0; index < indicators; index++) {
      HEADER.put(headerDraft, 0, 0, HeaderBits.INDICATOR, index, dora.indicatorTileType(index));
    }
    header(HeaderBits.INDICATOR_AKA, dora.visibleIndicatorAkaMask());
    header(HeaderBits.EVENTS, o.events().size());
    header(HeaderBits.LEGAL_ACTIONS, legal.size());
    header(HeaderBits.CALL_DISCARDS, callDiscards);
    header(HeaderBits.WINS, wins);
  }

  private void writeLastEvent(TurnEvent last) {
    switch (last) {
      case TurnEvent.Draw draw -> {
        lastEvent(LastEvent.DRAW, draw);
        flag(HeaderBits.LAST_EVENT_RINSHAN, draw.isRinshanDraw());
        flag(HeaderBits.DEFERRED_DORA, draw.doraRevealPending());
      }
      case TurnEvent.Discard discard -> lastEvent(LastEvent.DISCARD, discard);
      case TurnEvent.KanAttempt kan -> {
        lastEvent(LastEvent.KAN_ATTEMPT, kan);
        header(HeaderBits.LAST_KAN_KIND, kan.kanKind().ordinal());
        flag(HeaderBits.DEFERRED_DORA, kan.doraRevealPending());
      }
      case TurnEvent.None ignored -> {}
    }
  }

  private void lastEvent(LastEvent kind, TurnEvent.TileEvent event) {
    header(HeaderBits.LAST_EVENT, kind.ordinal());
    header(HeaderBits.LAST_EVENT_SEAT, relative(event.player()));
    header(HeaderBits.LAST_EVENT_TILE, event.tileType());
    flag(HeaderBits.LAST_EVENT_AKA, event.isAkaTile());
  }

  private void writeSeats(PublicObservation o) {
    int at = start(Section.SEAT);
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int player = absolute(seat);
      long record = SEAT.with(0L, SeatBits.SCORE, ObservationSchema.encodePoints(o.score(player)));
      record = SEAT.with(record, SeatBits.RIICHI, riichi(o, player).ordinal());
      record = SEAT.with(record, SeatBits.IPPATSU, o.isIppatsu(player) ? 1 : 0);
      SEAT.putRecord(words, at, seat, record);
    }
  }

  /** 手にある牌の種類だけ枚数を書く。 */
  private void writeHand(PublicObservation.VisibleHand hand) {
    int at = start(Section.HAND);
    for (long tiles = hand.concealedTileTypeMask(); tiles != 0L; tiles &= tiles - 1) {
      int tile = Long.numberOfTrailingZeros(tiles);
      HAND.put(words, at, 0, HandBits.COUNT, tile, hand.count(tile));
    }
  }

  /**
   * チー・ポンごとに鳴いた後の手を作り、打てる牌（エンジンの決まり {@link PostCallDiscards}、エンジンの遷移の順）を作業域へ写す。
   *
   * @return 打牌の数の和（O の頭の CALL_DISCARDS）
   */
  private int projectCalls(PublicObservation o, List<Action> legal) {
    int total = 0;
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      Action call = legal.get(action);
      if (!ObservationSchema.hasCallDiscards(call.type())) {
        callDiscardCounts[action] = 0;
        continue;
      }
      int count = PostCallDiscards.generateInto(afterCall, o.projectCall(call, afterCall), discards);
      if (count < 1) {
        throw new IllegalStateException("call has no discard after it: " + call);
      }
      if (count > Limits.MAX_CALL_DISCARDS) {
        throw new IllegalArgumentException("call discards exceed the O field: " + count);
      }
      for (int discard = 0; discard < count; discard++) {
        callDiscardActions[total + discard] = (short) discards.get(discard).toIndex();
      }
      callDiscardCounts[action] = (byte) count;
      total += count;
    }
    return total;
  }

  private static int countWins(List<Action> legal) {
    int wins = 0;
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      if (ObservationSchema.hasWin(legal.get(action).type())) {
        wins++;
      }
    }
    return wins;
  }

  private void writeLegal(List<Action> legal) {
    int at = start(Section.LEGAL);
    int discardsAt = start(Section.CALL_DISCARD);
    int next = 0;
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      Action chosen = legal.get(action);
      long record = LEGAL.with(0L, LegalBits.ACTION, chosen.toIndex());
      if (ObservationSchema.hasCallDiscards(chosen.type())) {
        int count = callDiscardCounts[action];
        record = LEGAL.with(record, LegalBits.CALL_DISCARDS, count);
        for (int discard = 0; discard < count; discard++, next++) {
          write(CALL_DISCARD, discardsAt, next, CallDiscardBits.ACTION, callDiscardActions[next]);
        }
      }
      LEGAL.putRecord(words, at, action, record);
    }
  }

  /** 即時の和了ごとに公開の射影（エンジンの {@link WinSettlementProjection}）を求めて書く。 */
  private void writeWins(PublicObservation o, List<Action> legal) {
    int at = start(Section.WIN);
    int win = 0;
    for (int action = 0, actions = legal.size(); action < actions; action++) {
      Action chosen = legal.get(action);
      if (!ObservationSchema.hasWin(chosen.type())) {
        continue;
      }
      o.projectWin(chosen, scoreEvaluator, projection);
      VisibleHandScoreBuffer score = projection.visibleScore();
      write(WIN, at, win, WinBits.HAN, score.visibleHan());
      write(WIN, at, win, WinBits.FU_CODE, ObservationSchema.encodeFu(score.visibleHan(), score.fu()));
      write(WIN, at, win, WinBits.YAKUMAN, score.yakumanMultiplier());
      flag(WIN, at, win, WinBits.PAO, projection.paoApplies());
      flag(WIN, at, win, WinBits.URA_ELIGIBLE, projection.uraEligible());
      write(WIN, at, win, WinBits.URA_INDICATORS, projection.uraIndicatorCount());
      write(WIN, at, win, WinBits.SETTLEMENT, projection.settlementAssumption().ordinal());
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        int player = absolute(seat);
        write(WIN, at, win, WinBits.RANK, seat, projection.projectedRank(player));
        write(
            WIN,
            at,
            win,
            WinBits.PAYMENT,
            seat,
            ObservationSchema.encodePoints(projection.paymentFloor(player)));
      }
      win++;
    }
  }

  /** 節の先頭（置き場の添字）。 */
  private int start(Section section) {
    return base + start[section.ordinal()];
  }

  private int relative(int player) {
    return Math.floorMod(player - observer, GameState.NUM_PLAYERS);
  }

  private int absolute(int seat) {
    return (observer + seat) % GameState.NUM_PLAYERS;
  }

  /** 頭の下書きに書く（下書きは 0 で始まるので、最小の値は書かない）。 */
  private void header(HeaderBits field, int value) {
    if (value != HEADER.min(field)) {
      HEADER.put(headerDraft, 0, 0, field, value);
    }
  }

  private void flag(HeaderBits field, boolean value) {
    if (value) {
      HEADER.put(headerDraft, 0, 0, field, 1);
    }
  }

  /** 置き場に書く（0 のビットの欄は最小の値を読むので、最小の値は書かない）。 */
  private <E extends Enum<E> & BitField> void write(
      BitLayout<E> layout, int at, int record, E field, int value) {
    if (value != layout.min(field)) {
      layout.put(words, at, record, field, value);
    }
  }

  /** 置き場の繰り返しのある欄に書く（最小の値は書かない）。 */
  private <E extends Enum<E> & BitField> void write(
      BitLayout<E> layout, int at, int record, E field, int index, int value) {
    if (value != layout.min(field)) {
      layout.put(words, at, record, field, index, value);
    }
  }

  private <E extends Enum<E> & BitField> void flag(
      BitLayout<E> layout, int at, int record, E field, boolean value) {
    if (value) {
      layout.put(words, at, record, field, 1);
    }
  }

  private static RiichiState riichi(PublicObservation o, int player) {
    if (o.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return o.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }
}
