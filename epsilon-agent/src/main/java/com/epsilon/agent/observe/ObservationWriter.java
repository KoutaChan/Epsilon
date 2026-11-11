package com.epsilon.agent.observe;

import com.epsilon.agent.observe.ObservationSchema.CallDiscardBits;
import com.epsilon.agent.observe.ObservationSchema.HandBits;
import com.epsilon.agent.observe.ObservationSchema.HeaderBits;
import com.epsilon.agent.observe.ObservationSchema.LastEvent;
import com.epsilon.agent.observe.ObservationSchema.LegalBits;
import com.epsilon.agent.observe.ObservationSchema.SeatBits;
import com.epsilon.agent.observe.ObservationSchema.Section;
import com.epsilon.agent.observe.ObservationSchema.WinBits;
import com.epsilon.agent.observe.codec.AkaMask;
import com.epsilon.agent.observe.codec.BitField;
import com.epsilon.agent.observe.codec.BitLayout;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Legal;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 判断する席から見た局面と、席ごとの判断をもとに記録 O を作り、バッチに足す。作業領域を持つのでスレッドセーフではなく、スレッドごとに
 * 1 つ持つ。収集・評価・Reviewer・クライアントは、どれもこの同じ処理で O を作る。ウォームアップの後は、バッチを伸ばすときのほかは
 * メモリを確保しない。
 *
 * <p>ここでは解析をしない。ルールで決まる値は、ルールの関数の結果を読むだけにする。チー・ポンの後の打牌は
 * {@code Legal.postCallDiscards} から、選べる和了は WinSettlementProjection から読む。どちらも席から見た局面だけを読む。公開の
 * イベントは、エンジンの記録を {@link EventRecords} でそのまま書く。槓ドラの表示の位置が、O から求めた位置
 * （{@link Observation#kanDoraEvent}）と合わなければ IllegalStateException を投げる。
 *
 * <p>O は作業領域に書き終えてからバッチにコピーするので、例外で終わった書き込みはバッチに何も残さない。作業領域は書く分を 0 で
 * 埋めてから書く。全てのビットが 0 のフィールドは最小の値として読めるので、最小の値は書かない。{@code long} 1 つに収まる記録
 * （席・イベント・合法手）は、フィールドを {@code long} の中で詰めてから丸ごと書く。
 */
public final class ObservationWriter {

  private final Variant variant;
  private final ObservationSchema schema;
  private final RuleWorkspace ws;
  private final WinSettlementProjection projection;

  /** 1 回のチー・ポンの後に選べる打牌（{@code Legal.postCallDiscards} が書く先）。合法手ごとに上書きする。 */
  private final short[] discards = new short[ObservationSchema.MAX_CALL_DISCARDS];

  /** 合法手ごとのチー・ポンの後の打牌の数（チー・ポンでない手は 0）。 */
  private final byte[] callDiscardCounts = new byte[ObservationSchema.MAX_LEGAL];

  /** チー・ポンの後に選べる打牌ごとの行動の番号。合法手の順に続けて並べ、O のセクション（{@link Section#CALL_DISCARD}）と同じ順にする。 */
  private final short[] callDiscardActions =
      new short[ObservationSchema.MAX_LEGAL * ObservationSchema.MAX_CALL_DISCARDS];

  /** 書いている O（位置 0 から書く）。どの O も入る長さを取る。 */
  private final long[] words;

  private final int headerLongs;
  private final int[] start = new int[ObservationSchema.sections()];
  private final Observation written;
  private int observer;

  public ObservationWriter(Variant variant) {
    this.variant = variant;
    schema = ObservationSchema.of(variant);
    ws = new RuleWorkspace(variant);
    projection = new WinSettlementProjection(variant);
    words = new long[schema.maxLongs()];
    headerLongs = schema.header.longs(1);
    written = new Observation(variant);
  }

  /**
   * 判断する席の 1 つについて O を作り、バッチの末尾に足す。ルールか席が合わないとき、値が O のフィールドの範囲の外のときは
   * IllegalArgumentException を投げる。
   *
   * @param view 判断する席から見た、同じ判断の時点の局面（呼び出しの間だけ使う）
   * @param decisions 席ごとの判断（呼び出しの間だけ使う）
   * @param slot {@code decisions} の中で、何番目の判断する席か
   * @param into 足す先のバッチ
   * @return バッチの中の O の番号
   */
  public int capture(SeatView view, Decisions decisions, int slot, ObservationBatch into) {
    if (view.variant() != variant || into.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + view.variant() + " and " + into.variant());
    }
    observer = view.seat();
    if (decisions.seat(slot) != observer) {
      throw new IllegalArgumentException(
          "Slot " + slot + " belongs to seat " + decisions.seat(slot) + ", not " + observer);
    }
    int legal = decisions.legalCount(slot);
    int callDiscards = projectCalls(view, decisions, slot, legal);
    Arrays.fill(words, 0, headerLongs, 0L);
    writeHeader(view, decisions, slot, legal, callDiscards);
    int longs = schema.place(words, 0, start);
    Arrays.fill(words, headerLongs, longs, 0L);
    writeSeats(view);
    writeHand(view);
    EventRecords.write(schema, view.events(), observer, words, start[Section.EVENT.ordinal()]);
    writeLegal(decisions, slot, legal);
    writeWins(view, decisions, slot, legal);
    written.bind(words, 0);
    write(schema.header, 0, 0, HeaderBits.TRANSITIONS, written.countTransitions());
    for (int kan = 0; kan < written.kanDoraIndicators(); kan++) {
      int before = written.kanDoraEvent(kan);
      if (before + 1 != view.indicatorEvent(kan + 1)) {
        throw new IllegalStateException(
            "Kan dora " + kan + " follows public event " + (view.indicatorEvent(kan + 1) - 1) + ", not " + before);
      }
    }
    return into.add(words, 0, longs);
  }

  /** ヘッダーを書く（記録の数を含む。遷移の数は、合法手を書いた後で書く）。 */
  private void writeHeader(SeatView view, Decisions decisions, int slot, int legal, int callDiscards) {
    header(HeaderBits.DECISION_KIND, schema.decisionKinds.index(decisions.kind(slot)));
    header(HeaderBits.OBSERVER, observer);
    header(HeaderBits.DEALER, relative(view.dealer()));
    header(HeaderBits.KYOKU, view.round());
    header(HeaderBits.HONBA, view.honba());
    header(HeaderBits.KYOTAKU, view.kyotaku());
    header(HeaderBits.WALL, view.liveLeft());
    flag(HeaderBits.FURITEN_TEMPORARY, view.furiten());
    writeLastEvent(view.last());
    header(HeaderBits.HAND_AKA, AkaMask.suitMask(Legal.redFives(view)));
    TileSet set = variant.tiles();
    int indicators = view.indicatorCount();
    long redIndicators = 0;
    header(HeaderBits.INDICATORS, indicators);
    for (int index = 0; index < indicators; index++) {
      int tile = view.indicator(index);
      write(schema.header, 0, 0, HeaderBits.INDICATOR, index, Tiles.kind(tile));
      if (set.red(tile)) {
        redIndicators |= Tiles.bit(Tiles.kind(tile));
      }
    }
    header(HeaderBits.INDICATOR_AKA, AkaMask.suitMask(redIndicators));
    header(HeaderBits.EVENTS, view.events().size());
    header(HeaderBits.LEGAL_ACTIONS, legal);
    header(HeaderBits.CALL_DISCARDS, callDiscards);
    header(HeaderBits.WINS, countWins(decisions, slot, legal));
  }

  private void writeLastEvent(TurnEvent last) {
    switch (last.kind()) {
      case NONE -> {}
      case DRAW -> {
        lastEvent(LastEvent.DRAW, last);
        flag(HeaderBits.LAST_EVENT_RINSHAN, last.rinshan());
        flag(HeaderBits.DEFERRED_DORA, last.doraPending());
      }
      case DISCARD -> lastEvent(LastEvent.DISCARD, last);
      case KAN -> {
        lastEvent(LastEvent.KAN_ATTEMPT, last);
        header(HeaderBits.LAST_KAN_KIND, schema.kanKinds.index(last.kanKind()));
        flag(HeaderBits.DEFERRED_DORA, last.doraPending());
      }
      case NUKI -> {
        lastEvent(LastEvent.NUKI, last);
        flag(HeaderBits.DEFERRED_DORA, last.doraPending());
      }
    }
  }

  private void lastEvent(LastEvent kind, TurnEvent last) {
    header(HeaderBits.LAST_EVENT, schema.lastEvents.index(kind));
    header(HeaderBits.LAST_EVENT_SEAT, relative(last.seat()));
    header(HeaderBits.LAST_EVENT_TILE, last.tile());
    flag(HeaderBits.LAST_EVENT_AKA, last.red());
  }

  private void writeSeats(SeatView view) {
    int at = start[Section.SEAT.ordinal()];
    for (int seat = 0; seat < variant.seats(); seat++) {
      int player = absolute(seat);
      long record = schema.seat.with(0L, SeatBits.SCORE, ObservationSchema.encodePoints(view.score(player)));
      record = schema.seat.with(record, SeatBits.RIICHI, schema.riichiStates.index(view.riichi(player)));
      record = schema.seat.with(record, SeatBits.IPPATSU, view.ippatsu(player) ? 1 : 0);
      schema.seat.putRecord(words, at, seat, record);
    }
  }

  /** 純手牌にある種類だけ枚数を書く。 */
  private void writeHand(SeatView view) {
    int at = start[Section.HAND.ordinal()];
    for (long kinds = variant.tiles().kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      write(schema.hand, at, 0, HandBits.COUNT, kind, view.count(kind));
    }
  }

  /**
   * チー・ポンの合法手ごとに、鳴いた後の打牌を作業領域へ書く。
   *
   * @return 打牌の数の和（ヘッダーに書く、チー・ポンの後の打牌の数）
   */
  private int projectCalls(SeatView view, Decisions decisions, int slot, int legal) {
    int total = 0;
    for (int action = 0; action < legal; action++) {
      int code = decisions.legal(slot, action);
      if (!ObservationSchema.hasCallDiscards(Action.of(code).type())) {
        callDiscardCounts[action] = 0;
        continue;
      }
      int count = Legal.postCallDiscards(view, code, discards);
      System.arraycopy(discards, 0, callDiscardActions, total, count);
      callDiscardCounts[action] = (byte) count;
      total += count;
    }
    return total;
  }

  private static int countWins(Decisions decisions, int slot, int legal) {
    int wins = 0;
    for (int action = 0; action < legal; action++) {
      if (ObservationSchema.hasWin(Action.of(decisions.legal(slot, action)).type())) {
        wins++;
      }
    }
    return wins;
  }

  private void writeLegal(Decisions decisions, int slot, int legal) {
    int at = start[Section.LEGAL.ordinal()];
    int discardsAt = start[Section.CALL_DISCARD.ordinal()];
    int next = 0;
    for (int action = 0; action < legal; action++) {
      long record = schema.legal.with(0L, LegalBits.ACTION, decisions.legal(slot, action));
      int count = callDiscardCounts[action];
      if (count > 0) {
        record = schema.legal.with(record, LegalBits.CALL_DISCARDS, count);
        for (int discard = 0; discard < count; discard++, next++) {
          write(schema.callDiscard, discardsAt, next, CallDiscardBits.ACTION, callDiscardActions[next]);
        }
      }
      schema.legal.putRecord(words, at, action, record);
    }
  }

  /** ロン・ツモの合法手ごとに、和了したときの精算（WinSettlementProjection）を求めて書く。 */
  private void writeWins(SeatView view, Decisions decisions, int slot, int legal) {
    int at = start[Section.WIN.ordinal()];
    int win = 0;
    for (int action = 0; action < legal; action++) {
      int code = decisions.legal(slot, action);
      if (!ObservationSchema.hasWin(Action.of(code).type())) {
        continue;
      }
      projection.load(view, code, ws);
      WinScore score = projection.score();
      write(schema.win, at, win, WinBits.HAN, score.han());
      write(schema.win, at, win, WinBits.FU_CODE, ObservationSchema.encodeFu(score.han(), score.fu()));
      write(schema.win, at, win, WinBits.YAKUMAN, score.yakuman());
      write(schema.win, at, win, WinBits.PAO, projection.pao() != RoundOutcome.NO_SEAT ? 1 : 0);
      write(schema.win, at, win, WinBits.URA_ELIGIBLE, projection.uraEligible() ? 1 : 0);
      write(schema.win, at, win, WinBits.URA_INDICATORS, projection.uraIndicators());
      write(schema.win, at, win, WinBits.SETTLEMENT, schema.settlements.index(projection.assumption()));
      for (int seat = 0; seat < variant.seats(); seat++) {
        int player = absolute(seat);
        write(schema.win, at, win, WinBits.RANK, seat, projection.rank(player));
        write(schema.win, at, win, WinBits.PAYMENT, seat, ObservationSchema.encodePoints(projection.payment(player)));
      }
      win++;
    }
  }

  private int relative(int player) {
    return Math.floorMod(player - observer, variant.seats());
  }

  private int absolute(int seat) {
    return (observer + seat) % variant.seats();
  }

  /** ヘッダーに書く。全てのビットが 0 のフィールドは最小の値として読めるので、最小の値は書かない。 */
  private void header(HeaderBits field, int value) {
    write(schema.header, 0, 0, field, value);
  }

  private void flag(HeaderBits field, boolean value) {
    header(field, value ? 1 : 0);
  }

  /**
   * 記録のフィールドに値を書く。最小の値は書かない。
   *
   * @param at セクションの先頭の位置
   * @param record セクションの中の記録の番号
   */
  private <E extends Enum<E> & BitField> void write(BitLayout<E> layout, int at, int record, E field, int value) {
    if (value != layout.min(field)) {
      layout.put(words, at, record, field, value);
    }
  }

  /**
   * 記録の、繰り返しのあるフィールドに値を書く。最小の値は書かない。
   *
   * @param at セクションの先頭の位置
   * @param record セクションの中の記録の番号
   * @param index 繰り返しの番号
   */
  private <E extends Enum<E> & BitField> void write(
      BitLayout<E> layout, int at, int record, E field, int index, int value) {
    if (value != layout.min(field)) {
      layout.put(words, at, record, field, index, value);
    }
  }
}
