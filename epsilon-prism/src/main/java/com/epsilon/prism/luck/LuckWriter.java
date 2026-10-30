package com.epsilon.prism.luck;

import static com.epsilon.prism.luck.LuckSchema.HAND;
import static com.epsilon.prism.luck.LuckSchema.HEADER;
import static com.epsilon.prism.luck.LuckSchema.SEAT;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.PublicEventLog;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.luck.LuckSchema.HandBits;
import com.epsilon.prism.luck.LuckSchema.HeaderBits;
import com.epsilon.prism.luck.LuckSchema.SeatBits;
import com.epsilon.prism.luck.LuckSchema.Section;
import com.epsilon.prism.observe.EventRecords;
import com.epsilon.prism.observe.ObservationArena;
import java.util.Arrays;
import java.util.Objects;

/**
 * 全ての状態（{@link GameState}）から、偶然の出来事の直前の観測 O°（{@link LuckSchema}）を置き場に書く。進行ワーカーのスレッドに
 * 1 つ持つ。
 *
 * <p>収集の道は {@code EventLedger.Tracker.observe} と同じ再構成点（判断の境界）で書く。境界の state は出来事の後なので、結果に依る
 * 変化だけを戻す: その後のツモ（出来事そのもののツモか、表示の後のツモ）の牌を引いた席の手と赤から外して山の残りを 1 増やし、槓ドラの
 * 表示はその表示とそれより後の表示を隠す。
 *
 * <ul>
 *   <li>{@link #captureDraw}: 境界の直前のツモ（生きた山・嶺上）。
 *   <li>{@link #captureReveal}: 境界までに開いた槓ドラの表示。行為者はその表示の槓の作者（槓の出来事の番号の順で数える）。
 *   <li>{@link #captureBefore}: 出来事の直前の state そのもの（エンジンの {@code GameRecorder.beforeDraw}・
 *       {@code beforeDoraReveal} と同じ時点）。何も戻さない。
 * </ul>
 *
 * <p>境界から書いた O° は、出来事の直前の state から書いた O° とビットで同じになる。ただ一つの違いは、打牌の後に開く遅れた槓ドラの表示で、
 * エンジンがその打牌への応答を作るときに付ける見逃しの振聴（{@link SeatBits#FURITEN}）が境界では先に立っていること。この振聴は表示牌に
 * 依らない（ドラは役でない）ので、結果は見ない。
 *
 * <p>O° は作業域に書いてから置き場へ写すので、範囲の外の値で例外になっても置き場は変わらない。置き場を伸ばすときを除いて割り当てを
 * しない。
 */
public final class LuckWriter {

  private static final int SEATS = GameState.NUM_PLAYERS;

  private final EventRecords events = new EventRecords();
  private final long[] draft = new long[LuckSchema.MAX_LONGS];
  private final int[] start = new int[LuckSchema.SECTIONS.length];
  private final int[] melds = new int[SEATS];
  private final int[] discards = new int[SEATS];
  private int actor;

  /**
   * 境界の直前のツモの O° を書く。
   *
   * @param state 境界の状態（ツモの後）
   * @param draw そのツモ（境界の手番のイベント）
   * @param arena 書く先
   * @return 置き場の中の O° の先頭
   * @throws IllegalArgumentException 牌の分からないツモのとき、または値が O° の欄の範囲の外のとき
   * @throws IllegalStateException 引いた牌が引いた席の手にないとき
   */
  public int captureDraw(GameState state, TurnEvent.Draw draw, ObservationArena arena) {
    Objects.requireNonNull(draw, "draw");
    LuckEvent event = LuckEvent.of(draw);
    return write(
        state,
        event,
        draw.player(),
        state.doraState().indicatorCount(),
        draw,
        event == LuckEvent.RINSHAN,
        draw.tileType(),
        draw.isAkaTile(),
        arena);
  }

  /**
   * 境界までに開いた槓ドラの表示の O° を書く。
   *
   * @param state 境界の状態（表示の後）
   * @param indicator 開いた表示の番号（表示の順、1 始まり。初めの表示は 0 で出来事にしない）
   * @param laterDraw 表示の後、境界の前のツモ（なければ null）。嶺上のツモなら表示は嶺上待ちの間に開いた
   * @param arena 書く先
   * @return 置き場の中の O° の先頭
   * @throws IllegalArgumentException 表示の番号が範囲の外のとき、または値が O° の欄の範囲の外のとき
   * @throws IllegalStateException その表示の槓がないとき
   */
  public int captureReveal(
      GameState state, int indicator, TurnEvent.Draw laterDraw, ObservationArena arena) {
    int indicators = state.doraState().indicatorCount();
    if (indicator < 1 || indicator >= indicators) {
      throw new IllegalArgumentException(
          "kan dora indicator " + indicator + " is outside 1.." + (indicators - 1));
    }
    int physical = state.doraIndicatorPhysicalTileId(indicator);
    return write(
        state,
        LuckEvent.KAN_DORA,
        kanAuthor(state.events(), indicator),
        indicator,
        laterDraw,
        laterDraw != null && laterDraw.isRinshanDraw(),
        Tile.typeOf(physical),
        Tile.isAka(physical),
        arena);
  }

  /**
   * 出来事の直前の state から O° を書く（何も戻さない）。行為者はその時点の手番の席。
   *
   * @param state 出来事の直前の状態
   * @param event 出来事の種類
   * @param tile 結果の牌
   * @param aka 結果の牌が赤
   * @param arena 書く先
   * @return 置き場の中の O° の先頭
   * @throws IllegalArgumentException 値が O° の欄の範囲の外のとき
   */
  public int captureBefore(
      GameState state, LuckEvent event, int tile, boolean aka, ObservationArena arena) {
    boolean rinshanPending =
        switch (event) {
          case LIVE_DRAW -> false;
          case RINSHAN -> true;
          case KAN_DORA -> state.turnEvent() instanceof TurnEvent.KanAttempt;
        };
    return write(
        state,
        event,
        state.currentPlayer(),
        state.doraState().indicatorCount(),
        null,
        rinshanPending,
        tile,
        aka,
        arena);
  }

  /**
   * O° を作業域に書いて置き場へ写す。
   *
   * @param indicators 出来事の前に開いている表示の数
   * @param undo 出来事の前の時点の後に起きたツモ（戻す。なければ null）
   */
  private int write(
      GameState state,
      LuckEvent event,
      int actor,
      int indicators,
      TurnEvent.Draw undo,
      boolean rinshanPending,
      int tile,
      boolean aka,
      ObservationArena arena) {
    this.actor = Objects.checkIndex(actor, SEATS);
    int undoSeat = -1;
    if (undo != null) {
      undoSeat = relative(undo.player());
      requireDrawnTile(state.hand(undo.player()), undo);
    }
    LuckSchema.outcome(tile, aka);
    int meldTotal = 0;
    int discardTotal = 0;
    for (int seat = 0; seat < SEATS; seat++) {
      int player = absolute(seat);
      melds[seat] = state.hand(player).meldCount();
      discards[seat] = state.river(player).size();
      meldTotal += melds[seat];
      discardTotal += discards[seat];
    }
    int longs = LuckSchema.place(meldTotal, discardTotal, start);
    Arrays.fill(draft, 0, longs, 0L);
    writeHeader(state, event, indicators, undo, rinshanPending, tile, aka);
    for (int seat = 0; seat < SEATS; seat++) {
      writeSeat(state, seat, seat == undoSeat ? undo : null);
    }
    events.writeLuck(
        state.events(),
        actor,
        melds,
        discards,
        draft,
        start[Section.MELD.ordinal()],
        start[Section.DISCARD.ordinal()]);
    int base = arena.allocate(longs);
    System.arraycopy(draft, 0, arena.words(), base, longs);
    return base;
  }

  private void writeHeader(
      GameState state,
      LuckEvent event,
      int indicators,
      TurnEvent.Draw undo,
      boolean rinshanPending,
      int tile,
      boolean aka) {
    put(HEADER, 0, 0, HeaderBits.EVENT, event.ordinal());
    put(HEADER, 0, 0, HeaderBits.ACTOR, actor);
    put(HEADER, 0, 0, HeaderBits.DEALER, relative(state.dealer()));
    put(HEADER, 0, 0, HeaderBits.KYOKU, state.roundIndex());
    put(HEADER, 0, 0, HeaderBits.HONBA, state.honba());
    put(HEADER, 0, 0, HeaderBits.KYOTAKU, state.riichiSticks());
    put(HEADER, 0, 0, HeaderBits.WALL, state.remainingWallTiles() + (undo == null ? 0 : 1));
    flag(HEADER, 0, 0, HeaderBits.RINSHAN_PENDING, rinshanPending);
    put(HEADER, 0, 0, HeaderBits.INDICATORS, indicators);
    DoraState dora = state.doraState();
    for (int index = 0; index < indicators; index++) {
      put(HEADER, 0, 0, HeaderBits.INDICATOR, index, dora.indicatorTileType(index));
      if (Tile.isAka(state.doraIndicatorPhysicalTileId(index))) {
        put(HEADER, 0, 0, HeaderBits.INDICATOR_AKA, index, 1);
      }
    }
    put(HEADER, 0, 0, HeaderBits.OUTCOME_TILE, tile);
    flag(HEADER, 0, 0, HeaderBits.OUTCOME_AKA, aka);
  }

  /** 席と手を書く。undo があればその牌を手と赤から外す。 */
  private void writeSeat(GameState state, int seat, TurnEvent.Draw undo) {
    int player = absolute(seat);
    Hand hand = state.hand(player);
    int at = start[Section.SEAT.ordinal()];
    put(SEAT, at, seat, SeatBits.SCORE, LuckSchema.encodePoints(state.score(player)));
    put(SEAT, at, seat, SeatBits.RIICHI, riichi(state, player).ordinal());
    flag(SEAT, at, seat, SeatBits.IPPATSU, state.isIppatsu(player));
    flag(SEAT, at, seat, SeatBits.FURITEN, state.isTemporaryFuriten(player));
    int akaMask = hand.concealedAkaMask();
    if (undo != null && undo.isAkaTile()) {
      akaMask = AkaTileMask.excludeTile(akaMask, undo.tileType());
    }
    put(SEAT, at, seat, SeatBits.HAND_AKA, akaMask);
    put(SEAT, at, seat, SeatBits.MELDS, melds[seat]);
    put(SEAT, at, seat, SeatBits.DISCARDS, discards[seat]);
    int handAt = start[Section.HAND.ordinal()];
    for (long tiles = hand.concealedTileTypeMask(); tiles != 0L; tiles &= tiles - 1) {
      int tile = Long.numberOfTrailingZeros(tiles);
      int count = hand.count(tile) - (undo != null && undo.tileType() == tile ? 1 : 0);
      put(HAND, handAt, seat, HandBits.COUNT, tile, count);
    }
  }

  /**
   * 表示 {@code indicator}（1 始まり）を生んだ槓の作者（絶対席）を返す。槓ドラは槓の順に開くので、表示 k は公開の出来事の記録で
   * k 番目の槓（大明槓・暗槓・加槓の出来事）のもの。
   */
  private static int kanAuthor(PublicEventLog log, int indicator) {
    int kans = 0;
    for (int event = 0, events = log.size(); event < events; event++) {
      if (!log.isDiscard(event) && log.meld(event).isKan() && ++kans == indicator) {
        return log.seat(event);
      }
    }
    throw new IllegalStateException("kan dora indicator " + indicator + " has no kan");
  }

  private static void requireDrawnTile(Hand hand, TurnEvent.Draw draw) {
    int tile = draw.tileType();
    if (tile < 0) {
      throw new IllegalArgumentException("draw has no tile: player " + draw.player());
    }
    if (hand.count(tile) < 1
        || draw.isAkaTile() && !AkaTileMask.containsTile(hand.concealedAkaMask(), tile)) {
      throw new IllegalStateException(
          "drawn tile " + Tile.name(tile) + " is not in the hand of player " + draw.player());
    }
  }

  private static RiichiState riichi(GameState state, int player) {
    if (state.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }

  private int relative(int player) {
    return Math.floorMod(player - actor, SEATS);
  }

  private int absolute(int seat) {
    return (actor + seat) % SEATS;
  }

  /** 作業域に書く（範囲の外の値は {@link BitLayout} が例外にする）。 */
  private <E extends Enum<E> & BitField> void put(
      BitLayout<E> layout, int at, int record, E field, int value) {
    layout.put(draft, at, record, field, value);
  }

  /** 作業域の繰り返しのある欄に書く。 */
  private <E extends Enum<E> & BitField> void put(
      BitLayout<E> layout, int at, int record, E field, int index, int value) {
    layout.put(draft, at, record, field, index, value);
  }

  /** 作業域の 2 値の欄に書く（作業域は 0 で始まるので、偽は書かない）。 */
  private <E extends Enum<E> & BitField> void flag(
      BitLayout<E> layout, int at, int record, E field, boolean value) {
    if (value) {
      layout.put(draft, at, record, field, 1);
    }
  }
}
