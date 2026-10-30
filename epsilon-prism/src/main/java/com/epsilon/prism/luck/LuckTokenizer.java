package com.epsilon.prism.luck;

import com.epsilon.calculate.scoring.HandScoreBatchBuffer;
import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.core.Tile;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.BoundedField;
import com.epsilon.prism.input.InputSchema.Wind;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism.luck.LuckInputSchema.RoundField;
import com.epsilon.prism.luck.LuckInputSchema.RoundScalar;
import com.epsilon.prism.luck.LuckInputSchema.SeatField;
import com.epsilon.prism.luck.LuckInputSchema.SeatScalar;
import com.epsilon.prism.luck.LuckInputSchema.TileField;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import com.epsilon.prism.luck.LuckInputSchema.WinGrade;
import java.util.Arrays;

/**
 * 観測 O° から運の網 Ω の 1 行の入力（{@link LuckInputSchema} の全ての欄・数値・関係の符号）と、正準の結果の順のプールを書く。O° の
 * 純関数で、結果の欄（{@link LuckObservation#outcome}）は読まない。結果の牌だけが違う O° からは、ビットで同じ入力とプールを書く。符号化のスレッドに
 * 1 つ持ち、書くときに割り当てをしない。
 *
 * <p>トークンは局 1・席 4（行為者から見た相対席の順）・牌 34（正準の色の順、{@link LuckSuits}）の順に書く。席ごとの形の解析（向聴・受け入れ・
 * 待ち）は O° の手で 1 回ずつ行い、聴牌の席だけ待ちを採点する（行為者はツモ、ほかの 3 席はロン。条件は自風・場風・立直だけで、裏ドラは数え
 * ない）。4 枚とも自分が持つ牌（手と副露）は待ちにしない（その牌では和了できず、採点も五枚目になる）。
 */
final class LuckTokenizer {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int ROUND_WIDTH = LuckInputSchema.ROUND.width();
  private static final int SEAT_WIDTH = LuckInputSchema.SEAT.width();
  private static final int TILE_WIDTH = LuckInputSchema.TILE.width();
  private static final int ROUND_SCALARS = Token.ROUND.scalars();
  private static final int SEAT_SCALARS = Token.SEAT.scalars();
  private static final int KYOKU_IDS = HanchanProgression.WEST_END_KYOKU + 1;
  private static final Wind[] WINDS = Wind.values();

  private final LuckSuits suits = new LuckSuits();
  private final SeatHand[] hands = new SeatHand[SEATS];
  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
  private final HandScoreEvaluator evaluator = new HandScoreEvaluator();
  private final HandScoreBatchBuffer batch = new HandScoreBatchBuffer(2);
  private final int[][] held = new int[SEATS][TILES];
  private final int[] shanten = new int[SEATS];
  private final long[] waits = new long[SEATS];
  private final long[] rivers = new long[SEATS];
  private final boolean[] discardFuriten = new boolean[SEATS];
  private final int[] dora = new int[TILES];
  private final WinGrade[] tsumo = new WinGrade[TILES];
  private final WinGrade[] tsumoAka = new WinGrade[TILES];
  private final WinGrade[][] ron = new WinGrade[SEATS][TILES];
  private final int[] absoluteScores = new int[SEATS];
  private final int[] ranks = new int[SEATS];
  private long actorUkeire;
  private long doraPacked;

  LuckTokenizer() {
    for (int seat = 0; seat < SEATS; seat++) {
      hands[seat] = new SeatHand();
    }
  }

  /**
   * O° から 1 行の入力とプールを書く。
   *
   * @param o 観測 O°（付け替えた読み手）
   * @param rows 書く先
   * @param row 行
   * @throws IllegalStateException O° が規則と合わないか、値が欄の語彙の外のとき
   */
  void write(LuckObservation o, LuckRows rows, int row) {
    rows.clear(row);
    suits.load(o);
    loadSeats(o);
    loadDora(o);
    scoreWins(o);
    writeRound(o, rows, row);
    writeSeats(o, rows, row);
    writeTiles(o, rows, row);
    LuckRelation.write(rows.codes, row * LuckRelation.ROW_BYTES, held, waits, rivers, suits);
    writePool(o, rows, row);
    validate(rows, row);
  }

  /**
   * 直前に {@link #write} した O° の色の写しで、物理の結果の番号を正準の結果の番号にする。学習の選び（どの結果の V を使うか）のための口で、入力に
   * は使わない。
   *
   * @param outcome 物理の結果の番号（{@link LuckSchema#outcome}）
   * @return 正準の結果の番号（網の出力の結果の軸の位置）
   */
  int canonicalOutcome(int outcome) {
    return LuckSchema.outcome(
        suits.tile(LuckSchema.outcomeTile(outcome)), LuckSchema.outcomeAka(outcome));
  }

  private void loadSeats(LuckObservation o) {
    for (int seat = 0; seat < SEATS; seat++) {
      SeatHand hand = hands[seat];
      hand.load(o, seat);
      for (int tile = 0; tile < TILES; tile++) {
        held[seat][tile] = hand.count(tile);
      }
      long river = 0L;
      for (int index = 0, count = o.discardCount(seat); index < count; index++) {
        river |= 1L << o.discardTile(seat, index);
      }
      rivers[seat] = river;
      shapes.analyzeInto(hand.shape(), analysis);
      shanten[seat] = Math.max(0, analysis.minimumShanten());
      if (seat == 0) {
        actorUkeire = analysis.ukeireTileTypeMask();
      }
      long live = 0L;
      for (long rest = analysis.agariTileTypeMask(); rest != 0L; rest &= rest - 1) {
        int tile = Long.numberOfTrailingZeros(rest);
        if (hand.owned(tile) < Tile.TILES_PER_TYPE) {
          live |= 1L << tile;
        }
      }
      waits[seat] = live;
      discardFuriten[seat] = (live & river) != 0L;
    }
    for (int seat = 0; seat < SEATS; seat++) {
      absoluteScores[o.absoluteSeat(seat)] = o.score(seat);
    }
    ScoreRanking.byScoreThenSeat(absoluteScores, ranks);
  }

  private void loadDora(LuckObservation o) {
    Arrays.fill(dora, 0);
    doraPacked = 0L;
    for (int index = 0, count = o.indicators(); index < count; index++) {
      int tile = Tile.doraFrom(o.indicatorTile(index));
      dora[tile]++;
      doraPacked = DoraState.packTile(doraPacked, index, tile);
    }
  }

  /** 行為者のツモの格（赤でない牌と赤の 5）と、ほかの 3 席のロンの格を採点する。 */
  private void scoreWins(LuckObservation o) {
    Arrays.fill(tsumo, null);
    Arrays.fill(tsumoAka, null);
    int roundWind = o.roundWindTile();
    int indicators = o.indicators();
    if (waits[0] != 0L) {
      SeatHand actor = hands[0];
      WinConditions conditions =
          WinConditions.publicDecision(seatWind(o, 0), roundWind, o.riichi(0));
      int owned = actor.ownedAkaMask();
      int poolAka = o.poolAkaMask();
      for (long rest = waits[0]; rest != 0L; rest &= rest - 1) {
        int tile = Long.numberOfTrailingZeros(rest);
        if (o.pool(tile) == 0) {
          continue;
        }
        boolean aka = AkaTileMask.containsTile(poolAka, tile);
        batch.configure(0, WinMethod.TSUMO, conditions, owned);
        if (aka) {
          batch.configure(1, WinMethod.TSUMO, conditions, AkaTileMask.includeTile(owned, tile));
        }
        batch.size(aka ? 2 : 1);
        evaluator.scoreAfterAddingBatch(actor.shape(), actor, tile, doraPacked, indicators, batch);
        tsumo[tile] = grade(batch.result(0));
        if (aka) {
          tsumoAka[tile] = grade(batch.result(1));
        }
      }
    }
    for (int seat = 1; seat < SEATS; seat++) {
      WinGrade[] grades = ron[seat];
      Arrays.fill(grades, null);
      if (waits[seat] == 0L || o.furiten(seat) || discardFuriten[seat]) {
        continue;
      }
      SeatHand hand = hands[seat];
      WinConditions conditions =
          WinConditions.publicDecision(seatWind(o, seat), roundWind, o.riichi(seat));
      batch.configure(0, WinMethod.RON, conditions, hand.ownedAkaMask()).size(1);
      for (long rest = waits[seat]; rest != 0L; rest &= rest - 1) {
        int tile = Long.numberOfTrailingZeros(rest);
        evaluator.scoreAfterAddingBatch(hand.shape(), hand, tile, doraPacked, indicators, batch);
        grades[tile] = grade(batch.result(0));
      }
    }
  }

  private static WinGrade grade(HandScoreBuffer score) {
    return score.available() ? WinGrade.of(score.basePoints()) : null;
  }

  /** 相対席 {@code seat} の自風の牌。 */
  private static int seatWind(LuckObservation o, int seat) {
    return Tile.TON + Math.floorMod(seat - o.dealer(), SEATS);
  }

  private void writeRound(LuckObservation o, LuckRows rows, int row) {
    TokenLayout<RoundField> layout = LuckInputSchema.ROUND;
    byte[] target = rows.round;
    int at = row * ROUND_WIDTH;
    set(target, at, layout, RoundField.EVENT, Ids.of(o.event()));
    set(target, at, layout, RoundField.ROUND_WIND, Ids.of(Wind.ofTile(o.roundWindTile())));
    set(target, at, layout, RoundField.KYOKU, Ids.index(o.kyoku(), KYOKU_IDS));
    set(
        target,
        at,
        layout,
        RoundField.HONBA,
        Ids.saturated(o.honba(), InputSchema.STICK_SATURATION));
    set(
        target,
        at,
        layout,
        RoundField.KYOTAKU,
        Ids.saturated(o.kyotaku(), InputSchema.STICK_SATURATION));
    set(target, at, layout, RoundField.KANS, o.kans());
    set(target, at, layout, RoundField.INDICATORS, o.indicators());
    set(target, at, layout, RoundField.RINSHAN_PENDING, flag(o.rinshanPending()));
    int scalars = row * ROUND_SCALARS;
    rows.roundScalar[scalars + RoundScalar.WALL.ordinal()] =
        o.remainingWall() / RoundScalar.WALL.scale();
    rows.roundScalar[scalars + RoundScalar.POOL.ordinal()] =
        o.poolTotal() / RoundScalar.POOL.scale();
  }

  private void writeSeats(LuckObservation o, LuckRows rows, int row) {
    TokenLayout<SeatField> layout = LuckInputSchema.SEAT;
    byte[] target = rows.seat;
    int actorScore = o.score(0);
    for (int seat = 0; seat < SEATS; seat++) {
      SeatHand hand = hands[seat];
      int at = (row * SEATS + seat) * SEAT_WIDTH;
      set(target, at, layout, SeatField.RELATIVE_SEAT, Ids.seat(seat));
      Wind wind = WINDS[Math.floorMod(seat - o.dealer(), SEATS)];
      set(target, at, layout, SeatField.WIND, Ids.of(wind));
      set(target, at, layout, SeatField.RANK, Ids.index(ranks[o.absoluteSeat(seat)], SEATS));
      set(target, at, layout, SeatField.RIICHI, Ids.of(o.riichi(seat)));
      set(target, at, layout, SeatField.IPPATSU, flag(o.ippatsu(seat)));
      set(target, at, layout, SeatField.FURITEN, flag(o.furiten(seat)));
      set(target, at, layout, SeatField.DISCARD_FURITEN, flag(discardFuriten[seat]));
      set(target, at, layout, SeatField.MENZEN, flag(hand.isMenzen()));
      set(target, at, layout, SeatField.FIRST_DRAW, flag(o.firstDraw(seat)));
      set(target, at, layout, SeatField.MELDS, o.meldCount(seat));
      set(target, at, layout, SeatField.KANS, hand.kanCount());
      set(target, at, layout, SeatField.DISCARDS, o.discardCount(seat));
      set(target, at, layout, SeatField.SHANTEN, Ids.index(shanten[seat], Limits.MAX_SHANTEN + 1));
      set(target, at, layout, SeatField.WAITS, Long.bitCount(waits[seat]));
      set(target, at, layout, SeatField.DORA, handDora(hand));
      int scalars = (row * SEATS + seat) * SEAT_SCALARS;
      int score = o.score(seat);
      rows.seatScalar[scalars + SeatScalar.SCORE.ordinal()] = score / SeatScalar.SCORE.scale();
      rows.seatScalar[scalars + SeatScalar.GAP_TO_ACTOR.ordinal()] =
          (score - actorScore) / SeatScalar.GAP_TO_ACTOR.scale();
      int live = 0;
      for (long rest = waits[seat]; rest != 0L; rest &= rest - 1) {
        live += o.pool(Long.numberOfTrailingZeros(rest));
      }
      rows.seatScalar[scalars + SeatScalar.LIVE_WAITS.ordinal()] =
          live / SeatScalar.LIVE_WAITS.scale();
    }
  }

  /** 手と副露のドラ（表ドラの倍率 × 枚数と赤）。 */
  private int handDora(SeatHand hand) {
    int total = Integer.bitCount(hand.ownedAkaMask());
    for (int tile = 0; tile < TILES; tile++) {
      if (dora[tile] != 0) {
        total += dora[tile] * hand.owned(tile);
      }
    }
    return total;
  }

  private void writeTiles(LuckObservation o, LuckRows rows, int row) {
    TokenLayout<TileField> layout = LuckInputSchema.TILE;
    byte[] target = rows.tile;
    int poolAka = o.poolAkaMask();
    for (int canonical = 0; canonical < TILES; canonical++) {
      int tile = suits.physical(canonical);
      long bit = 1L << tile;
      int at = (row * TILES + canonical) * TILE_WIDTH;
      int pool = o.pool(tile);
      int inHands = 0;
      int doraTile = Tile.doraFrom(tile);
      set(target, at, layout, TileField.TILE, Ids.tile(canonical));
      for (int seat = 0; seat < SEATS; seat++) {
        SeatHand hand = hands[seat];
        inHands += held[seat][tile];
        set(target, at, layout, TileField.HELD, seat, held[seat][tile]);
        set(target, at, layout, TileField.HELD_AKA, seat, flag(hand.hasAkaTile(tile)));
        set(target, at, layout, TileField.DORA_HELD, seat, hand.owned(doraTile));
        set(target, at, layout, TileField.RIVER, seat, flag((rivers[seat] & bit) != 0L));
        set(target, at, layout, TileField.WAIT, seat, flag((waits[seat] & bit) != 0L));
      }
      set(target, at, layout, TileField.VISIBLE, Tile.TILES_PER_TYPE - pool - inHands);
      set(target, at, layout, TileField.POOL, pool);
      set(target, at, layout, TileField.POOL_AKA, flag(AkaTileMask.containsTile(poolAka, tile)));
      set(target, at, layout, TileField.DORA, dora[tile]);
      set(target, at, layout, TileField.UKEIRE, flag((actorUkeire & bit) != 0L));
      set(target, at, layout, TileField.TSUMO, Ids.of(tsumo[tile]));
      set(target, at, layout, TileField.TSUMO_AKA, Ids.of(tsumoAka[tile]));
      for (int seat = 1; seat < SEATS; seat++) {
        set(target, at, layout, TileField.RON, seat - 1, Ids.of(ron[seat][tile]));
      }
    }
  }

  /** 正準の結果の順のプールの枚数（赤でない 34 種、赤 5 の 3 色）。 */
  private void writePool(LuckObservation o, LuckRows rows, int row) {
    int at = row * LuckRows.OUTCOMES;
    for (int canonical = 0; canonical < TILES; canonical++) {
      rows.pool[at + canonical] = o.poolOf(LuckSchema.outcome(suits.physical(canonical), false));
    }
    for (int suit = 0; suit < LuckSuits.SUITS; suit++) {
      int five = suits.physicalSuit(suit) * Tile.TILE_TYPES_PER_SUIT + Tile.M5;
      rows.pool[at + TILES + suit] = o.poolOf(LuckSchema.outcome(five, true));
    }
  }

  /** 行の全てのトークンの値を語彙で確かめる。 */
  private static void validate(LuckRows rows, int row) {
    LuckInputSchema.ROUND.requireValid(rows.round, row * ROUND_WIDTH);
    for (int seat = 0; seat < SEATS; seat++) {
      LuckInputSchema.SEAT.requireValid(rows.seat, (row * SEATS + seat) * SEAT_WIDTH);
    }
    for (int tile = 0; tile < TILES; tile++) {
      LuckInputSchema.TILE.requireValid(rows.tile, (row * TILES + tile) * TILE_WIDTH);
    }
  }

  private static int flag(boolean value) {
    return value ? 1 : 0;
  }

  /** 繰り返しのない欄に書く。 */
  private static <E extends Enum<E> & BoundedField> void set(
      byte[] target, int at, TokenLayout<E> layout, E field, int value) {
    target[at + layout.column(field)] = unsigned(field, value);
  }

  /** 繰り返しのある欄の repeat 番目に書く。 */
  private static <E extends Enum<E> & BoundedField> void set(
      byte[] target, int at, TokenLayout<E> layout, E field, int repeat, int value) {
    target[at + layout.column(field, repeat)] = unsigned(field, value);
  }

  /** u8 に収まらない値は例外にする（語彙の確かめは行の終わりにまとめて行う）。 */
  private static byte unsigned(Enum<?> field, int value) {
    if ((value & ~0xFF) != 0) {
      throw new IllegalStateException(field.name() + " does not fit u8: " + value);
    }
    return (byte) value;
  }
}
