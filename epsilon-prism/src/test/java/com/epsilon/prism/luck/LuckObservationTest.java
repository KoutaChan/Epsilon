package com.epsilon.prism.luck;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.River;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.RoundResult;
import com.epsilon.prism.luck.LuckSchema.SeatBits;
import com.epsilon.prism.observe.ObservationArena;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 観測 O° の捕捉と読み。
 *
 * <p>槓の多い打ち方の固定の種の全ての局で、判断の境界で復元した出来事ごとに、境界の state から書いた O°（収集の道）が、エンジンの記録処理が
 * 偶然の点の直前（beforeDraw・beforeDoraReveal）に渡した state から書いた O° とビットで同じになる。打牌の後に開く遅れた表示だけは、
 * その打牌への応答を作るときに付く見逃しの振聴が境界で先に立っていてよい（直前に立っていた振聴は境界でも立っている）。読んだ全ての値
 * （4 席の手と赤・副露・河・点と旗・表示・結果）が直前の state と一致し、プール（4 − 公開 − 全員の手）がエンジンの数え方と一致する。
 * O° の手とプールは結果を見ない: 引いた牌は引いた席の手になく、結果はプールに残っている（直前の state と同じなので）。
 */
public class LuckObservationTest {
  private static final Logger log = LoggerFactory.getLogger(LuckObservationTest.class);

  /** 最初の半荘の種。 */
  private static final long SEED = 9_100;

  private static final int GAMES = 150;

  @Test(timeOut = 900_000)
  public void boundaryCaptureMatchesTheEngineBeforeEveryEvent() {
    Instant instant = new Instant();
    Check check = new Check(instant);
    LuckGames.play(SEED, GAMES, instant, check);
    log.info("LUCK-O fingerprint {} sample {}", LuckSchema.fingerprint(), LuckSampleFormat.fingerprint());
    log.info(check.summary());
    assertTrue(check.events[LuckEvent.LIVE_DRAW.ordinal()] > 20_000, check.summary());
    assertTrue(check.events[LuckEvent.RINSHAN.ordinal()] > 200, check.summary());
    assertTrue(check.immediateReveals > 50, check.summary());
    assertTrue(check.deferredReveals > 50, check.summary());
    assertTrue(check.akaOutcomes > 100, check.summary());
    assertTrue(check.calledDiscards > 1_000, check.summary());
  }

  /** エンジンの偶然の点の直前の state（写し）と、その結果。 */
  private record Event(LuckEvent event, int actor, int tile, boolean aka, GameState before) {}

  /** 偶然の点の直前の state を写し、結果と一緒に記録する記録処理。 */
  private static final class Instant implements GameRecorder {
    final List<Event> events = new ArrayList<>();
    private GameState before;
    private int drawer = -1;
    private boolean rinshan;

    @Override
    public void beforeDraw(int player, boolean rinshan, GameState state) {
      before = state.snapshotCopy();
      drawer = player;
      this.rinshan = rinshan;
    }

    @Override
    public void recordDraw(int player, int tileType, boolean isAkaTile) {
      take(LuckEvent.LIVE_DRAW, player, tileType, isAkaTile);
    }

    @Override
    public void recordRinshanDraw(int player, int tileType, boolean isAkaTile) {
      take(LuckEvent.RINSHAN, player, tileType, isAkaTile);
    }

    @Override
    public void beforeDoraReveal(GameState state) {
      before = state.snapshotCopy();
      drawer = -1;
    }

    @Override
    public void recordNewDora(int physicalTileId) {
      events.add(
          new Event(
              LuckEvent.KAN_DORA,
              before.currentPlayer(),
              Tile.typeOf(physicalTileId),
              Tile.isAka(physicalTileId),
              before));
      before = null;
    }

    private void take(LuckEvent event, int player, int tile, boolean aka) {
      assertEquals(player, drawer);
      assertEquals(event == LuckEvent.RINSHAN, rinshan);
      assertEquals(before.currentPlayer(), player);
      events.add(new Event(event, player, tile, aka, before));
      before = null;
    }

    @Override
    public void startRound(int roundIndex, int honba, int kyotakuCount, int[] startingScores) {}

    @Override
    public void recordHaipai(int player, int[] concealedTileCounts, int concealedAkaMask) {}

    @Override
    public void recordDoraIndicators(int[] physicalTileIds) {}

    @Override
    public void recordDahai(
        int player, int tileType, boolean isAkaTile, boolean tsumogiri, boolean riichi) {}

    @Override
    public void recordRiichiAccepted(int player) {}

    @Override
    public void recordChi(
        int player,
        int[] chiTileTypes,
        int calledTileType,
        boolean calledTileIsAka,
        boolean consumedHandTileIsAka) {}

    @Override
    public void recordPon(
        int player,
        int tileType,
        int sourcePlayerOffset,
        boolean calledTileIsAka,
        boolean consumedHandTileIsAka) {}

    @Override
    public void recordDaiminkan(
        int player,
        int tileType,
        int sourcePlayerOffset,
        boolean calledTileIsAka,
        boolean consumedHandTileIsAka) {}

    @Override
    public void recordAnkan(int player, int tileType, boolean containsAkaTile) {}

    @Override
    public void recordKakan(int player, int tileType, boolean addedTileIsAka) {}

    @Override
    public void recordRoundResult(RoundResult result) {}

    @Override
    public void endRound(RoundEnd roundEnd) {}
  }

  /** 境界で復元した出来事を、直前の state と照らす。 */
  private static final class Check implements LuckGames.Visitor, LuckSink {
    private final Instant instant;
    private final LuckWriter boundaryWriter = new LuckWriter();
    private final LuckWriter instantWriter = new LuckWriter();
    private final ObservationArena boundaryArena = new ObservationArena(256);
    private final ObservationArena instantArena = new ObservationArena(256);
    private final LuckObservation read = new LuckObservation();
    private final LuckStratum stratum = new LuckStratum();
    private final long[] events = new long[LuckEvent.values().length];
    private int next;
    private int revealsHere;
    private long immediateReveals;
    private long deferredReveals;
    private long doubleReveals;
    private long furitenAhead;
    private long akaOutcomes;
    private long calledDiscards;
    private int maxLongs;

    Check(Instant instant) {
      this.instant = instant;
    }

    @Override
    public LuckSink sink() {
      return this;
    }

    @Override
    public void reveal(GameState state, int indicator, TurnEvent.Draw laterDraw) {
      revealsHere++;
      compare(boundaryWriter.captureReveal(state, indicator, laterDraw, boundaryArena));
    }

    @Override
    public void draw(GameState state, TurnEvent.Draw draw) {
      compare(boundaryWriter.captureDraw(state, draw, boundaryArena));
    }

    @Override
    public void restored() {
      assertEquals(next, instant.events.size(), "events between two boundaries");
      if (revealsHere >= 2) {
        doubleReveals++;
      }
      revealsHere = 0;
      next = 0;
      instant.events.clear();
      boundaryArena.clear();
      instantArena.clear();
    }

    @Override
    public void roundEnded(int game, int round) {
      next = 0;
      instant.events.clear();
    }

    /** 境界の O° を、同じ出来事の直前の state から書いた O° とエンジンの値に照らす。 */
    private void compare(int boundaryBase) {
      assertTrue(next < instant.events.size(), "restored an event the engine did not have");
      Event engine = instant.events.get(next++);
      int instantBase =
          instantWriter.captureBefore(
              engine.before(), engine.event(), engine.tile(), engine.aka(), instantArena);
      long[] restored = copy(boundaryArena.words(), boundaryBase);
      long[] direct = copy(instantArena.words(), instantBase);
      read.bind(restored, 0);
      boolean deferred = read.event() == LuckEvent.KAN_DORA && !read.rinshanPending();
      if (deferred) {
        long[] restoredMasked = withoutFuriten(restored);
        long[] directMasked = withoutFuriten(direct);
        assertTrue(Arrays.equals(restoredMasked, directMasked), where(engine) + " O° bits");
        if (!Arrays.equals(restored, direct)) {
          furitenAhead++;
        }
      } else {
        assertTrue(Arrays.equals(restored, direct), where(engine) + " O° bits");
      }
      assertMatches(read.bind(restored, 0), engine, deferred);
      int key = stratum.of(read);
      assertTrue(key >= 0 && key < LuckStratum.COUNT, "stratum " + key);
      events[engine.event().ordinal()]++;
      if (engine.event() == LuckEvent.KAN_DORA) {
        if (deferred) {
          deferredReveals++;
        } else {
          immediateReveals++;
        }
      }
      if (engine.aka()) {
        akaOutcomes++;
      }
      maxLongs = Math.max(maxLongs, restored.length);
    }

    /** 読んだ値を直前の state（エンジン）と照らす。 */
    private void assertMatches(LuckObservation o, Event engine, boolean deferred) {
      GameState before = engine.before();
      String where = where(engine);
      assertEquals(o.event(), engine.event(), where);
      assertEquals(o.actor(), engine.actor(), where);
      assertEquals(o.outcomeTile(), engine.tile(), where);
      assertEquals(o.outcomeAka(), engine.aka(), where);
      assertEquals(o.absoluteSeat(o.dealer()), before.dealer(), where);
      assertEquals(o.kyoku(), before.roundIndex(), where);
      assertEquals(o.roundWindTile(), before.roundWindTileType(), where);
      assertEquals(o.honba(), before.honba(), where);
      assertEquals(o.kyotaku(), before.riichiSticks(), where);
      assertEquals(o.remainingWall(), before.remainingWallTiles(), where);
      assertEquals(o.kans(), before.publicState().totalKanCount(), where);
      assertEquals(
          o.rinshanPending(),
          engine.event() == LuckEvent.RINSHAN
              || engine.event() == LuckEvent.KAN_DORA
                  && before.turnEvent() instanceof TurnEvent.KanAttempt,
          where);
      assertEquals(o.indicators(), before.doraState().indicatorCount(), where);
      for (int index = 0; index < o.indicators(); index++) {
        assertEquals(o.indicatorTile(index), before.doraState().indicatorTileType(index), where);
        assertEquals(
            o.indicatorAka(index), Tile.isAka(before.doraIndicatorPhysicalTileId(index)), where);
      }
      int seenAka = before.publicState().visibleDiscardOrMeldAkaMask();
      seenAka |= before.doraState().visibleIndicatorAkaMask();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        int player = o.absoluteSeat(seat);
        String at = where + " seat " + seat;
        Hand hand = before.hand(player);
        River river = before.river(player);
        assertEquals(o.score(seat), before.score(player), at);
        assertEquals(o.riichi(seat), riichi(before, player), at);
        assertEquals(o.ippatsu(seat), before.isIppatsu(player), at);
        if (deferred) {
          assertTrue(o.furiten(seat) || !before.isTemporaryFuriten(player), at + " furiten");
        } else {
          assertEquals(o.furiten(seat), before.isTemporaryFuriten(player), at + " furiten");
        }
        assertEquals(o.handAkaMask(seat), hand.concealedAkaMask(), at);
        seenAka |= hand.concealedAkaMask();
        assertEquals(o.menzen(seat), hand.isMenzen(), at);
        assertEquals(o.firstDraw(seat), before.isFirstDraw(player), at);
        for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
          assertEquals(o.handCount(seat, tile), hand.count(tile), at + " tile " + tile);
        }
        assertEquals(o.meldCount(seat), hand.meldCount(), at);
        for (int index = 0; index < hand.meldCount(); index++) {
          assertEquals(o.meld(seat, index), hand.meld(index), at);
          assertEquals(o.meldEvent(seat, index), hand.meldCreationEvent(index), at);
          assertEquals(o.meldKakanEvent(seat, index), hand.meldKakanEvent(index), at);
        }
        assertEquals(o.discardCount(seat), river.size(), at);
        for (int index = 0; index < river.size(); index++) {
          assertEquals(o.discardTile(seat, index), river.tileType(index), at);
          assertEquals(o.discardAka(seat, index), river.aka(index), at);
          assertEquals(o.discardTsumogiri(seat, index), river.tsumogiri(index), at);
          assertEquals(o.discardRiichi(seat, index), river.riichiDeclaration(index), at);
          assertEquals(o.discardCalled(seat, index), river.called(index), at);
          assertEquals(o.discardEvent(seat, index), river.event(index), at);
          if (river.called(index)) {
            calledDiscards++;
          }
        }
      }
      int total = 0;
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        int unseen = Tile.TILES_PER_TYPE - before.publicState().visibleTileCount(tile);
        for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
          unseen -= before.hand(player).count(tile);
        }
        assertEquals(o.pool(tile), unseen, where + " pool " + Tile.name(tile));
        total += unseen;
      }
      assertEquals(o.poolTotal(), total, where);
      assertEquals(o.poolAkaMask(), LuckSchema.AKA_MASK & ~seenAka, where + " aka pool");
      int outcomes = 0;
      for (int outcome = 0; outcome < LuckSchema.OUTCOMES; outcome++) {
        outcomes += o.poolOf(outcome);
      }
      assertEquals(outcomes, total, where);
      assertTrue(o.poolOf(o.outcome()) >= 1, where + " outcome is not in the pool");
    }

    private static long[] copy(long[] words, int base) {
      return Arrays.copyOfRange(words, base, base + LuckSchema.longs(words, base));
    }

    /** 4 席の見逃しの振聴を消した写し。 */
    private static long[] withoutFuriten(long[] words) {
      long[] copy = words.clone();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        LuckSchema.SEAT.put(copy, LuckSchema.SEAT_START, seat, SeatBits.FURITEN, 0);
      }
      return copy;
    }

    private static RiichiState riichi(GameState state, int player) {
      if (state.isDoubleRiichi(player)) {
        return RiichiState.DOUBLE_RIICHI;
      }
      return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
    }

    private static String where(Event engine) {
      return engine.event()
          + " kyoku "
          + engine.before().roundIndex()
          + " wall "
          + engine.before().remainingWallTiles()
          + " actor "
          + engine.actor()
          + " outcome "
          + Tile.name(engine.tile());
    }

    String summary() {
      return "LUCK-O events "
          + Arrays.toString(events)
          + " immediateReveals "
          + immediateReveals
          + " deferredReveals "
          + deferredReveals
          + " doubleRevealBoundaries "
          + doubleReveals
          + " furitenAhead "
          + furitenAhead
          + " akaOutcomes "
          + akaOutcomes
          + " calledDiscards "
          + calledDiscards
          + " maxLongs "
          + maxLongs
          + "/"
          + LuckSchema.MAX_LONGS;
    }
  }
}
