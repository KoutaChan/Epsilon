package com.epsilon.prism.collect;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 出来事の順序（{@link EventLedger.Tracker#observe} の復元）: 主経路の engine の記録処理が偶然の点の直前（beforeDraw・beforeDoraReveal）に読んだ
 * プールと 4 席の手牌と、境界で復元した出来事の前のプールと手牌が、出来事ごとに一致する。槓を必ず選ぶ打ち方で、遅れた表示・連続の槓・嶺上の後の
 * 打牌を含む。
 */
public class EventLedgerOrderTest {
  private static int[][] hands(GameState state) {
    int[][] hands = new int[GameState.NUM_PLAYERS][Tile.NUM_TILE_TYPES];
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) state.hand(seat).copyConcealedTileCountsTo(hands[seat]);
    return hands;
  }

  /** 偶然の点の直前の状態と、その実現（引いた牌・表示牌の種類）を記録する。 */
  private static final class Recorder extends LedgerGames.SilentRecorder {
    final List<EventLedger.Event> events = new ArrayList<>();
    int seat = -1;
    int[] pool;
    int[][] hands;

    @Override
    public void beforeDraw(int player, boolean rinshan, GameState state) {
      seat = player;
      pool = new int[Tile.NUM_TILE_TYPES];
      EventLedger.pool(state, pool);
      hands = hands(state);
    }

    @Override
    public void recordDraw(int player, int tileType, boolean isAkaTile) {
      Assert.assertEquals(player, seat);
      events.add(new EventLedger.Event(true, player, tileType, pool, hands));
    }

    @Override
    public void beforeDoraReveal(GameState state) {
      pool = new int[Tile.NUM_TILE_TYPES];
      EventLedger.pool(state, pool);
      hands = hands(state);
    }

    @Override
    public void recordNewDora(int physicalTileId) {
      events.add(new EventLedger.Event(false, -1, Tile.typeOf(physicalTileId), pool, hands));
    }
  }

  @Test(timeOut = 600_000)
  public void restoredEventsMatchTheEngine() {
    var shapes = new HandShapeAnalyzer();
    int reveals = 0, rinshan = 0, compared = 0, doubleReveals = 0;
    for (long seed = 1; seed <= 8000; seed++) {
      var recorder = new Recorder();
      var engine = new GameEngine(SeedMixer.mix64(seed * 6_700_417L), recorder);
      engine.getState().enableSeatStreamDraws();
      var tracker = new EventLedger.Tracker();
      List<EventLedger.Event> traced = new ArrayList<>();
      List<Slot> slots = null;
      var random = new SplittableRandom(seed);
      LedgerGames.Style style = seed % 2 == 0 ? LedgerGames.CALLS : LedgerGames.KANS;
      GameStepResult step = engine.stepHanchan();
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        List<EngineDecisionPoint> points = awaiting.decisions();
        GameState state = engine.getState();
        if (slots == null) {
          // 最初の境界の前の出来事（親の最初のツモ）は数えない
          tracker.observe(state, points, List.of(), null);
          slots = LedgerGames.slots();
          tracker.trace(traced);
          recorder.events.clear();
        } else {
          int before = traced.size();
          tracker.observe(state, points, slots, null);
          int reveal = 0;
          for (int k = before; k < traced.size(); k++) if (!traced.get(k).draw()) reveal++;
          if (reveal >= 2) doubleReveals++;
          Assert.assertEquals(traced.size(), recorder.events.size(), "seed " + seed + " events");
          for (int k = before; k < traced.size(); k++) {
            EventLedger.Event restored = traced.get(k), engineEvent = recorder.events.get(k);
            String where = "seed " + seed + " event " + k + " " + engineEvent.draw() + "/" + engineEvent.tile();
            Assert.assertEquals(restored.draw(), engineEvent.draw(), where);
            Assert.assertEquals(restored.seat(), engineEvent.seat(), where);
            Assert.assertEquals(restored.tile(), engineEvent.tile(), where);
            Assert.assertEquals(restored.pool(), engineEvent.pool(), where + " pool");
            Assert.assertTrue(Arrays.deepEquals(restored.hands(), engineEvent.hands()), where + " hands");
            compared++;
            if (!engineEvent.draw()) reveals++;
          }
          if (state.turnEvent() instanceof TurnEvent.Draw draw && draw.isRinshanDraw()) rinshan++;
        }
        step = engine.commitDecisions(LedgerGames.select(state, points, style, random, shapes, null));
      }
    }
    Assert.assertTrue(compared > 50_000, "compared " + compared);
    Assert.assertTrue(reveals >= 100, "dora reveals " + reveals);
    Assert.assertTrue(rinshan >= 100, "rinshan boundaries " + rinshan + " (two reveals at one boundary: " + doubleReveals + ")");
  }
}
