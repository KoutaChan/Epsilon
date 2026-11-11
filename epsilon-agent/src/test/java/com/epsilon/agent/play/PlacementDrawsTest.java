package com.epsilon.agent.play;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 割り当ての番号が乱数に入らないことを確かめる。one-hot でない分布を返し、卓が共通乱数で手を引く打ち手を全ての席に置くと、同じ牌山の
 * 全ての割り当てが同じ半荘になる。行動方策があってもなくても、どちらのツモの順でも、四人打ちでも三人打ちでも同じ。牌山が違えば
 * 半荘も違う。
 */
public class PlacementDrawsTest {

  private static final long RUN_SEED = 4_101L, ITERATION_SEED = 77_003L, BASE = 300L;
  private static final int WALLS = 3, PLACEMENTS = 4;

  /** 半荘ごとの主経路の要約。判断と選んだ手、局末の点数の移動、終局の点数をハッシュに混ぜる。 */
  static final class Digests implements GameObserver {
    final Map<GameKey, long[]> games = new ConcurrentHashMap<>();

    private long[] of(GameKey game) {
      return games.computeIfAbsent(game, key -> new long[] {0x5eedL});
    }

    private static void mix(long[] state, long value) {
      state[0] = SeedMixer.mix64(state[0] ^ SeedMixer.mix64(value));
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      mix(of(game), decisions.seat(slot) * 1_000_003L + decisions.legal(slot, choice));
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      long[] state = of(game);
      for (int seat = 0; seat < full.variant().seats(); seat++) mix(state, end.delta(seat));
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      long[] state = of(game);
      for (int seat = 0; seat < last.variant().seats(); seat++) mix(state, last.score(seat));
    }
  }

  @Test(timeOut = 600_000)
  public void everyPlacementOfAWallDrawsTheSameGame() {
    List<GameKey> games = new ArrayList<>();
    for (int wall = 0; wall < WALLS; wall++) {
      for (int placement = 0; placement < PLACEMENTS; placement++) {
        games.add(new GameKey(BASE + wall, placement));
      }
    }
    try (var table = new TableLoop(new TableSettings(6, 3, 200L, 4))) {
      for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
        for (DrawOrder order : DrawOrder.values()) {
          for (Behaviour behaviour : new Behaviour[] {null, new Behaviour(0.1, 0, CrnScheme.TWO_STAGE_FLOOR)}) {
            String label = variant + " " + order + " " + behaviour;
            var digests = new Digests();
            // 割り当てごとに別の打ち手を置く（打ち手は状態を持たないので、どの割り当てでも打ち方は同じ）
            TableLoop.Result result =
                table.play(
                    new TablePlan<>(
                        variant,
                        RUN_SEED,
                        ITERATION_SEED,
                        games,
                        order,
                        behaviour,
                        game -> {
                          List<TableSeat<Query>> seats = new ArrayList<>();
                          for (int seat = 0; seat < variant.seats(); seat++) seats.add(new TableLoopTest.InlineSeat());
                          return seats;
                        },
                        digests),
                    TableHooks.plain());
            Assert.assertEquals(result.games(), WALLS * PLACEMENTS, label);
            Assert.assertEquals(digests.games.size(), WALLS * PLACEMENTS, label);
            for (int wall = 0; wall < WALLS; wall++) {
              long first = digests.games.get(new GameKey(BASE + wall, 0))[0];
              for (int placement = 1; placement < PLACEMENTS; placement++) {
                Assert.assertEquals(digests.games.get(new GameKey(BASE + wall, placement))[0], first, label + " wall " + (BASE + wall) + " placement " + placement);
              }
              if (wall > 0) {
                Assert.assertNotEquals(digests.games.get(new GameKey(BASE + wall - 1, 0))[0], first, label + " walls " + (BASE + wall - 1) + " and " + (BASE + wall));
              }
            }
          }
        }
      }
    }
  }
}
