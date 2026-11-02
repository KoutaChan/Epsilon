package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.River;
import com.epsilon.core.Tile;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 決まった対局の組（{@link GoldenGames}: 一様・偏りの打ち方の半荘と分岐、作った場面、天鳳の牌譜）の全ての判断と局末で確かめる、エンジンの
 * 不変条件。
 *
 * <ul>
 *   <li>牌: 4 席の手牌・副露・鳴かれていない河・牌山の残り・王牌で 136 枚。牌の種類ごとに見える枚数（手牌・副露・河・公開済みのドラ表示牌）は
 *       4 枚以下、赤の 5 は色ごとに 1 枚以下。
 *   <li>点: 持ち点と供託（1 本 1000 点）の和は局の中でも局をまたいでも変わらない（半荘と牌譜は 100000、場面は局の初めの値）。局の結果の
 *       点の移動の和は 0。
 *   <li>判断の種類: 観測と合法手だけで決めた種類（{@link EngineDecisionKind#of}）がエンジンの判断点の種類と等しい。
 *   <li>決定性: 同じ種で打ち直すと、判断・合法手・選んだ手・局の結果・牌譜の判断が同じ順で現れる。
 * </ul>
 *
 * <p>場面が筋どおりの結果で終わることと、場面の分岐が主経路と同じ結果になることは {@link GoldenGames#play} が確かめる。
 */
public class EngineInvariantsTest {
  /** 王牌の枚数。嶺上を引くたびに牌山の最後の 1 枚が王牌へ回るので、局の中で変わらない。 */
  private static final int DEAD_WALL = 14;

  /** 半荘と天鳳の牌譜の持ち点と供託の和（25000 点 × 4 席）。 */
  private static final int HANCHAN_POINTS = 100_000;

  /** 赤の 5 がある牌の種類（萬子・筒子・索子）。 */
  private static final int[] AKA_TYPES = {Tile.M5, Tile.P5, Tile.S5};

  @Test(timeOut = 600_000)
  public void everyDecisionKeepsTilesPointsAndDecisionKinds() {
    GoldenGames.Coverage coverage = GoldenGames.play(new Invariants());
    for (GoldenGames.Path path : GoldenGames.Path.values()) {
      Assert.assertTrue(
          coverage.count(path) > 0, "the hanchan sections never reach " + path + ": " + coverage);
    }
  }

  @Test(timeOut = 600_000)
  public void theSameSeedsPlayTheSameGames() {
    Digest first = new Digest();
    GoldenGames.play(first);
    Digest second = new Digest();
    GoldenGames.play(second);
    Assert.assertTrue(first.count > 0, "no decision was played");
    Assert.assertEquals(second.count, first.count, "decision count");
    Assert.assertEquals(
        second.hash.value(), first.hash.value(), "the same seeds played different games");
  }

  /** 判断と局末ごとに牌・点・判断の種類を確かめる訪問者。 */
  private static final class Invariants implements GoldenGames.Visitor {
    /** 今の主経路（半荘か場面）のエンジン。分岐はこの経路の持ち点と供託の和を引き継ぐ。 */
    private GameEngine route;

    /** 今の主経路の持ち点と供託の和。 */
    private int routePoints;

    @Override
    public void decisions(
        GameEngine engine, List<EngineDecisionPoint> points, GoldenGames.Position at) {
      GameState state = engine.getState();
      String where = GoldenGames.Position.describe(at.pack(points.getFirst().id()));
      if (at.fork() == 0 && engine != route) {
        route = engine;
        routePoints = pointsAndSticks(state);
        if (at.section() != GoldenGames.Section.SCENE) {
          Assert.assertEquals(routePoints, HANCHAN_POINTS, "points and sticks at " + where);
        }
      }
      Assert.assertEquals(pointsAndSticks(state), routePoints, "points and sticks at " + where);
      requireTilesConserved(state, where);
      for (EngineDecisionPoint point : points) {
        Assert.assertEquals(
            EngineDecisionKind.of(engine.observation(point.player()), point.legalActions()),
            point.kind(),
            "decision kind from the observation at "
                + GoldenGames.Position.describe(at.pack(point.id())));
      }
    }

    @Override
    public void settled(GameEngine engine, RoundSettlement settlement) {
      RoundResult result = settlement.result();
      int moved = 0;
      int total =
          settlement.transition() instanceof RoundTransition.NextRound next
              ? next.kyotakuCount() * HanchanProgression.RIICHI_COST
              : 0;
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        moved += result.pointDelta().get(seat);
        total += settlement.finalScore(seat);
      }
      Assert.assertEquals(moved, 0, "point movement of " + result);
      Assert.assertEquals(total, routePoints, "points and sticks after " + result);
    }

    @Override
    public void replayed(
        GameState state,
        int seat,
        List<Action> legal,
        int chosenSlot,
        int stateId,
        GoldenGames.Position at) {
      String where = GoldenGames.Position.describe(at.pack(stateId));
      Assert.assertEquals(pointsAndSticks(state), HANCHAN_POINTS, "points and sticks at " + where);
      requireTilesConserved(state, where);
    }
  }

  /** 持ち点と供託の和。 */
  private static int pointsAndSticks(GameState state) {
    int total = state.getKyotakuCount() * HanchanProgression.RIICHI_COST;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) total += state.getScore(seat);
    return total;
  }

  /** 牌の数え上げ（クラスの説明の「牌」）。鳴かれた打牌は副露の牌として 1 回だけ数える。 */
  private static void requireTilesConserved(GameState state, String where) {
    int[] seen = new int[Tile.NUM_TILE_TYPES];
    int[] aka = new int[AKA_TYPES.length];
    int placed = 0;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      Hand hand = state.hand(seat);
      placed += hand.concealedTileCount();
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) seen[tile] += hand.count(tile);
      countAka(aka, hand.concealedAkaMask());
      for (int index = 0; index < hand.meldCount(); index++) {
        Meld meld = hand.meld(index);
        placed += meld.size();
        for (int tile = 0; tile < meld.size(); tile++) seen[meld.tileAt(tile)]++;
        if (meld.containsAkaTile()) {
          countAka(aka, AkaTileMask.includeTile(AkaTileMask.EMPTY, meld.akaTileType()));
        }
      }
      River river = state.river(seat);
      for (int index = 0; index < river.size(); index++) {
        if (river.called(index)) continue;
        int tile = river.tileType(index);
        placed++;
        seen[tile]++;
        countAka(aka, AkaTileMask.includeTileIfAka(AkaTileMask.EMPTY, tile, river.aka(index)));
      }
    }
    DoraState dora = state.doraState();
    for (int index = 0; index < dora.indicatorCount(); index++) {
      seen[dora.indicatorTileType(index)]++;
    }
    countAka(aka, dora.visibleIndicatorAkaMask());

    Assert.assertEquals(
        placed + state.remainingWallTiles() + DEAD_WALL, Tile.NUM_TILES, "tile count at " + where);
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      Assert.assertTrue(
          seen[tile] <= Tile.TILES_PER_TYPE,
          seen[tile] + " visible copies of tile type " + tile + " at " + where);
    }
    for (int color = 0; color < AKA_TYPES.length; color++) {
      Assert.assertTrue(
          aka[color] <= 1,
          aka[color] + " red fives of tile type " + AKA_TYPES[color] + " at " + where);
    }
  }

  private static void countAka(int[] aka, int mask) {
    for (int color = 0; color < AKA_TYPES.length; color++) {
      if (AkaTileMask.containsTile(mask, AKA_TYPES[color])) aka[color]++;
    }
  }

  /** 判断・合法手・選んだ手・局の結果・牌譜の判断を、現れた順に混ぜる訪問者。 */
  private static final class Digest implements GoldenGames.Visitor {
    private final GoldenHash hash = new GoldenHash();

    /** 訪れた判断（エンジンの判断と牌譜の判断）の数。 */
    private int count;

    @Override
    public void decisions(
        GameEngine engine, List<EngineDecisionPoint> points, GoldenGames.Position at) {
      for (EngineDecisionPoint point : points) {
        hash.add(point.id()).add(point.player()).add(point.kind().ordinal());
        addLegal(point.legalActions());
        count++;
      }
    }

    @Override
    public void committed(GameEngine engine, List<EngineDecisionOutcome> outcomes) {
      for (EngineDecisionOutcome outcome : outcomes) {
        hash.add(outcome.selectedAction().toIndex()).add(outcome.selectedActionWasExecuted());
      }
    }

    @Override
    public void settled(GameEngine engine, RoundSettlement settlement) {
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        hash.add(settlement.result().pointDelta().get(seat)).add(settlement.finalScore(seat));
      }
    }

    @Override
    public void replayed(
        GameState state,
        int seat,
        List<Action> legal,
        int chosenSlot,
        int stateId,
        GoldenGames.Position at) {
      hash.add(seat).add(chosenSlot).add(stateId);
      addLegal(legal);
      count++;
    }

    @Override
    public void replayEnded(int[] finalScores) {
      for (int score : finalScores) hash.add(score);
    }

    private void addLegal(List<Action> legal) {
      hash.add(legal.size());
      for (Action action : legal) hash.add(action.toIndex());
    }
  }
}
