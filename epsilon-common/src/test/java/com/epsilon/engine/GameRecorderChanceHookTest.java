package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 偶然の点のフックが牌山に触る前に呼ばれ、実現値の記録と対になり、記録処理がなくても対局が変わらないことを確かめる。 */
public class GameRecorderChanceHookTest {

  /** 何も記録しない記録処理。 */
  static class SilentRecorder implements GameRecorder {
    @Override
    public void startRound(int roundIndex, int honba, int kyotakuCount, int[] startingScores) {}

    @Override
    public void recordHaipai(int player, int[] concealedTileCounts, int concealedAkaMask) {}

    @Override
    public void recordDoraIndicators(int[] physicalTileIds) {}

    @Override
    public void recordDraw(int player, int tileType, boolean isAkaTile) {}

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

  /** 前後の対を確かめる記録処理。 */
  private static final class PairingRecorder extends SilentRecorder {
    final List<String> events = new ArrayList<>();
    int pendingPlayer = -1;
    boolean pendingRinshan;
    boolean pendingDora;
    int wallBefore;
    int handBefore;
    int indicatorsBefore;
    GameState state;
    int draws, rinshanDraws, doraReveals;

    @Override
    public void beforeDraw(int player, boolean rinshan, GameState state) {
      Assert.assertEquals(pendingPlayer, -1, "draw hook without its draw");
      Assert.assertFalse(pendingDora, "dora reveal hook without its reveal");
      pendingPlayer = player;
      pendingRinshan = rinshan;
      wallBefore = state.remainingWallTiles();
      handBefore = state.hand(player).concealedTileCount();
      this.state = state;
      events.add((rinshan ? "R" : "D") + player);
    }

    @Override
    public void beforeDoraReveal(GameState state) {
      Assert.assertEquals(pendingPlayer, -1, "draw hook without its draw");
      Assert.assertFalse(pendingDora);
      pendingDora = true;
      indicatorsBefore = state.doraState().indicatorCount();
      this.state = state;
      events.add("K");
    }

    @Override
    public void recordDraw(int player, int tileType, boolean isAkaTile) {
      Assert.assertEquals(player, pendingPlayer);
      Assert.assertFalse(pendingRinshan);
      Assert.assertEquals(state.remainingWallTiles(), wallBefore - 1);
      Assert.assertEquals(state.hand(player).concealedTileCount(), handBefore + 1);
      pendingPlayer = -1;
      draws++;
    }

    @Override
    public void recordRinshanDraw(int player, int tileType, boolean isAkaTile) {
      Assert.assertEquals(player, pendingPlayer);
      Assert.assertTrue(pendingRinshan);
      Assert.assertEquals(state.remainingWallTiles(), wallBefore - 1);
      Assert.assertEquals(state.hand(player).concealedTileCount(), handBefore + 1);
      pendingPlayer = -1;
      rinshanDraws++;
    }

    @Override
    public void recordNewDora(int physicalTileId) {
      Assert.assertTrue(pendingDora);
      Assert.assertEquals(state.doraState().indicatorCount(), indicatorsBefore + 1);
      pendingDora = false;
      doraReveals++;
    }
  }

  /** 槓を優先し、ほかは種で決まる一様な選び方で半荘を打つ。精算後の点数を局ごとに並べて返す。 */
  static List<int[]> play(long wallSeed, long choiceSeed, GameRecorder recorder) {
    var engine = new GameEngine(wallSeed, recorder);
    engine.getState().enableSeatStreamDraws();
    var random = new SplittableRandom(choiceSeed);
    var selections = new EngineSelectionBuffer();
    List<int[]> scores = new ArrayList<>();
    GameStepResult step = engine.stepHanchan();
    while (true) {
      switch (step) {
        case GameStepResult.AwaitingDecisions awaiting -> {
          selections.clear();
          for (EngineDecisionPoint point : awaiting.decisions()) {
            List<Action> legal = point.legalActions();
            Action chosen = legal.get(random.nextInt(legal.size()));
            for (Action action : legal)
              if (action.type() == Action.Type.ANKAN
                  || action.type() == Action.Type.KAKAN
                  || action.type() == Action.Type.DAIMINKAN) chosen = action;
            selections.add(point.id(), chosen);
          }
          step = engine.commitDecisions(selections);
        }
        case GameStepResult.RoundSettled settled -> {
          scores.add(settled.settlement().snapshotFinalScores());
          step = engine.stepHanchan();
        }
        case GameStepResult.HanchanEnded ended -> {
          scores.add(ended.settlement().snapshotFinalScores());
          return scores;
        }
        default -> throw new IllegalStateException("Unexpected " + step);
      }
    }
  }

  @Test
  public void hooksRunBeforeTheWallIsTouchedAndPairWithTheirEvents() {
    var recorder = new PairingRecorder();
    for (long game = 0; game < 6; game++) play(91 + game, 17 + game, recorder);
    Assert.assertEquals(recorder.pendingPlayer, -1);
    Assert.assertFalse(recorder.pendingDora);
    Assert.assertTrue(recorder.draws > 500, "draws=" + recorder.draws);
    Assert.assertTrue(recorder.rinshanDraws > 5, "rinshan=" + recorder.rinshanDraws);
    Assert.assertTrue(recorder.doraReveals > 5, "dora=" + recorder.doraReveals);
    // 即時公開の槓ドラ（暗槓）は嶺上ツモの前に公開する
    int revealThenRinshan = 0;
    for (int i = 1; i < recorder.events.size(); i++)
      if (recorder.events.get(i - 1).equals("K") && recorder.events.get(i).startsWith("R"))
        revealThenRinshan++;
    Assert.assertTrue(revealThenRinshan > 0);
  }

  @Test
  public void recordersDoNotChangeTheGame() {
    for (long game = 0; game < 4; game++) {
      List<int[]> plain = play(301 + game, 5 + game, null);
      List<int[]> recorded = play(301 + game, 5 + game, new PairingRecorder());
      Assert.assertEquals(recorded.size(), plain.size());
      for (int hand = 0; hand < plain.size(); hand++)
        Assert.assertEquals(recorded.get(hand), plain.get(hand), "hand " + hand);
      Assert.assertTrue(plain.stream().allMatch(s -> Arrays.stream(s).sum() <= 100_000));
    }
  }
}
