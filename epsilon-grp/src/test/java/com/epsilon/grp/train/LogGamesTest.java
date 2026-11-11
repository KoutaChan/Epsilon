package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.GameRecorder;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 牌譜から作る GRP の記録を確かめる。局末の卓は次の局の始まりの卓で、終局の点数は牌譜の終局の点数になる。半荘 ID は牌譜の id だけで
 * 決まる。
 */
public class LogGamesTest {

  @Test
  public void handEndsAreTheNextRoundStarts() {
    for (Variant variant : Variant.values()) {
      RecordedGame game = play(variant, 5);
      GameRecord record = LogGames.of(game);
      Assert.assertEquals(record.variant(), variant);
      Assert.assertEquals(record.handEnds().size(), game.rounds().size() - 1);
      for (int handEnd = 0; handEnd < record.handEnds().size(); handEnd++) {
        RoundStart next = game.rounds().get(handEnd + 1).start();
        GrpState state = record.handEnds().get(handEnd);
        Assert.assertEquals(state, new GrpState(variant, next.round(), next.honba(), next.kyotaku(), next.scores()));
      }
      Assert.assertEquals(record.finalScores(), game.finalScores());
      Assert.assertEquals(record.gameId(), LogGames.gameId(game.meta().id()));
      GameCohort cohort = LogGames.cohort(List.of(game));
      Assert.assertEquals(cohort.cohort(), LogGames.COHORT);
      Assert.assertEquals(cohort.games(), List.of(record));
    }
    Assert.assertNotEquals(LogGames.gameId("a"), LogGames.gameId("b"));
  }

  /** 乱数で手を選んで 1 半荘を打ち、牌譜に書き取る。 */
  private static RecordedGame play(Variant variant, long seed) {
    Random random = new Random(seed);
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    GameRecorder recorder = new GameRecorder(variant);
    int[] choices = new int[variant.seats() - 1];
    engine.startGame(random.nextLong(), DrawOrder.SEAT_STREAMS);
    Boundary boundary = engine.advance(ws);
    while (true) {
      recorder.advanced(engine.full());
      if (boundary == Boundary.DECISIONS) {
        Decisions decisions = engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          choices[slot] = random.nextBoolean() ? 0 : random.nextInt(decisions.legalCount(slot));
        }
        boundary = engine.commit(choices, ws);
        continue;
      }
      recorder.roundEnded(engine.full(), engine.roundEnd());
      if (boundary == Boundary.GAME_END) {
        return recorder.gameEnded(
            engine.roundEnd().next(),
            new RecordedGame.Meta("log-" + seed, RecordedGame.Source.TENHOU, Collections.nCopies(variant.seats(), "")));
      }
      boundary = engine.advance(ws);
    }
  }
}
