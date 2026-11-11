package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 手をランダムに選んで打った半荘を記録すると、ReplayDriver で記録と食い違わずに再生でき、天鳳の XML に書いて読み直しても同じになる。 */
public class GameRecorderTest {

  @Test
  public void recordedGamesReplayAndRoundTrip() {
    for (Variant variant : Variant.values()) {
      for (long seed = 1; seed <= 4; seed++) {
        RecordedGame game = play(variant, seed);
        new ReplayDriver(variant).replay(game, new RuleWorkspace(variant));
        Assert.assertEquals(TenhouXml.read(TenhouXml.write(game), game.meta().id()), game, variant + " " + seed);
      }
    }
  }

  @Test
  public void aRoundMustEndBeforeTheGame() {
    Variant variant = Variant.TENHOU_4P;
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    engine.startGame(7, DrawOrder.SEAT_STREAMS);
    engine.advance(ws);
    GameRecorder recorder = new GameRecorder(variant);
    recorder.advanced(engine.full());
    Assert.expectThrows(
        IllegalStateException.class,
        () -> recorder.gameEnded(engine.roundEnd().next(), RecordedGames.unnamed("open", variant)));
  }

  /** 判断の半分では最初の合法手（和了・槓・鳴き・立直が先に並ぶ）を、残りでは一様に手を選んで 1 半荘を打ち、記録する。 */
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
        return recorder.gameEnded(engine.roundEnd().next(), RecordedGames.unnamed("game-" + seed, variant));
      }
      boundary = engine.advance(ws);
    }
  }
}
