package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.GameRecorder;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

/** テスト用の牌譜。全ての席をルールベースの打ち手にして 1 半荘を打ち、牌譜に書き取る。牌山はシードで決まる。 */
public final class RuleGames {

  private RuleGames() {}

  public static RecordedGame play(Variant variant, long seed, String id) {
    Engine engine = new Engine(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    GameRecorder recorder = new GameRecorder(variant);
    ObservationWriter writer = new ObservationWriter(variant);
    ObservationBatch batch = new ObservationBatch(variant, 1 << 12);
    Observation reader = new Observation(variant);
    int[] choices = new int[variant.seats() - 1];
    try (RuleBot bot = new RuleBot(variant)) {
      engine.startGame(seed, DrawOrder.SEAT_STREAMS);
      Boundary boundary = engine.advance(ws);
      while (true) {
        recorder.advanced(engine.full());
        if (boundary == Boundary.DECISIONS) {
          Decisions decisions = engine.decisions();
          for (int slot = 0; slot < decisions.size(); slot++) {
            batch.clear();
            int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
            choices[slot] = bot.act(reader.bind(batch, index));
          }
          boundary = engine.commit(choices, ws);
          continue;
        }
        recorder.roundEnded(engine.full(), engine.roundEnd());
        if (boundary == Boundary.GAME_END) {
          return recorder.gameEnded(
              engine.roundEnd().next(),
              new RecordedGame.Meta(id, RecordedGame.Source.TENHOU, Collections.nCopies(variant.seats(), "heuristic")));
        }
        boundary = engine.advance(ws);
      }
    }
  }

  /** 半荘を天鳳の牌譜の XML で {@code <id>.xml} に書く。 */
  public static void write(Path directory, RecordedGame game) throws IOException {
    Files.createDirectories(directory);
    Files.writeString(directory.resolve(game.meta().id() + ".xml"), TenhouXml.write(game), StandardCharsets.UTF_8);
  }
}
