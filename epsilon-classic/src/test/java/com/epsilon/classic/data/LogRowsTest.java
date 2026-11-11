package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 牌譜から作る行を確かめる。ルールベースの打ち手の半荘を再生して作った行では、記録した手がいつも合法手の中にあり、合法手は 2 つ以上
 * あり、ルールベースの打ち手が観測から選ぶ手と全て一致する（再生で作る観測が、打ったときの観測と同じだから）。行の数は、記録した手が
 * 分かり、合法手が 2 つ以上ある判断の数。検証用かどうかの分け方は牌譜 id だけで決まり、ファイルに書いて読み戻しても行は変わらない。
 */
public class LogRowsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void ruleBotRowsReplayToTheRecordedChoices() throws Exception {
    ClassicRows.Builder builder = ClassicRows.builder(VARIANT);
    LogRows logs = new LogRows(VARIANT, 0.1);
    int expected = 0;
    for (long seed = 1; seed <= 3; seed++) {
      RecordedGame game = RuleGames.play(VARIANT, seed, "rule-" + seed);
      expected += observedDecisions(game);
      Assert.assertEquals(logs.add(game, builder), observedDecisions(game));
    }
    ClassicRows rows = builder.build();
    Assert.assertEquals(rows.size(), expected);
    Assert.assertTrue(rows.size() > 100, "rows " + rows.size());
    Observation reader = new Observation(VARIANT);
    try (RuleBot bot = new RuleBot(VARIANT)) {
      for (int row = 0; row < rows.size(); row++) {
        Observation o = rows.bind(row, reader);
        Assert.assertTrue(o.legalActions() >= 2, "row " + row);
        Assert.assertTrue(rows.chosen(row) < o.legalActions(), "row " + row);
        Assert.assertEquals(bot.act(o), rows.chosen(row), "row " + row);
        Assert.assertEquals(rows.validation(row), LogRows.validation(rows.game(row).id(), 0.1));
      }
    }
    Path file = Files.createTempFile("classic-rows", ".bin");
    Path again = Files.createTempFile("classic-rows", ".bin");
    try {
      rows.write(file);
      ClassicRows back = ClassicRows.read(file);
      Assert.assertEquals(back.size(), rows.size());
      Assert.assertEquals(back.games(), rows.games());
      for (int row = 0; row < rows.size(); row++) {
        Assert.assertEquals(back.chosen(row), rows.chosen(row));
        Assert.assertEquals(back.gameIndex(row), rows.gameIndex(row));
        Assert.assertEquals(back.round(row), rows.round(row));
      }
      // 観測は long の列をそのまま書くので、読み直して書いたファイルはバイト単位で同じになる
      back.write(again);
      Assert.assertEquals(Files.readAllBytes(again), Files.readAllBytes(file));
    } finally {
      Files.deleteIfExists(file);
      Files.deleteIfExists(again);
    }
  }

  @Test
  public void aboutOneGameInTenIsForValidation() {
    int validation = 0;
    for (int index = 0; index < 4_000; index++) {
      validation += LogRows.validation("game-" + index, 0.1) ? 1 : 0;
      Assert.assertEquals(LogRows.validation("game-" + index, 0.1), LogRows.validation("game-" + index, 0.1));
    }
    Assert.assertTrue(validation > 330 && validation < 470, "validation games " + validation);
  }

  /** 割合を増やしても、検証用だった半荘は検証用のまま。割合は 0 より大きく 1 より小さくなければならない。 */
  @Test
  public void aLargerShareKeepsTheValidationGames() {
    int wider = 0;
    for (int index = 0; index < 4_000; index++) {
      String id = "game-" + index;
      if (LogRows.validation(id, 0.1)) {
        Assert.assertTrue(LogRows.validation(id, 0.3), id);
      }
      wider += LogRows.validation(id, 0.3) ? 1 : 0;
    }
    Assert.assertTrue(wider > 1_080 && wider < 1_320, "validation games " + wider);
    for (double share : new double[] {0, 1, -0.1, Double.NaN}) {
      Assert.assertThrows(IllegalArgumentException.class, () -> new LogRows(VARIANT, share));
    }
  }

  /** 記録の手が分かり、合法手が 2 つ以上の判断の数。 */
  private static int observedDecisions(RecordedGame game) {
    ReplayDriver driver = new ReplayDriver(VARIANT);
    RuleWorkspace ws = new RuleWorkspace(VARIANT);
    int count = 0;
    for (var round : game.rounds()) {
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) {
        var decisions = driver.engine().decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          if (driver.choice(slot) != ReplayDriver.UNOBSERVED && decisions.legalCount(slot) >= 2) {
            count++;
          }
        }
        boundary = driver.step(ws);
      }
    }
    return count;
  }
}
