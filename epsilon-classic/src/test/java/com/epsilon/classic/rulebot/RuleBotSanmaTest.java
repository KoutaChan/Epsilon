package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.RandomPlay;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 三人打ちでルールベースの打ち手を確かめる。ランダムな自己対局の全ての判断で合法手を選び、北を抜けるときは、和了できる場合を
 * 除いて必ず抜く。3 席ともルールベースの打ち手の半荘は終局まで進み、持ち点と供託の合計が変わらず、北抜きと和了がある。同じシードなら
 * 同じ半荘になる。
 */
public class RuleBotSanmaTest {

  private static final Logger LOG = LoggerFactory.getLogger(RuleBotSanmaTest.class);

  private static final Variant VARIANT = Variant.TENHOU_3P;

  /** ランダムな自己対局のステップ数。 */
  private static final int RANDOM_STEPS = 60_000;

  /** 1 つの半荘のステップ数の上限。 */
  private static final int GAME_STEPS = 100_000;

  @Test
  public void everyRandomDecisionGetsALegalChoiceAndNorthIsExtracted() {
    RuleBot bot = new RuleBot(VARIANT);
    RandomPlay play = new RandomPlay(VARIANT, 23);
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
    Observation observation = new Observation(VARIANT);
    int decisions = 0;
    int nukiOffered = 0;
    int nukiResponses = 0;
    for (int step = 0; step < RANDOM_STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions open = play.engine.decisions();
        for (int slot = 0; slot < open.size(); slot++) {
          batch.clear();
          observation.bind(batch, writer.capture(play.engine.view(open.seat(slot)), open, slot, batch));
          int choice = bot.act(observation);
          assertTrue(choice >= 0 && choice < open.legalCount(slot), choice + " of " + open.legalCount(slot));
          boolean win = false;
          boolean nuki = false;
          for (int index = 0; index < open.legalCount(slot); index++) {
            Action.Type type = Action.of(open.legal(slot, index)).type();
            win |= type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI;
            nuki |= type == Action.Type.NUKI;
          }
          if (nuki && !win) {
            nukiOffered++;
            assertEquals(Action.of(open.legal(slot, choice)).type(), Action.Type.NUKI);
          }
          nukiResponses += open.kind(slot) == EngineDecisionKind.NUKI_RESPONSE ? 1 : 0;
          decisions++;
        }
      }
      play.step();
    }
    LOG.info("Three-player rule bot: {} random decisions, {} nuki offered, {} nuki responses", decisions, nukiOffered,
        nukiResponses);
    assertTrue(decisions > 20_000, decisions + " decisions");
    assertTrue(nukiOffered > 0, "no nuki offered");
  }

  @Test
  public void ruleBotsPlaySeededThreePlayerGamesToTheEnd() {
    int wins = 0;
    int nuki = 0;
    for (long seed = 1; seed <= 3; seed++) {
      BotTable table = new BotTable(new RuleBot(VARIANT), seed);
      table.finishGame(GAME_STEPS);
      TableState last = table.engine().roundEnd().next();
      assertTrue(last.gameOver());
      int total = Variant.DEPOSIT * last.kyotaku();
      int[] scores = new int[VARIANT.seats()];
      for (int seat = 0; seat < scores.length; seat++) {
        scores[seat] = last.score(seat);
        total += scores[seat];
      }
      assertEquals(total, VARIANT.seats() * VARIANT.startPoints(), Arrays.toString(scores));
      LOG.info(
          "Three-player rule bot game {}: {} decisions, {} rounds, {} wins, {} riichi, {} nuki, final scores {}",
          seed,
          table.decisions(),
          table.rounds(),
          table.wins(),
          table.riichi(),
          table.nuki(),
          Arrays.toString(scores));
      wins += table.wins();
      nuki += table.nuki();
    }
    assertTrue(wins > 0 && nuki > 0, wins + " wins and " + nuki + " nuki");
  }

  @Test
  public void aSeedPlaysTheSameThreePlayerGameTwice() {
    BotTable first = new BotTable(new RuleBot(VARIANT), 7);
    BotTable second = new BotTable(new RuleBot(VARIANT), 7);
    first.finishGame(GAME_STEPS);
    second.finishGame(GAME_STEPS);
    assertEquals(second.decisions(), first.decisions());
    assertEquals(second.digest(), first.digest());
  }
}
