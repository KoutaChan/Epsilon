package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertTrue;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowTokens;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.play.SnapshotScorer;
import com.epsilon.prism.run.PrismConfig;
import java.lang.reflect.Field;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.annotations.Test;

/**
 * 符号化（{@link Encoding}）は問い合わせが運ぶ記録 O だけを読み、公開の観測（その先の局面）を持たず読まない（捕捉は進行のスレッドがする）。
 *
 * <p>判断の境界ごとに O を捕捉した問い合わせを貯め、局面を先へ進めてから（観測は局面の借用の参照なので、捕捉の時点と違う局面を指す）まとめて
 * 符号化する。塊の行は、どれも捕捉した O を 1 行の塊に書いた入力と同じ。符号化が観測を読んだなら、進んだ局面の行になって合わない（観測の
 * 公開の出来事の数が O の頭の数と違う問い合わせが多いことも確かめる）。合法手は答えを書けるように写しを持たせる。
 */
public class EncodingTest {
  private static final int GAMES = 3;
  private static final int REQUESTS = 300;

  @Test
  public void encodingHoldsNoPublicObservation() {
    for (Field field : Encoding.class.getDeclaredFields()) {
      Class<?> type = field.getType();
      while (type.isArray()) {
        type = type.getComponentType();
      }
      assertFalse(
          type == PublicObservation.class || type == GameState.class,
          "Encoding." + field.getName() + " holds " + type.getSimpleName());
    }
  }

  @Test(timeOut = 300_000)
  public void encodesOnlyTheCapturedObservations() {
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(4096);
    List<PolicyRequest> requests = new ArrayList<>();
    GameEngine[] engines = new GameEngine[GAMES];
    GameStepResult[] steps = new GameStepResult[GAMES];
    SplittableRandom random = new SplittableRandom(11);
    var selections = new EngineSelectionBuffer();
    for (int game = 0; game < GAMES; game++) {
      engines[game] = new GameEngine(31 + game);
      steps[game] = engines[game].stepHanchan();
    }
    while (requests.size() < REQUESTS) {
      for (int game = 0; game < GAMES; game++) {
        if (steps[game] instanceof GameStepResult.HanchanEnded) {
          engines[game] = new GameEngine(31 + game + GAMES * requests.size());
          steps[game] = engines[game].stepHanchan();
        }
        if (!(steps[game] instanceof GameStepResult.AwaitingDecisions awaiting)) {
          steps[game] = engines[game].stepHanchan();
          continue;
        }
        selections.clear();
        for (EngineDecisionPoint point : awaiting.decisions()) {
          List<Action> legal = point.legalActions();
          if (legal.size() > 1) {
            PolicyRequest request =
                SnapshotScorer.request(engines[game].observation(point.player()), List.copyOf(legal));
            request.capture(writer, arena);
            requests.add(request.copy(true));
          }
          selections.add(point.id(), legal.get(random.nextInt(legal.size())));
        }
        steps[game] = engines[game].commitDecisions(selections);
      }
    }

    long[] expected = new long[requests.size()];
    int moved = 0;
    RowTokens oneRow = new RowTokens();
    Observation reader = new Observation();
    for (int row = 0; row < expected.length; row++) {
      PolicyRequest request = requests.get(row);
      expected[row] = oneRow.tokenize(request.recordWords(), request.recordBase());
      int captured = reader.bind(request.recordWords(), request.recordBase()).events();
      if (request.observation().events().size() != captured) {
        moved++;
      }
    }
    assertTrue(moved > REQUESTS / 2, "the game state must have moved on: " + moved);

    var encoder = new BatchEncoder(BatchEncoder.UNBOUNDED, false, BatchEncoder.Memory.DIRECT);
    Encoding encoding = encoder.acquire(requests);
    try {
      encoder.encode(encoding, null).join();
      long[] served = new long[requests.size()];
      int at = 0;
      for (Encoding.Chunk chunk : encoding.chunks()) {
        HostBatch batch = chunk.batch().input();
        BatchPlan plan = batch.plan();
        for (int row = 0; row < plan.rows(); row++) {
          served[at++] = RowTokens.row(batch, row);
        }
      }
      assertEquals(at, served.length);
      Arrays.sort(expected);
      Arrays.sort(served);
      assertEquals(served, expected, "the encoded rows are not the captured observations");
      encoding.answer(
          FloatBuffer.allocate(encoding.actions()),
          PrismConfig.fromSettings(EpsilonSettings.defaults()).priorSettings());
      for (PolicyRequest request : requests) {
        assertNotNull(request.sigma);
        assertEquals(request.sigma.length, request.legal().size());
      }
    } finally {
      encoder.release(encoding);
      encoder.close();
    }
  }
}
