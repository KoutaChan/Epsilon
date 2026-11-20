package com.epsilon.prism.policy;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.RowTokens;
import java.lang.reflect.Field;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.testng.annotations.Test;

/**
 * 符号化（{@link Encoding}）のテスト。符号化は要求が運ぶ記録 O だけを読み、エンジンの局面（席のビュー・全体のビュー・エンジン・
 * その時点の判断）をフィールドに持たない。
 *
 * <p>新しいエンジンで打った半荘の判断の O を要求にして、まとめて符号化する。バッチのどの行も、同じ O を 1 行のバッチに書いた入力と
 * 同じになる。答えを書くと、要求ごとに合法手の数だけの σ が入る。
 */
public class EncodingTest {
  private static final int GAMES = 3;
  private static final int PER_GAME = 100;

  @Test
  public void encodingHoldsNoEngineState() {
    Set<Class<?>> engine = Set.of(SeatView.class, FullView.class, Engine.class, Decisions.class);
    for (Field field : Encoding.class.getDeclaredFields()) {
      Class<?> type = field.getType();
      while (type.isArray()) {
        type = type.getComponentType();
      }
      assertFalse(
          engine.contains(type), "Encoding." + field.getName() + " holds " + type.getSimpleName());
    }
  }

  @Test
  public void eventHistoriesNeedEventsThatSettleOnPublication() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new BatchEncoder(
                FOUR_PLAYER, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT, true));
    var causal = InputFormat.of(FOUR_PLAYER.variant(), EnumSet.allOf(InputFeature.class));
    new BatchEncoder(causal, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT, true).close();
  }

  @Test(timeOut = 60_000)
  public void reservedInputsCoverEveryChunkAndWakeOnReturn() {
    var causal = InputFormat.of(FOUR_PLAYER.variant(), EnumSet.allOf(InputFeature.class));
    int budget = causal.fixedTokens() + causal.maxEvents();
    var rows = PolicyRequests.requests(PolicyRequests.observations(31, TableWalk.Policy.CALLS, 14));
    try (var encoder = new BatchEncoder(causal, budget, BatchEncoder.Memory.DIRECT, true);
        var normal = new BatchEncoder(causal, budget, BatchEncoder.Memory.DIRECT, true)) {
      encoder.reserve(2, rows.size());
      var first = encoder.acquire(rows);
      var second = encoder.acquire(rows);
      var wake = encoder.whenFull();
      assertNotNull(wake);
      assertFalse(wake.isDone());
      assertThrows(IllegalStateException.class, encoder::borrow);
      var reference = normal.acquire(rows);
      normal.encode(reference, null).join();
      encoder.encode(first, null).join();
      assertTrue(first.chunks().size() > 1);
      assertEquals(first.chunks().size(), reference.chunks().size());
      var chunks = List.copyOf(first.chunks());
      for (int i = 0; i < chunks.size(); i++) {
        var expected = reference.chunks().get(i);
        var actual = chunks.get(i);
        assertEquals(actual.first(), expected.first());
        var used = actual.batch().plan().bytes();
        assertEquals(used, expected.batch().plan().bytes());
        assertEquals(
            actual.batch().input().storage().slice(0, used),
            expected.batch().input().storage().slice(0, used));
      }
      encoder.release(first);
      wake.join();
      assertNull(encoder.whenFull());
      assertThrows(IllegalArgumentException.class, () -> encoder.acquire(List.of()));
      first = encoder.acquire(rows);
      encoder.encode(first, null).join();
      for (int i = 0; i < chunks.size(); ++i) {
        assertSame(first.chunks().get(i).batch(), chunks.get(i).batch());
        assertSame(first.chunks().get(i).pages(), chunks.get(i).pages());
      }
      normal.release(reference);
      encoder.release(first);
      encoder.release(second);
    }
  }

  @Test
  public void closingInputReservationsWakesAnUnsubmittedAdmission() {
    var encoder = new BatchEncoder(FOUR_PLAYER, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT);
    encoder.reserve(1, 1);
    encoder.borrow();
    var wake = encoder.whenFull();
    encoder.close();
    assertTrue(wake.isCompletedExceptionally());
    assertThrows(IllegalStateException.class, encoder::borrow);
  }

  @Test(timeOut = 300_000)
  public void encodesOnlyTheCapturedObservations() {
    List<long[]> rows = new ArrayList<>();
    for (int game = 0; game < GAMES; game++) {
      rows.addAll(PolicyRequests.observations(31 + game, TableWalk.Policy.CALLS, PER_GAME));
    }
    List<PolicyRequest> requests = PolicyRequests.requests(rows);

    long[] expected = new long[requests.size()];
    RowTokens oneRow = new RowTokens();
    for (int row = 0; row < expected.length; row++) {
      PolicyRequest request = requests.get(row);
      expected[row] = oneRow.tokenize(request.recordWords(), request.recordBase());
    }

    var encoder = new BatchEncoder(FOUR_PLAYER, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT);
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
      encoding.answer(FloatBuffer.allocate(encoding.actions()), PriorSettings.STANDARD);
      for (PolicyRequest request : requests) {
        assertNotNull(request.sigma);
        assertEquals(request.sigma.length, request.legalCount());
      }
    } finally {
      encoder.release(encoding);
      encoder.close();
    }
  }
}
