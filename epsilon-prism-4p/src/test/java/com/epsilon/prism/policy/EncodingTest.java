package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;

import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.RowTokens;
import com.epsilon.prism4p.input.InputSchema;
import java.lang.reflect.Field;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
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
      assertFalse(engine.contains(type), "Encoding." + field.getName() + " holds " + type.getSimpleName());
    }
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

    var encoder = new BatchEncoder(InputSchema.FORMAT, BatchEncoder.UNBOUNDED, BatchEncoder.Memory.DIRECT);
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
