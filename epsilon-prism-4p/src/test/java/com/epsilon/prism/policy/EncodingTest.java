package com.epsilon.prism.policy;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.AgariOrder;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.RowTokens;
import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
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

  /**
   * 推論のクランプ（{@link Encoding#answer}）: CERTIFIED では首位確定の和了を他手の最大の nextUp、最下位確定の和了を他手の最小の
   * nextDown へ厳密にクランプし、証明辺の無い行は LEGACY とビット同一にする。LEGACY は全行をクランプしない。
   */
  @Test
  public void certifiedAnswerClampsCertifiedEdgesAndKeepsLegacyAndNonEdgeRowsBitIdentical() {
    List<long[]> rows =
        List.of(
            southFourTsumoWords(new int[] {40_000, 20_000, 20_000, 20_000}),
            southFourTsumoWords(new int[] {5_000, 35_000, 30_000, 30_000}),
            southFourTsumoWords(new int[] {25_000, 40_000, 15_000, 20_000}));
    List<PolicyRequest> legacyReq = PolicyRequests.requests(rows);
    List<PolicyRequest> certReq = PolicyRequests.requests(rows);

    try (var legacyEncoder =
            new BatchEncoder(
                FOUR_PLAYER,
                BatchEncoder.UNBOUNDED,
                BatchEncoder.Memory.DIRECT,
                false,
                AgariOrder.Mode.LEGACY);
        var certEncoder =
            new BatchEncoder(
                FOUR_PLAYER,
                BatchEncoder.UNBOUNDED,
                BatchEncoder.Memory.DIRECT,
                false,
                AgariOrder.Mode.CERTIFIED)) {
      Encoding legacyEnc = legacyEncoder.acquire(legacyReq);
      Encoding certEnc = certEncoder.acquire(certReq);
      try {
        legacyEncoder.encode(legacyEnc, null).join();
        certEncoder.encode(certEnc, null).join();
        int totalActions = legacyEnc.actions();
        float[] raw = new float[totalActions];
        int at = 0;
        Observation reader = new Observation(Variant.TENHOU_4P);
        int[] winIndex = new int[rows.size()];
        for (int r = 0; r < rows.size(); r++) {
          Observation o = reader.bind(rows.get(r), 0);
          winIndex[r] = -1;
          for (int a = 0; a < o.legalActions(); a++) {
            boolean win = o.action(a).type() == Action.Type.TSUMO_AGARI;
            if (win) winIndex[r] = a;
            // 行 0 (CERT_TOP) は和了が低く、行 1 (CERT_BOTTOM) は和了が高く、行 2 (CERT_MID) も和了が低い逆転スコアにする
            raw[at++] = r == 1 ? (win ? 0.9f : 0.2f) : (win ? -0.4f : 0.5f);
          }
          assertTrue(winIndex[r] >= 0);
        }
        legacyEnc.answer(FloatBuffer.wrap(raw), PriorSettings.STANDARD);
        certEnc.answer(FloatBuffer.wrap(raw), PriorSettings.STANDARD);

        // 行 0: CERT_TOP -> クランプされて和了が選ばれる
        int w0 = winIndex[0];
        assertEquals(legacyReq.get(0).scores[w0], -0.4f, 0f);
        assertEquals(
            Float.floatToRawIntBits(certReq.get(0).scores[w0]),
            Float.floatToRawIntBits(Math.nextUp(0.5f)));
        assertEquals(certReq.get(0).sigma[w0], 1f, 0f);

        // 行 1: CERT_BOTTOM -> クランプされて和了は選ばれない
        int w1 = winIndex[1];
        assertEquals(legacyReq.get(1).scores[w1], 0.9f, 0f);
        assertEquals(
            Float.floatToRawIntBits(certReq.get(1).scores[w1]),
            Float.floatToRawIntBits(Math.nextDown(0.2f)));
        assertEquals(certReq.get(1).sigma[w1], 0f, 0f);

        // 行 2: CERT_MID（順序辺なし）-> LEGACY とビット同一
        for (int a = 0; a < legacyReq.get(2).legalCount(); a++) {
          assertEquals(
              Float.floatToRawIntBits(certReq.get(2).scores[a]),
              Float.floatToRawIntBits(legacyReq.get(2).scores[a]));
          assertEquals(
              Float.floatToRawIntBits(certReq.get(2).sigma[a]),
              Float.floatToRawIntBits(legacyReq.get(2).sigma[a]));
        }
      } finally {
        legacyEncoder.release(legacyEnc);
        certEncoder.release(certEnc);
      }
    }
  }

  /** policy.json の読み書き: agariOrder が欠けていれば LEGACY、CERTIFIED は往復一致、未知・不正な値は IOException。 */
  @Test
  public void policyParamsReadsLegacyDefaultRoundTripsCertifiedAndRejectsUnknownMode()
      throws Exception {
    Path dir = Files.createTempDirectory("prism-policy-params");
    try {
      Path file = dir.resolve("policy.json");
      Files.writeString(file, "{\"version\": 7}\n", StandardCharsets.UTF_8);
      PolicyParams missing = PolicyParams.read(file);
      assertEquals(missing, PolicyParams.of(7, AgariOrder.Mode.LEGACY));

      PolicyParams.of(7, AgariOrder.Mode.LEGACY).write(file);
      assertFalse(Files.readString(file, StandardCharsets.UTF_8).contains("agariOrder"));
      assertEquals(PolicyParams.read(file), PolicyParams.of(7, AgariOrder.Mode.LEGACY));

      PolicyParams.of(8, AgariOrder.Mode.CERTIFIED).write(file);
      assertTrue(Files.readString(file, StandardCharsets.UTF_8).contains("CERTIFIED"));
      assertEquals(PolicyParams.read(file), PolicyParams.of(8, AgariOrder.Mode.CERTIFIED));

      for (String bad :
          List.of(
              "{\"version\": 8, \"agariOrder\": \"UNKNOWN\"}",
              "{\"version\": 8, \"agariOrder\": null}",
              "{\"version\": 8, \"agariOrder\": 1}")) {
        Files.writeString(file, bad, StandardCharsets.UTF_8);
        assertThrows(IOException.class, () -> PolicyParams.read(file));
      }
    } finally {
      for (Path p : Files.list(dir).toList()) {
        Files.deleteIfExists(p);
      }
      Files.deleteIfExists(dir);
    }
  }

  private static long[] southFourTsumoWords(int[] scores) {
    Engine engine = new Engine(Variant.TENHOU_4P);
    RuleWorkspace ws = new RuleWorkspace(Variant.TENHOU_4P);
    new Scenes(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, scores))
        .hand(0, "123m456m789m11z56p")
        .hand(1, "19m258p369s147s5z6z")
        .hand(2, "19m369p258s147p6z7z")
        .hand(3, "19m258p369s258s3z7z")
        .indicator("9s")
        .draws("8s8p7p3z1z4p")
        .start(engine, ws);
    for (int guard = 0; guard < 32; guard++) {
      Decisions decisions = engine.decisions();
      int slot = decisions.slotOf(0);
      if (slot >= 0) {
        for (int a = 0; a < decisions.legalCount(slot); a++) {
          if (Action.of(decisions.legal(slot, a)).type() == Action.Type.TSUMO_AGARI) {
            ObservationWriter writer = new ObservationWriter(Variant.TENHOU_4P);
            ObservationBatch batch = new ObservationBatch(Variant.TENHOU_4P, 1 << 12);
            int index = writer.capture(engine.view(0), decisions, slot, batch);
            Observation o = new Observation(Variant.TENHOU_4P).bind(batch, index);
            return Arrays.copyOfRange(
                batch.words(), batch.start(index), batch.start(index) + o.longs());
          }
        }
      }
      int[] choices = new int[decisions.size()];
      for (int s = 0; s < decisions.size(); s++) {
        int chosen = 0;
        for (int a = 0; a < decisions.legalCount(s); a++) {
          Action action = Action.of(decisions.legal(s, a));
          if (action.type() == Action.Type.PASS
              || (action.type() == Action.Type.DAHAI && action.selection().tsumogiri())) {
            chosen = a;
            break;
          }
        }
        choices[s] = chosen;
      }
      assertEquals(engine.commit(choices, ws), Boundary.DECISIONS);
    }
    throw new AssertionError("never reached TSUMO_AGARI");
  }
}
