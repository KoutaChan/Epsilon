package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 主経路の対局の記録（{@link GameLogObserver}）を本物の収集（{@link CollectionDriver}、CPU の inline の方策、進行ワーカー 3 本）で確かめる:
 * 記録から組み直した次局 root の公開系列が収集の {@code RootState.publicPrefix}（{@code handValued} に渡る配列）とビットで同じで、局末の数が
 * 収集の hands と合い、記録の往復でも変わらない。記録を付けても付けなくてもコホートの行はビットで同じ。中の失敗は記録だけを止め、封印で投げる。
 */
public class GameLogObserverTest {
  private static final long SEED = 9_101L;
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  /** ラベル行・ロールイン・写しの中の行がどれも出る率（本番より高い）。 */
  private static final Sampling SAMPLING =
      new Sampling(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 床 ε_b = 0.05 の行動方策。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.V2);

  @Test(timeOut = 600_000)
  public void reconstructedPrefixesMatchTheCollectionBitwise() throws IOException {
    var tapped = new Tapped(GameLogObserver.open(4, PhiEntry.scoreRank(), FLOOR));
    Collected collected = collect(tapped, 3, 4);
    CohortGames games = tapped.log.seal(collected.result().counts());
    CollectionMetrics.Snapshot counts = collected.result().counts();
    Assert.assertEquals(games.games().size(), GAMES);
    Assert.assertEquals(games.handEnds(), counts.hands());
    long decisions = 0;
    for (long count : counts.mainDecisions()) decisions += count;
    Assert.assertEquals(
        games.header(),
        new GameLogHeader(4, PhiEntry.SCORE_RANK_ID, "", 0.05, 0.0, decisions, counts.mainNonGreedy()));
    assertPrefixes(games, tapped.prefixes, tapped.finals);
    // 記録の往復の後も同じ
    Path directory = Files.createTempDirectory("prism-gamelog");
    try {
      Path file = new GrpFiles(directory).games(4);
      GameLog.write(file, games);
      assertPrefixes(GameLog.read(file), tapped.prefixes, tapped.finals);
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  @Test(timeOut = 600_000)
  public void recordingDoesNotChangeTheRows() throws IOException {
    Collected off = collect(null, 3, 4);
    var log = GameLogObserver.open(4, PhiEntry.scoreRank(), FLOOR);
    Collected on = collect(log, 3, 4);
    Assert.assertTrue(off.rows().rows() > 0, "no rows");
    Assert.assertEquals(on.rows(), off.rows());
    Assert.assertEquals(on.result().counts().hands(), off.result().counts().hands());
    Assert.assertEquals(log.seal(on.result().counts()).handEnds(), on.result().counts().hands());
  }

  @Test
  public void failureStopsOnlyTheRecording() {
    var log = GameLogObserver.open(2, PhiEntry.scoreRank(), FLOOR);
    log.handProgressed(1, 1, 0, 0, new int[] {25000, 25000, 25000, 25000});
    log.gameEnded(1, new int[] {40000, 30000, 20000, 10000});
    // 点数が 4 つでない局末は記録を止める（例外は外へ出ない）
    log.handProgressed(2, 1, 0, 0, new int[] {25000, 25000, 25000});
    log.gameEnded(2, new int[] {40000, 30000, 20000, 10000});
    IllegalStateException stopped = Assert.expectThrows(IllegalStateException.class, () -> log.seal(counts(2, 3)));
    Assert.assertTrue(stopped.getMessage().contains("stopped"), stopped.getMessage());
    // 局末の数が収集と合わない記録は封印しない
    var mismatched = GameLogObserver.open(2, PhiEntry.scoreRank(), FLOOR);
    mismatched.handProgressed(1, 1, 0, 0, new int[] {25000, 25000, 25000, 25000});
    mismatched.gameEnded(1, new int[] {40000, 30000, 20000, 10000});
    Assert.assertEquals(mismatched.seal(counts(1, 2)).handEnds(), 2);
    Assert.expectThrows(IllegalStateException.class, () -> mismatched.seal(counts(1, 3)));
    // 終わっていない半荘が残る記録も封印しない
    var unfinished = GameLogObserver.open(2, PhiEntry.scoreRank(), FLOOR);
    unfinished.handProgressed(5, 1, 0, 0, new int[] {25000, 25000, 25000, 25000});
    Assert.expectThrows(IllegalStateException.class, () -> unfinished.seal(counts(0, 1)));
  }

  /** 収集した半荘の数。 */
  private static final int GAMES = 6;

  /**
   * 1 つの収集の結果と、封印したコホートの行の指紋。
   *
   * @param rows 行の指紋
   */
  record Collected(CollectionDriver.Result result, RowHash.Digest rows) {}

  /** 半荘 300..305 を observer 付きで集め、行を一時ディレクトリのコホートに書いて封印する。 */
  private static Collected collect(CollectionDriver.GameObserver observer, int workers, int concurrent)
      throws IOException {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < GAMES; id++) ids.add(300 + id);
    Path directory = Files.createTempDirectory("prism-rows");
    try (var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(concurrent, workers, new InferenceBatchingSettings(1000)));
        var phi = HandValue.scoreRank(PROFILE);
        var writer = new CohortWriter(directory, 4, 0, 2, CohortWriter.Tap.NONE)) {
      var cohort =
          new CollectionDriver.Cohort(SEED, SEED * 5, ids, PROFILE, SAMPLING, FLOOR, null, observer);
      var result = driver.collect(cohort, new HashedScorePolicy(), phi, new RowSink(writer));
      writer.seal();
      return new Collected(result, RowHash.of(directory));
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  /** 記録の半荘ごとに、組み直した接頭辞が収集の接頭辞とビットで同じで、最終の点数も同じ。 */
  private static void assertPrefixes(
      CohortGames games, Map<Long, List<float[]>> prefixes, Map<Long, int[]> finals) {
    int compared = 0;
    for (RecordedGame game : games.games()) {
      List<float[]> expected = prefixes.getOrDefault(game.gameId(), List.of());
      Assert.assertEquals(game.continuingHandEnds(), expected.size(), "game " + game.gameId());
      Assert.assertEquals(game.finalScores(), finals.get(game.gameId()));
      for (int k = 0; k < expected.size(); k++) {
        float[] actual = game.prefix(k);
        Assert.assertEquals(actual.length, expected.get(k).length);
        for (int i = 0; i < actual.length; i++)
          Assert.assertEquals(
              Float.floatToRawIntBits(actual[i]),
              Float.floatToRawIntBits(expected.get(k)[i]),
              "game " + game.gameId() + " hand end " + k + " element " + i);
        compared++;
      }
    }
    Assert.assertTrue(compared > GAMES, "too few continuing hand ends: " + compared);
  }

  /** 収集の数のうち記録が突き合わせる欄だけを入れたもの。 */
  private static CollectionMetrics.Snapshot counts(long games, long hands) {
    return new CollectionMetrics.Snapshot(
        0, 0, hands, games, 0, 0, new long[1], new long[1], new long[1], new long[1], 0, 0, 0, 0, 0, 0, 0, 0, 0);
  }

  /** 記録に流し、同時に収集が渡す次局 root の公開系列（RootState.publicPrefix）と最終の点数を控える観察の口。 */
  static final class Tapped implements CollectionDriver.GameObserver {
    final GameLogObserver log;
    final Map<Long, List<float[]>> prefixes = new ConcurrentHashMap<>();
    final Map<Long, int[]> finals = new ConcurrentHashMap<>();

    Tapped(GameLogObserver log) {
      this.log = log;
    }

    @Override
    public void gameEnded(long gameId, int[] finalScores) {
      finals.put(gameId, finalScores.clone());
      log.gameEnded(gameId, finalScores);
    }

    @Override
    public void handProgressed(long gameId, int kyokuIndex, int honba, int kyotakuCount, int[] scores) {
      log.handProgressed(gameId, kyokuIndex, honba, kyotakuCount, scores);
    }

    @Override
    public void handValued(long gameId, float[] nextRootPrefix, int[] scores) {
      if (nextRootPrefix == null) return;
      prefixes.compute(
          gameId,
          (id, list) -> {
            List<float[]> values = list == null ? new ArrayList<>() : list;
            values.add(nextRootPrefix.clone());
            return values;
          });
    }
  }

  /**
   * s を手の ID と席のハッシュに置き、σ をその argmax にする inline 方策（手の選びは決定的で、打ち筋は手ごとにばらける）。retain なら p0 も
   * 書く（収集の試験の {@code CollectionDriverTest.HashedScorePolicy} と同じ）。
   */
  static final class HashedScorePolicy implements PolicyService {
    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(GameState state, EngineDecisionPoint point, PolicyRequest request) {
      int n = request.legal().size();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        long bits = SeedMixer.mix64(request.legal().get(a).toIndex() * 31L + request.seat());
        scores[a] = (float) CrnSeeds.unit(bits);
        if (scores[a] > scores[best]) best = a;
      }
      float[] sigma = new float[n];
      sigma[best] = 1f;
      request.scores = scores;
      request.sigma = sigma;
      if (request.retain()) {
        float[] prior = new float[n];
        double sum = 0;
        for (int a = 0; a < n; a++) {
          long bits = SeedMixer.mix64(request.legal().get(a).toIndex() * 17L + 5);
          prior[a] = 1f + (float) CrnSeeds.unit(bits);
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }
  }
}
