package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.luck.LuckSampleStore;
import com.epsilon.prism.luck.LuckSampleWriter;
import com.epsilon.prism.luck.LuckSampling;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 運の網の標本は標本を足すだけ: 主経路・写し・分岐・主経路の探索の収集を、標本なしとありで打つと、行が 1 バイトも違わない（標本は状態を読むだけで、
 * 乱数も進行も変えない）。率 1 では全ての出来事が標本になり、硬貨を引いた出来事・捕捉した標本・局末の値を付けて書いた標本・封印して
 * {@link LuckSampleStore} で読み戻した標本が同じ数になる（主経路で枠のない局の標本も、局末の値を求めて書く）。
 */
public class RecordLuckTest {
  private static final Logger log = LoggerFactory.getLogger(RecordLuckTest.class);

  private static final long SEED = 7_001L;
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  /** 1 つの収集の構成（半荘 first..first+3）。 */
  private record Collection(Sampling sampling, long iterationSeed, long first, Behaviour behaviour) {}

  /**
   * 1 つの収集の結果。
   *
   * @param rows 行の符号の正準
   * @param luck 標本の件数（標本を集めない収集は null）
   * @param stored 封印して読み戻した標本の数
   */
  private record Collected(List<String> rows, LuckSampleSink.Counts luck, long stored) {}

  @Test(timeOut = 900_000)
  public void luckSamplesLeaveTheRowsAlone() throws IOException {
    var sampling =
        new Sampling(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2)
            .withEventLedger(true);
    List<Collection> collections =
        List.of(
            new Collection(sampling, SEED * 3, 100, new Behaviour(0.05, 0, CrnScheme.V1)),
            new Collection(sampling, SEED * 7, 300, new Behaviour(0.05, 0, CrnScheme.V2)),
            new Collection(sampling, SEED * 3, 100, new Behaviour(0, 0.2, CrnScheme.V1)));
    long events = 0, luckOnlyHands = 0;
    for (int i = 0; i < collections.size(); i++) {
      Collected off = collect(collections.get(i), 0);
      Collected on = collect(collections.get(i), 1.0);
      Assert.assertEquals(on.rows(), off.rows(), "rows of collection " + i);
      Assert.assertFalse(on.rows().isEmpty(), "collection " + i);
      LuckSampleSink.Counts luck = on.luck();
      Assert.assertTrue(luck.events() > 0, "collection " + i);
      Assert.assertEquals(luck.captured(), luck.events(), "collection " + i);
      Assert.assertEquals(luck.samples(), luck.captured(), "collection " + i);
      Assert.assertEquals(on.stored(), luck.samples(), "collection " + i);
      events += luck.events();
      luckOnlyHands += luck.luckOnlyHands();
    }
    Assert.assertTrue(luckOnlyHands > 0, "no hand was valued for its samples alone");
    log.info("luckSamplesLeaveTheRowsAlone: events={} luckOnlyHands={}", events, luckOnlyHands);
  }

  /** 収集を打つ（rate が正なら標本も集めて封印し、読み戻す）。 */
  private static Collected collect(Collection collection, double rate) throws IOException {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < 4; id++) ids.add(collection.first() + id);
    try (var driver = new CollectionDriver(new CollectionDriver.Settings(4, 1, new InferenceBatchingSettings(1000)));
        var policy = new CollectionDriverTest.HashedScorePolicy();
        var phi = HandValue.scoreRank(PROFILE);
        var temporary = new CollectionDriverTest.TemporaryCohort(collection.sampling());
        var writer = rate > 0 ? new LuckSampleWriter(temporary.directory, 1) : null) {
      LuckSampleSink luck =
          writer == null ? null : new LuckSampleSink(new LuckSampling(collection.iterationSeed(), rate), writer);
      var cohort =
          new CollectionDriver.Cohort(
              SEED, collection.iterationSeed(), ids, PROFILE, collection.sampling(), collection.behaviour(), null, null);
      driver.collect(cohort, policy, phi, temporary.sink, luck);
      LuckSampleSink.Counts counts = null;
      long stored = 0;
      if (luck != null) {
        if (luck.failure() != null) throw new AssertionError("luck samples failed", luck.failure());
        counts = luck.counts();
        writer.seal();
        LuckSampleStore store = LuckSampleStore.open(temporary.directory);
        try (LuckSampleStore.Cursor cursor = store.cursor()) {
          while (cursor.next()) stored++;
        }
        Assert.assertEquals(store.samples(), stored);
      }
      return new Collected(CollectionDriverTest.encoded(temporary.sealAndRead()), counts, stored);
    }
  }
}
