package com.epsilon.cfr.collection;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;

/** 少数の対局を比較するテストだけで、逐次公開される教師経路を全件保持する。 */
final class CfrCollectorTestSupport {
  private CfrCollectorTestSupport() {}

  static CfrCollector.Config configuration(
      CfrCollector.Mode mode,
      double epsilon,
      int batch,
      int concurrent,
      long seed,
      boolean center,
      boolean retain) {
    return new CfrCollector.Config(
        mode,
        epsilon,
        batch,
        concurrent,
        seed,
        center,
        retain,
        batch,
        new DecisionDuelArenaSettings(
            Math.max(
                1, Math.min(concurrent, Math.min(32, Runtime.getRuntime().availableProcessors()))),
            2,
            64),
        new InferenceBatchingSettings(0),
        EpsilonUtilityProfile.TENHOU);
  }

  record Result(
      List<CfrEpisode> episodes,
      List<RootState> observedRoots,
      CfrCollector.CollectionStats stats) {
    static Result from(
        List<CfrCollector.CollectedEpisode> frames, CfrCollector.CollectionStats stats) {
      return new Result(
          frames.stream()
              .map(CfrCollector.CollectedEpisode::episode)
              .sorted(Comparator.comparingLong(CfrEpisode::episodeId))
              .toList(),
          frames.stream().flatMap(frame -> frame.observedRoots().stream()).toList(),
          stats);
    }
  }

  static final class Collector {
    private final CfrCollector collector;

    Collector(CfrCollector.Config config) {
      this(config, new DecisionBranchComparisonSettings(false, false, false, 1, 8, .01));
    }

    Collector(CfrCollector.Config config, DecisionBranchComparisonSettings branches) {
      collector = new CfrCollector(config, branches);
    }

    Result collect(List<Long> ids, CfrPolicy policy, BoundaryValue boundary) {
      var frames = new ArrayList<CfrCollector.CollectedEpisode>();
      var stats = collect(ids, policy, boundary, frames::add);
      return Result.from(frames, stats);
    }

    CfrCollector.CollectionStats collect(
        List<Long> ids,
        CfrPolicy policy,
        BoundaryValue boundary,
        Consumer<CfrCollector.CollectedEpisode> sink) {
      return collector.collect(ids, policy, boundary, CfrEpisodeSink.consuming(sink));
    }

    CfrCollector.CollectionStats collect(
        List<Long> ids, CfrPolicy policy, BoundaryValue boundary, CfrEpisodeSink sink) {
      return collector.collect(ids, policy, boundary, sink);
    }
  }
}
