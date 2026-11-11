package com.epsilon.agent.run;

import java.io.IOException;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1 つの反復の段を、宣言した順に回す。各段の前と後がセーフポイント。完了マーカーのある段は回さずに、その報告を使う。段が落ちたら、
 * その段の {@code .part} は、次に回すときに段が自分で消す。
 *
 * @param <K> 段の宣言
 * @param <I> 反復
 */
public final class Pipeline<K extends Enum<K> & StepKind<I>, I extends Iteration> {

  private static final Logger LOG = LoggerFactory.getLogger(Pipeline.class);

  private final Class<K> kinds;
  private final List<K> steps;
  private final RunControl control;

  public Pipeline(Class<K> steps, RunControl control) {
    this.kinds = steps;
    this.steps = List.of(steps.getEnumConstants());
    this.control = control;
  }

  /**
   * 段を回す。
   *
   * @return 段ごとの報告（宣言した順）。セーフポイントで止まると決めたときは空（理由は {@link RunControl#halt}）
   */
  public Optional<Map<K, StepReport>> run(I iteration) throws Exception {
    Map<K, StepReport> reports = new EnumMap<>(kinds);
    for (K kind : steps) {
      if (control.check(SafePoint.beforeStep(kind.name())).isPresent()) {
        return Optional.empty();
      }
      StepReport report;
      control.step(kind.name());
      try {
        report = report(kind, iteration);
      } catch (Halted halted) {
        return Optional.empty();
      } finally {
        control.step(null);
      }
      reports.put(kind, report);
      LOG.info(
          "t={} {} {} {}s", iteration.number(), kind.name(), report.status(), String.format("%.1f", report.seconds()));
      if (control.check(SafePoint.afterStep(kind.name())).isPresent()) {
        return Optional.empty();
      }
    }
    return Optional.of(Collections.unmodifiableMap(reports));
  }

  /**
   * 固定した jar でしか続けられない段（{@link StepKind#jarBound}）のうち、有効で完了マーカーのある最初の段を探す。
   *
   * @return 見つかった段。どれも済んでいなければ空（反復を新しい jar で固定し直してよい）
   */
  public Optional<K> jarBoundDone(I iteration) throws IOException {
    for (K kind : steps) {
      if (kind.jarBound()) {
        Step<I> step = kind.create(iteration);
        if (step.enabled(iteration) && step.done(iteration).isPresent()) {
          return Optional.of(kind);
        }
      }
    }
    return Optional.empty();
  }

  private StepReport report(K kind, I iteration) throws Exception {
    Step<I> step = kind.create(iteration);
    if (!step.enabled(iteration)) {
      return StepReport.skipped(iteration);
    }
    Optional<StepReport> done = step.done(iteration);
    StepReport report = done.isPresent() ? done.get().reused() : step.run(iteration);
    if (report.status() == StepReport.Status.SKIPPED) {
      throw new IllegalStateException("The enabled step " + kind.name() + " reported SKIPPED");
    }
    if (report.metrics() != null && !kind.metrics().isInstance(report.metrics())) {
      throw new IllegalStateException(
          "The step " + kind.name() + " reported " + report.metrics().getClass().getName() + ", not "
              + kind.metrics().getName());
    }
    return report;
  }
}
