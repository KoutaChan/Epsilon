package com.epsilon.prism.play;

import ai.djl.Device;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * 公開した版（snapshots/&lt;t&gt;）を開き、1 判断ずつ同期で s（合法手ごとの網の出力、効用の単位）を求める（牌譜解析）。差分整合の KL の基準は
 * C の上の一様分布なので s = 手の価値 + 判断ごとの定数で、同じ判断の手どうしの差が価値の差になる。
 *
 * <p>バッチの上限と p0 の係数は設定の [epsilon.prism]（evaluate と同じ）、device は epsilon.devices.inference から読む。実行文脈は版ごとに持ち、
 * {@link #close} で網と一緒に閉じる（{@link Snapshot#open}。CPU は FP32 の eager）。
 *
 * <p>網の方策は記録 O だけを読むので、呼んだスレッドで O を捕捉して渡す（収集と同じ {@link PolicyRequest#capture}）。書き手と置き場はこの版の
 * 持ち物で、1 判断ずつ使い直す（{@link #scores} は同時に 1 つだけ走る）。
 */
public final class SnapshotScorer implements AutoCloseable {
  /** 置き場の初めの {@code long} の数（判断 1 つの O は 20〜40 語）。 */
  private static final int ARENA_LONGS = 64;

  private final PolicyExecutionContext execution;
  private final PolicyService service;
  private final ObservationWriter writer = new ObservationWriter();
  private final ObservationArena arena = new ObservationArena(ARENA_LONGS);

  private SnapshotScorer(PolicyExecutionContext execution, PolicyService service) {
    this.execution = execution;
    this.service = service;
  }

  public static SnapshotScorer open(Path snapshot, SettingsLoader settings) throws IOException {
    PrismConfig config = PrismConfig.fromSettings(settings);
    config.validate();
    List<Device> devices =
        NetworkDeviceResolver.inference(settings.bind(DeviceSettings.class)).asList();
    var execution = new PolicyExecutionContext(1, 1, 1);
    try {
      return new SnapshotScorer(
          execution,
          Snapshot.open(
              snapshot,
              devices,
              execution,
              config.inferBatch(),
              config.priorSettings(),
              config.inference()));
    } catch (IOException | RuntimeException | Error failure) {
      execution.close();
      throw failure;
    }
  }

  /**
   * 判断の問い合わせ（行は作らない）。判断の種類はエンジンの判断点の代わりに観測と合法手から決め（{@link
   * EngineDecisionKind#of}。live の判断点と同じ値）、種別・類は収集と同じ分類にする。
   */
  public static PolicyRequest request(PublicObservation observation, List<Action> legal) {
    EngineDecisionKind decisionKind = EngineDecisionKind.of(observation, legal);
    Kind kind = Kind.classify(decisionKind, legal);
    return new PolicyRequest(
        null,
        0,
        0,
        observation.observer(),
        observation,
        decisionKind,
        legal,
        kind,
        DecisionClass.of(
            kind, Stage.of(observation.remainingWallTiles()), Threats.count(observation)),
        false);
  }

  /**
   * 1 判断の s（合法手の順、所有権ごと返す）を同期で求める。合法手が 1 つなら網を呼ばずに {0} を返す。GPU で受付が満杯なら空くのを待つ。観測と
   * 合法手は呼出中だけ借用する。
   */
  public synchronized float[] scores(PublicObservation observation, List<Action> legal) {
    if (legal.size() == 1) return new float[] {0};
    PolicyRequest request = request(observation, legal);
    arena.clear();
    request.capture(writer, arena);
    while (true) {
      var attempt = service.tryAcquireIngress();
      if (attempt.acquired()) {
        service.submit(List.of(request), attempt.ingress()).join();
        return request.scores;
      }
      attempt.available().join();
    }
  }

  @Override
  public void close() {
    try {
      service.close();
    } finally {
      execution.close();
    }
  }
}
