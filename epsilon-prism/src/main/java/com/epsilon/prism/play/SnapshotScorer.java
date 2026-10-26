package com.epsilon.prism.play;

import ai.djl.Device;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.policy.NetPolicyService;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * 公開した版（snapshots/&lt;t&gt;）を開き、1 判断ずつ同期で s（合法手ごとの網の出力、効用の単位）を求める（牌譜解析）。一様の KL の基準なら
 * s = 手の価値 + 判断ごとの定数なので、同じ判断の手どうしの差が価値の差になる。
 *
 * <p>バッチの上限と p0 の係数は設定の [epsilon.prism]（evaluate と同じ）、device は epsilon.devices.inference から読む。実行文脈は版ごとに持ち、
 * {@link #close} で網と一緒に閉じる。
 */
public final class SnapshotScorer implements AutoCloseable {
  private final PolicyExecutionContext execution;
  private final NetPolicyService service;

  private SnapshotScorer(PolicyExecutionContext execution, NetPolicyService service) {
    this.execution = execution;
    this.service = service;
  }

  /** 版を開く。 */
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
              config.multiTransitionBatch(),
              config.priorSettings()));
    } catch (IOException | RuntimeException | Error failure) {
      execution.close();
      throw failure;
    }
  }

  /**
   * 判断の問い合わせ（行の payload は作らない）。種別・類は収集と同じ分類で、エンジンの判断の種別の代わりに観測から鳴いた直後の打牌を見分ける
   * （{@link Kind#classify(PublicObservation, List)}）。
   */
  public static PolicyRequest request(PublicObservation observation, List<Action> legal) {
    Kind kind = Kind.classify(observation, legal);
    return new PolicyRequest(
        null,
        0,
        0,
        observation.observer(),
        observation,
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
  public float[] scores(PublicObservation observation, List<Action> legal) {
    if (legal.size() == 1) return new float[] {0};
    PolicyRequest request = request(observation, legal);
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
