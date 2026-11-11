package com.epsilon.classic.campaign;

import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import java.io.IOException;
import java.nio.file.Path;

/**
 * campaign の自己対局。macro ごとに、反復に固定した設定の行（半荘の数・探索・相手の選び方・推論の計算の型）で {@link Collect} を作って
 * 打つ。推論のランタイムと卓のコアは campaign のもので、閉じるのは campaign。
 */
final class SelfPlay implements Collector {

  private final ClassicSettings machine;
  private final NetworkRuntime runtime;
  private final TableLoop table;
  private final GrpNetwork grp;

  private SelfPlay(ClassicSettings machine, NetworkRuntime runtime, TableLoop table, GrpNetwork grp) {
    this.machine = machine;
    this.runtime = runtime;
    this.table = table;
    this.grp = grp;
  }

  /**
   * 自己対局を開く（{@link Collector.Factory}）。設定に GRP（{@code run.grp}）があれば、JVM の間に 1 回だけ読む。
   *
   * @throws IllegalArgumentException GRP のチェックポイントがネットワークの GRP でないとき
   */
  static Collector open(ClassicSettings machine, NetworkRuntime runtime, TableLoop table) throws IOException {
    Path path = machine.run().grp();
    GrpNetwork grp = null;
    if (path != null) {
      GrpModel model = GrpCheckpoint.read(path);
      if (!(model instanceof GrpNetwork network)) {
        throw new IllegalArgumentException(
            "The self-play GRP run.grp " + path + " must be a network GRP, not " + model.getClass().getSimpleName());
      }
      grp = network;
    }
    return new SelfPlay(machine, runtime, table, grp);
  }

  @Override
  public Collected collect(Macro macro) throws IOException {
    ClassicSettings settings = macro.settings();
    Collect collect =
        new Collect(
            settings.run().variant(),
            macro.seed(),
            new Collect.Settings(settings.schedule().games(), settings.selfPlay().championProbability()),
            settings.explore(),
            grp,
            options(machine, settings),
            runtime);
    Collect.Result result =
        collect.macro(
            table, macro.number(), macro.learner(), new OpponentPool(macro.champion(), macro.pool()), macro.calibration());
    return new Collected(result.trajectories(), result.calibration(), result.opponent());
  }

  /** 閉じるものを持たない（ランタイムと卓のコアは campaign が閉じる）。 */
  @Override
  public void close() {}

  /**
   * 自己対局と決闘の打ち手を開く設定。デバイスは、起動の行が推論に並べた最初のデバイス。計算の型は反復に固定した行から、1 回の推論の行の
   * 上限と推論の道は起動の行から読む。
   *
   * @param machine 起動の行の設定
   * @param settings 反復に固定した設定の行
   */
  static PolicyOptions options(ClassicSettings machine, ClassicSettings settings) {
    InferenceSettings inference = machine.inference();
    return new PolicyOptions(
        inference.devices().getFirst(), settings.inference().precision(), inference.maxBatch(), inference.kernels());
  }
}
