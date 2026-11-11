package com.epsilon.prism.run;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.JobSpec;
import com.epsilon.agent.run.Launch;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunKind;
import com.epsilon.agent.run.RunSession;
import com.epsilon.agent.run.Runner;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.BooleanSupplier;

/**
 * PRISM の系譜の run の種類。段は {@link PrismStep}、state は {@link PrismState}、反復の報告で持ち越す値は {@link PrismCarry}。
 * 反復 t はバージョン t（王者へ戻した後は王者のバージョン）で集め、バージョン t + 1 を公開する。
 */
public enum PrismRun implements RunKind<PrismSettings, PrismState, PrismIteration, PrismStep> {
  LINEAGE;

  @Override
  public Class<PrismSettings> settings() {
    return PrismSettings.class;
  }

  @Override
  public Class<PrismState> state() {
    return PrismState.class;
  }

  @Override
  public Class<PrismStep> steps() {
    return PrismStep.class;
  }

  @Override
  public Class<PrismCarry> carry() {
    return PrismCarry.class;
  }

  @Override
  public int iterations(PrismSettings settings) {
    return settings.schedule().iterations();
  }

  /**
   * バージョンを予定どおりに測る卓のジョブを返す（卓の名前の順）。基準の卓の結果でラダーを判定する回は、ラダーを打たないので、ラダーの
   * ジョブを作らない（{@link EvaluationSettings#scheduled}）。卓の定義は卓の値の木。
   */
  @Override
  public List<JobSpec> jobs(int version, PrismSettings settings) {
    List<JobSpec> specs = new ArrayList<>();
    for (Map.Entry<String, TableSpec> entry : settings.evaluation().scheduled(version).entrySet()) {
      specs.add(new JobSpec(entry.getKey(), RecordFile.tree(entry.getValue())));
    }
    return specs;
  }

  @Override
  public RunSession<PrismSettings, PrismState, PrismIteration> open(Launch<PrismSettings> launch) throws IOException {
    return PrismSession.open(launch);
  }

  /**
   * run を回す {@link Runner} を作る。GPU の使用権を示す環境変数があるときだけ、ほかのプロセスの GPU の要求に譲る。
   *
   * @param run run のディレクトリ
   * @param stopAfter この呼び出しで閉じる反復の数の上限（無ければ null）
   * @param signalled 停止の要求が来たかを返す
   */
  public static Runner<PrismSettings, PrismState, PrismIteration, PrismStep> runner(
      Path run, Integer stopAfter, BooleanSupplier signalled) throws IOException {
    return new Runner<>(new RunDirectory(run), LINEAGE, Runner.Options.of(Jars.running(), stopAfter, signalled));
  }

  /** run を回した呼び出しの終わり方を、コマンドの終わり方に直す。 */
  public static Outcome outcome(Ending ending) {
    return switch (ending) {
      case FINISHED -> Outcome.ok("finished the iterations and their jobs");
      case LIMIT -> Outcome.ok("closed the iterations this call allows");
      case STOPPED -> new Outcome(Exit.STOPPED, "stopped at a safe point");
      case RESTART -> new Outcome(Exit.RESTART, "stopped at a safe point to restart");
      case YIELD -> new Outcome(Exit.YIELD, "stopped at a safe point to yield the GPUs");
      case REFUSED -> new Outcome(Exit.USAGE, "the run cannot continue (see status.json)");
    };
  }
}
