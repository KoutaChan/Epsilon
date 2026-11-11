package com.epsilon.classic.campaign;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.JobSpec;
import com.epsilon.agent.run.Launch;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunKind;
import com.epsilon.agent.run.RunSession;
import com.epsilon.agent.run.Runner;
import com.epsilon.classic.app.ClassicSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * classic の campaign の run の種類: 段は {@link CampaignStep}、state は {@link CampaignState}、反復の報告の持ち越しは
 * {@link CampaignCarry}。macro t はバージョン t で自己対局を打ち、そこから学んでバージョン t + 1 を公開する。
 *
 * @param collectors 自己対局の開き方
 */
public record CampaignKind(Collector.Factory collectors)
    implements RunKind<ClassicSettings, CampaignState, CampaignIteration, CampaignStep> {

  /** この jar の campaign（自己対局は {@link SelfPlay}）。 */
  public static final CampaignKind STANDARD = new CampaignKind(SelfPlay::open);

  public CampaignKind {
    Objects.requireNonNull(collectors, "collectors");
  }

  @Override
  public Class<ClassicSettings> settings() {
    return ClassicSettings.class;
  }

  @Override
  public Class<CampaignState> state() {
    return CampaignState.class;
  }

  @Override
  public Class<CampaignStep> steps() {
    return CampaignStep.class;
  }

  @Override
  public Class<CampaignCarry> carry() {
    return CampaignCarry.class;
  }

  @Override
  public int iterations(ClassicSettings settings) {
    return settings.schedule().macros();
  }

  /** campaign はジョブを作らない（評価の卓は決闘の段が打つ）。 */
  @Override
  public List<JobSpec> jobs(int version, ClassicSettings settings) {
    return List.of();
  }

  @Override
  public RunSession<ClassicSettings, CampaignState, CampaignIteration> open(Launch<ClassicSettings> launch)
      throws IOException {
    return CampaignSession.open(launch, collectors);
  }

  /**
   * run を回す仕組み。GPU の使用権の印があるときだけ、GPU の要求に譲る。
   *
   * @param run run のディレクトリ
   * @param stopAfter この呼び出しで閉じる macro の数の上限（無ければ null）
   * @param signalled SIGTERM か SIGINT が来たか
   */
  public Runner<ClassicSettings, CampaignState, CampaignIteration, CampaignStep> runner(
      Path run, Integer stopAfter, BooleanSupplier signalled) throws IOException {
    return new Runner<>(new RunDirectory(run), this, Runner.Options.of(Jars.running(), stopAfter, signalled));
  }

  /** run を回す呼び出しの終わり方を、コマンドの終わり方に直す。 */
  public static Outcome outcome(Ending ending) {
    return switch (ending) {
      case FINISHED -> Outcome.ok("finished the macros");
      case LIMIT -> Outcome.ok("closed the macros this call allows");
      case STOPPED -> new Outcome(Exit.STOPPED, "stopped at a safe point");
      case RESTART -> new Outcome(Exit.RESTART, "stopped at a safe point to restart");
      case YIELD -> new Outcome(Exit.YIELD, "stopped at a safe point to yield the GPUs");
      case REFUSED -> new Outcome(Exit.USAGE, "the campaign cannot continue (see status.json)");
    };
  }
}
