package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * FAILED のジョブの落ちた回数を 0 に戻し、止めた所から続きを待つ状態（PENDING）にする（{@link Jobs#retry}）。学習の JVM は、次にジョブを
 * 回すときに続きから回す。
 *
 * @param run run のディレクトリ
 * @param name ジョブの名前（{@code v00824-x1n3}）
 */
@Named(value = "jobs retry", doc = "Set a FAILED job back to resume where it stopped")
public record JobsRetry(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Job name (v00824-x1n3)", presence = Presence.REQUIRED) String name)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    if (!Files.exists(directory.job(name).resolve(Jobs.JOB))) {
      throw new SettingsException(List.of("The run " + run + " has no job named " + name));
    }
    try {
      JobFile retried = new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS)).retry(name);
      return Outcome.ok(retried.name() + " is " + retried.state() + " from wall " + retried.walls());
    } catch (IllegalArgumentException notFailed) {
      throw new SettingsException(List.of(notFailed.getMessage()));
    }
  }
}
