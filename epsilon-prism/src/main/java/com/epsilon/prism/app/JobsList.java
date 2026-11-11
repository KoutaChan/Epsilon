package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.settings.PrismSettings;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Locale;
import java.util.Map;

/**
 * run のジョブを、回す順に 1 行ずつ並べる。行には名前、状態、始めた回数、落ちた回数、打ち終えた牌山の数、最後に落ちた理由を出す。
 * job.json を読めないジョブは最後に並べる。
 *
 * @param run run のディレクトリ
 */
@Named(value = "jobs list", doc = "List the run's jobs and their states")
public record JobsList(@Key(doc = "Run directory", presence = Presence.REQUIRED) Path run) implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    Jobs.Listing listing = new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS)).list();
    PrintStream out = context.out();
    Map<JobFile.State, Integer> counts = new EnumMap<>(JobFile.State.class);
    for (JobFile job : listing.jobs()) {
      counts.merge(job.state(), 1, Integer::sum);
      out.println(
          String.format(
              Locale.ROOT,
              "%-24s %-8s attempts=%d failures=%d walls=%d%s",
              job.name(),
              job.state(),
              job.attempts(),
              job.failures(),
              job.walls(),
              job.reason() == null ? "" : " reason=" + job.reason()));
    }
    for (String name : listing.unreadable()) {
      out.println(String.format(Locale.ROOT, "%-24s UNREADABLE", name));
    }
    return Outcome.ok(
        "jobs=" + listing.jobs().size() + " " + counts + " unreadable=" + listing.unreadable().size());
  }
}
