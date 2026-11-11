package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Reads;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Path;

/**
 * 止めた run で待っている評価と決闘のジョブだけを、系譜と同じ処理で回す（{@link PrismRun}）。学習のロックを取るので、学習と同時には
 * 走らない。SIGTERM・SIGINT や control からの停止の要求を受けると、次のセーフポイントで止まる。
 *
 * @param run run のディレクトリ
 */
@Named(value = "jobs run", doc = "Run the pending jobs of a stopped run", reads = Reads.RUN)
public record JobsRun(@Key(doc = "Run directory", presence = Presence.REQUIRED) Path run) implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    Ending ending = PrismRun.runner(run, null, context.signals()::requested).runJobs();
    return ending == Ending.FINISHED ? Outcome.ok("ran the waiting jobs") : PrismRun.outcome(ending);
  }
}
