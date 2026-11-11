package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Reads;
import com.epsilon.agent.app.Signals;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Path;

/**
 * run の反復を、止めた所から回す（{@link PrismRun}）。設定は run の設定の行から読む。SIGTERM・SIGINT や control からの停止の要求を
 * 受けると、次のセーフポイントで止まる。
 *
 * @param run run のディレクトリ
 * @param stopAfter この呼び出しで終える反復の数の上限（無ければ最後まで回す）
 */
@Named(value = "train", doc = "Run the run's iterations", reads = Reads.RUN)
public record Train(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Cap on the iterations to finish in this call", presence = Presence.ABSENT) @Range(min = 1) Integer stopAfter)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    Signals signals = context.signals();
    return PrismRun.outcome(PrismRun.runner(run, stopAfter, signals::requested).train());
  }
}
