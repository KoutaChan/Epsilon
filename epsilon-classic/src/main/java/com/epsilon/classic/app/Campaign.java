package com.epsilon.classic.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Reads;
import com.epsilon.agent.app.Signals;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.classic.campaign.CampaignKind;
import java.nio.file.Path;

/**
 * campaign の macro を、止めた所から回す（{@link CampaignKind}）。設定は run の設定の行から読む。SIGTERM・SIGINT と control の要求で、次の
 * セーフポイントで止まる。
 *
 * @param run run のディレクトリ
 * @param stopAfter この呼び出しで閉じる macro の数の上限（無ければ終わりまで）
 */
@Named(value = "campaign", doc = "Run the campaign macros", reads = Reads.RUN)
public record Campaign(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Maximum number of macros to close in this call", presence = Presence.ABSENT) @Range(min = 1) Integer stopAfter)
    implements ClassicCommand {

  @Override
  public Outcome run(Context<ClassicSettings> context) throws Exception {
    Signals signals = context.signals();
    return CampaignKind.outcome(CampaignKind.STANDARD.runner(run, stopAfter, signals::requested).train());
  }
}
