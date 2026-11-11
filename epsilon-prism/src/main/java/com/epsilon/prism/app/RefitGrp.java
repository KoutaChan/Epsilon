package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Status;
import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.grp.PhiControl;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Path;
import java.util.List;

/**
 * GRP を手動で学び直す要求を run の control に置く（{@link PhiControl#requestRefit}）。学習の JVM で GRP を裏で学ぶ処理が、次に動いたときに
 * 1 回だけ従う。学び終える前なら control cancel で取り消せる（学習の途中なら、学習を止めて試行を捨てる）。Φ を自分の GRP から作る run
 * （SELF_GRP）だけで使える。
 *
 * @param run run のディレクトリ
 * @param slot 学び直す時点（無ければ、半荘の記録がある最新の確定済みのコホート）
 */
@Named(value = "grp-refit", doc = "Put a manual GRP refit request into the run's control")
public record RefitGrp(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Point to refit at (default: the latest sealed cohort with game records)", presence = Presence.ABSENT) @Range(min = 1)
        Integer slot)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    PhiSettings phi = PrismApp.latest(run).phi();
    if (!(phi instanceof PhiSettings.SelfGrp self)) {
      throw new SettingsException(List.of("The run " + run + " has the Φ source " + phi.source() + ", not SELF_GRP"));
    }
    ControlRequest written = PhiControl.requestRefit(run, self.grp(), slot);
    return Outcome.ok("request " + written.number() + ": " + Status.describe(written.request()));
  }
}
