package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.grp.ModeLine;
import com.epsilon.prism.grp.PhiControl;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.grp.PhiMode;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * Φ を手動で切り替えるか、前に戻す。行き先がモードの別名（late・grp）なら、Φ のモードを切り替える（{@link PhiControl#switchMode}）。
 * これは Φ を自分の GRP か固定の GRP から作る run で使える。ほかの行き先は GRP の項目の id とみて、その項目に切り替えるか、前の項目に
 * 戻す（{@link PhiControl#switchTo}）。これは Φ を自分の GRP から作る run（SELF_GRP）だけで使える。
 *
 * @param run run のディレクトリ
 * @param to 切り替える先（READY の候補の試行の id、既にある項目の id、モードの別名 late・grp）
 * @param from 効かせ始める反復（無ければ、始まった最後の反復の次）
 * @param cutoff LATE の打ち切りの値（行き先が late のときだけ使う。無ければ設定の phi.late.cutoff）
 * @param reason 記録に書く理由
 */
@Named(value = "grp-switch", doc = "Switch the Φ entry or mode by hand, or go back to the previous entry")
public record SwitchGrp(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Target (id of a READY candidate trial, id of an existing entry, or a mode alias: late, grp)", presence = Presence.REQUIRED)
        String to,
    @Key(doc = "Iteration to take effect from (default: the one after the last started iteration)", presence = Presence.ABSENT) @Range(min = 1) Integer from,
    @Key(doc = "LATE cutoff (only for target late; default: phi.late.cutoff in the settings)", presence = Presence.ABSENT)
        @Range(min = 0, max = 0.01) Double cutoff,
    @Key(doc = "Reason written to the record", value = "grp-switch") String reason)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    PrismSettings settings = PrismApp.latest(run);
    PhiSettings phi = settings.phi();
    Optional<PhiMode> mode = PhiMode.byAlias(to);
    if (cutoff != null && mode.orElse(null) != PhiMode.LATE) {
      throw new SettingsException(List.of("--cutoff applies to --to late only, not to " + to));
    }
    if (mode.isPresent()) {
      if (phi.lateSettings().isEmpty()) {
        throw new SettingsException(
            List.of("The run " + run + " has the Φ source " + phi.source() + ", which has no Φ modes"));
      }
      var setup = new PhiSetup(phi, settings.run().variant(), settings.run().seed());
      Optional<ModeLine> appended = PhiControl.switchMode(run, setup, mode.get(), cutoff, from, reason);
      return Outcome.ok(
          appended
              .map(line -> "appended " + line.mode() + " from iteration " + line.effectiveFrom())
              .orElse(to + " is already in effect"));
    }
    if (!(phi instanceof PhiSettings.SelfGrp)) {
      throw new SettingsException(List.of("The run " + run + " has the Φ source " + phi.source() + ", not SELF_GRP"));
    }
    Optional<PhiEntry> appended = PhiControl.switchTo(run, to, from, reason);
    return Outcome.ok(appended.map(entry -> "appended " + entry.id()).orElse(to + " is already in effect"));
  }
}
