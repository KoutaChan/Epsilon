package com.epsilon.agent.app;

import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.nio.file.Path;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 層を重ねた設定を確かめ、run のディレクトリと設定の 1 行目を作る。1 行目は反復 1 から効く。
 *
 * @param run 作る run のディレクトリ。設定の行が既にあれば誤りにする
 */
@Named(value = "settings create", doc = "Layer and check the settings, then create the run directory and its first settings row")
public record SettingsCreate(@Key(doc = "Run directory to create", presence = Presence.REQUIRED) Path run)
    implements BuiltinCommand {

  private static final Logger log = LoggerFactory.getLogger(SettingsCreate.class);

  @Override
  public Outcome run(Context<Record> context) throws Exception {
    Settings<Record> settings = context.settings();
    SettingsRows rows = new SettingsRows(new RunDirectory(run));
    try (SettingsRows.Locked locked = rows.lock()) {
      if (!locked.rows().isEmpty()) {
        throw new SettingsException(
            List.of("The run " + run + " already has settings rows; change them with settings change"));
      }
      locked.append(1, Jars.running(), SettingsRow.CREATED, List.of(), settings.tree());
    }
    log.info("Created the run {} with settings row 1", run);
    return Outcome.ok("row 1, " + settings.effective().size() + " settings");
  }
}
