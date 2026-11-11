package com.epsilon.agent.app;

import com.epsilon.agent.run.RunDirectory;
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
 * 設定と、設定が参照するファイルを確かめる。問題があれば 1 行ずつログに書き、{@link Exit#PROBLEMS} で終わる。
 *
 * @param run 最新の設定の行を確かめる run のディレクトリ。指定しなければ、重ねた層から作った設定を確かめる
 */
@Named(value = "settings check", doc = "Check the settings and the files they refer to (exit code 4 on a problem)")
public record SettingsCheck(
    @Key(doc = "Run directory whose latest settings row to check", presence = Presence.ABSENT) Path run) implements BuiltinCommand {

  private static final Logger log = LoggerFactory.getLogger(SettingsCheck.class);

  @Override
  public Outcome run(Context<Record> context) throws Exception {
    Application<Record> application = context.application();
    if (run != null) {
      context.requireNoLayers();
    }
    Settings<Record> settings;
    try {
      settings =
          run == null
              ? context.settings()
              : new SettingsRows(new RunDirectory(run)).latest().bind(application.schema());
    } catch (SettingsException invalid) {
      return problems(invalid.problems());
    }
    List<String> problems = application.check().problems(settings.value());
    if (!problems.isEmpty()) {
      return problems(problems);
    }
    return Outcome.ok("no problems in " + settings.effective().size() + " settings");
  }

  static Outcome problems(List<String> problems) {
    problems.forEach(log::error);
    return new Outcome(Exit.PROBLEMS, problems.size() + " problem(s)");
  }
}
