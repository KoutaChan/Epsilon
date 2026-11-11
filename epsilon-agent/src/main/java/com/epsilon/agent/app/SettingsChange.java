package com.epsilon.agent.app;

import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Jars;
import com.epsilon.agent.run.JobSpec;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.RunKind;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Difference;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.agent.settings.Toml;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 続いている run の設定を変える。settings.lock のロックを取ったまま、最新の行に変更を重ねる（{@code --replace} なら、
 * 渡したファイルで置き換える）。バインドと、参照するファイルの確かめを通ったら、行を 1 つ足す。
 * run の途中で変えてはいけないキーの変更は拒む。足した行は、その後に始まる最初の反復から効く。run を持つアプリでは、その反復が
 * 公開するバージョンのジョブについて、変更で増える数・減る数・定義の変わる数も出す。
 *
 * @param run 変える run のディレクトリ
 * @param change 「パス=値」の変更。書いた順に重ねる
 * @param apply 重ねる設定ファイル。先頭の presets も重ねる
 * @param replace 最新の行を使わず、渡したファイルだけで全体を作り直す
 * @param reason 変える理由
 * @param restart 速さに関わるキー（MACHINE）を変えたら、反復を閉じた後で起動し直す要求も置く
 */
@Named(value = "settings change", doc = "Add a settings row to a continuing run (takes effect from the next iteration)")
public record SettingsChange(
    @Key(doc = "Run directory to change", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Change written as path=value", value = "[]") List<String> change,
    @Key(doc = "Settings file to layer on top", presence = Presence.ABSENT) Path apply,
    @Key(doc = "Replace the whole settings with the given file instead of the latest row", value = "false") boolean replace,
    @Key(doc = "Reason for the change (kept in the settings row)", presence = Presence.REQUIRED) String reason,
    @Key(doc = "If a speed key (MACHINE) changes, also request a restart after the iteration closes", value = "false") boolean restart)
    implements BuiltinCommand {

  private static final Logger log = LoggerFactory.getLogger(SettingsChange.class);

  public SettingsChange {
    change = List.copyOf(change);
  }

  @Override
  public Outcome run(Context<Record> context) throws Exception {
    List<String> problems = new ArrayList<>();
    if (!context.layers().isEmpty()) {
      problems.add("settings change takes --change and --apply, not --settings, --preset or --set");
    }
    if (change.isEmpty() && apply == null) {
      problems.add("settings change needs --change or --apply");
    }
    if (replace && apply == null) {
      problems.add("--replace needs --apply");
    }
    if (reason.isBlank()) {
      problems.add("--reason must not be empty");
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    Application<Record> application = context.application();
    RunDirectory directory = new RunDirectory(run);
    if (!Files.isRegularFile(directory.settings())) {
      throw new SettingsException(List.of("The run " + run + " has no settings rows: create it with settings create"));
    }
    try (SettingsRows.Locked locked = new SettingsRows(directory).lock()) {
      return append(application, locked);
    }
  }

  private Outcome append(Application<Record> application, SettingsRows.Locked locked) throws Exception {
    SettingsRow last = locked.latest();
    Settings<Record> previous = last.bind(application.schema());
    List<Layer> layers = new ArrayList<>();
    if (!replace) {
      layers.add(last.layer());
    }
    if (apply != null) {
      Layer file = Layer.toml(apply);
      for (String preset : file.presets()) {
        layers.add(application.preset(preset));
      }
      layers.add(file);
    }
    if (!change.isEmpty()) {
      layers.add(Layer.assignments("--change", change));
    }
    Settings<Record> next = Settings.bind(application.schema(), replace ? application.stack(layers) : layers);
    List<Difference> differences = next.since(previous);
    if (differences.isEmpty()) {
      log.info("The change leaves settings row {} as it is; nothing written", last.row());
      return Outcome.ok("no change to row " + last.row());
    }
    Difference.rejectFixed(differences);
    List<String> problems = application.check().problems(next.value());
    if (!problems.isEmpty()) {
      return SettingsCheck.problems(problems);
    }
    SettingsRow row =
        locked.append(
            locked.firstOpenIteration(),
            Jars.running(),
            reason,
            differences.stream().map(SettingsRow.ChangedKey::of).toList(),
            next.tree());
    Set<Change> changes = EnumSet.noneOf(Change.class);
    for (Difference difference : differences) {
      changes.add(difference.change());
      log.info(
          "  {}: {} -> {} ({})",
          difference.key(),
          difference.before() == null ? "(none)" : Toml.inline(difference.before()),
          difference.after() == null ? "(none)" : Toml.inline(difference.after()),
          difference.change());
    }
    log.info("Settings row {} takes effect from iteration {}", row.row(), row.from());
    for (Change kind : changes) {
      log.info("  {}", effect(kind, application.name()));
    }
    if (application.runKind() != null) {
      logJobs(application.runKind(), row, previous.value(), next.value());
    }
    if (restart) {
      if (changes.contains(Change.MACHINE)) {
        ControlRequest request = new ControlRequests(new RunDirectory(run)).submit(new Request.Restart(Request.At.ITERATION));
        log.info("Request {} restarts the run after the iteration closes", request.number());
      } else {
        log.info("No MACHINE key changed, so no restart is requested");
      }
    }
    return Outcome.ok("row " + row.row() + ", " + differences.size() + " key(s) changed");
  }

  /**
   * 行が効く最初の反復が公開するバージョンのジョブを、変更の前と後の設定で作り、卓の名前で比べて出す。両方にある卓は、定義が違えば
   * 変わったジョブに数える。
   *
   * @param before 前の行の設定
   * @param after 足した行の設定
   */
  private static void logJobs(RunKind<Record, ?, ?, ?> kind, SettingsRow row, Record before, Record after) {
    int version = row.from() + 1;
    Map<String, Node.Table> was = definitions(kind.jobs(version, before));
    Map<String, Node.Table> will = definitions(kind.jobs(version, after));
    List<String> added = new ArrayList<>();
    List<String> changed = new ArrayList<>();
    for (Map.Entry<String, Node.Table> job : will.entrySet()) {
      Node.Table definition = was.get(job.getKey());
      if (definition == null) {
        added.add(job.getKey());
      } else if (!definition.equals(job.getValue())) {
        changed.add(job.getKey());
      }
    }
    List<String> removed = was.keySet().stream().filter(table -> !will.containsKey(table)).toList();
    log.info(
        "Version {} is the first published with row {}: {} job(s) instead of {} ({} added {}, {} removed {}, {} changed"
            + " {})",
        version,
        row.row(),
        will.size(),
        was.size(),
        added.size(),
        added,
        removed.size(),
        removed,
        changed.size(),
        changed);
  }

  /** 卓の名前ごとのジョブの定義（ジョブの順）。 */
  private static Map<String, Node.Table> definitions(List<JobSpec> jobs) {
    Map<String, Node.Table> definitions = new LinkedHashMap<>();
    for (JobSpec job : jobs) {
      definitions.put(job.table(), job.definition());
    }
    return definitions;
  }

  private String effect(Change kind, String app) {
    return switch (kind) {
      case COLLECT -> "COLLECT: the next cohort is collected with it";
      case FIT -> "FIT: the next fit uses it";
      case MACHINE ->
          "MACHINE: takes effect after restart: run '" + app + " control restart --run " + run + " --at ITERATION'";
      case SCHEDULE -> "SCHEDULE: the jobs of the next published version use it";
      case FIXED, INHERIT -> throw new IllegalStateException("A settings row cannot change a " + kind + " key");
    };
  }
}
