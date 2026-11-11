package com.epsilon.agent.app;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.IterationFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.agent.settings.Toml;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 効いている設定を、値ごとの出どころを添えた TOML で出す。run を渡したときは、最新の行、反復に固定した行、ジョブを作った行の
 * どれかを、前の行から変わったキーと一緒に出す。行の値のうち既定値と同じものには出どころを添えず、ほかの値には、そのキーを最後に変えた
 * 行を添える。
 *
 * @param run 設定の行を出す run のディレクトリ。指定しなければ、重ねた層から作った設定を出す
 * @param iteration この反復に固定した行を出す。run と一緒に渡す
 * @param job このジョブを作った行を出す。run と一緒に渡す
 */
@Named(value = "settings show", doc = "Print the effective settings as TOML, with the source of each value")
public record SettingsShow(
    @Key(doc = "Run directory whose settings row to print", presence = Presence.ABSENT) Path run,
    @Key(doc = "Print the row pinned to this iteration", presence = Presence.ABSENT) Integer iteration,
    @Key(doc = "Print the row that created this job", presence = Presence.ABSENT) String job)
    implements BuiltinCommand {

  @Override
  public Outcome run(Context<Record> context) throws Exception {
    if (run == null) {
      if (iteration != null || job != null) {
        throw new SettingsException(List.of("--iteration and --job need --run"));
      }
      Settings<Record> settings = context.settings();
      context.out().print(settings.toml());
      return Outcome.ok(settings.effective().size() + " settings");
    }
    context.requireNoLayers();
    if (iteration != null && job != null) {
      throw new SettingsException(List.of("Give --iteration or --job, not both"));
    }
    RunDirectory directory = new RunDirectory(run);
    SettingsRows rows = new SettingsRows(directory);
    List<SettingsRow> all = rows.read();
    SettingsRow shown;
    String why;
    if (iteration != null) {
      shown = rows.row(fixedRow(directory, iteration));
      why = "fixed to iteration " + iteration;
    } else if (job != null) {
      if (!Files.exists(directory.job(job).resolve(Jobs.JOB))) {
        throw new SettingsException(List.of("The run " + run + " has no job " + job));
      }
      JobFile file = new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS)).read(job);
      shown = rows.row(file.row());
      why = "that created job " + job;
    } else {
      shown = rows.latest();
      why = "of " + all.size() + " (from iteration " + shown.from() + ")";
    }
    Settings<Record> settings = shown.bind(context.application().schema());
    StringBuilder text = new StringBuilder();
    text.append("# settings row ")
        .append(shown.row())
        .append(' ')
        .append(why)
        .append(", written ")
        .append(shown.time())
        .append(" by jar ")
        .append(shown.jar())
        .append(": ")
        .append(shown.reason())
        .append('\n');
    for (SettingsRow.ChangedKey changed : shown.changes()) {
      text.append("# changed ")
          .append(changed.key())
          .append(": ")
          .append(changed.before() == null ? "(none)" : Toml.inline(changed.before()))
          .append(" -> ")
          .append(changed.after() == null ? "(none)" : Toml.inline(changed.after()))
          .append(" (")
          .append(changed.change())
          .append(")\n");
    }
    context.out().print(text + listed(context.application(), all, shown, settings).toml());
    return Outcome.ok("row " + shown.row() + ", " + settings.effective().size() + " settings");
  }

  /**
   * 行の値を、出どころを付け直して結び付ける。既定値（四人打ち・三人打ちごとの層の値を含む）と同じ値は既定値の出どころのまま、ほかの
   * 値は、見せる行までのうち、そのキーを最後に変えた行（変えた行が無ければ 1 行目）を出どころにする。付け直しで値は変わらないはずだが、
   * 変わったら行の値をそのまま出す。
   */
  private static <R extends Record> Settings<R> listed(
      Application<R> application, List<SettingsRow> all, SettingsRow shown, Settings<R> settings) {
    Map<String, Integer> changedAt = new HashMap<>();
    for (SettingsRow row : all) {
      if (row.row() <= shown.row()) {
        for (SettingsRow.ChangedKey changed : row.changes()) {
          changedAt.put(changed.key(), row.row());
        }
      }
    }
    Node.Table beyond = Settings.beyondDefaults(application.schema(), shown.settings(), application::stack);
    Node.Table relabelled = Settings.relabelValues(beyond, path -> "settings row " + lastChange(changedAt, path));
    Layer row = new Layer("settings row " + shown.row(), Layer.Source.RECORD, relabelled, List.of());
    Settings<R> listed = Settings.bind(application.schema(), application.stack(List.of(row)));
    return listed.effective().equals(settings.effective()) ? listed : settings;
  }

  /** そのキーを（空のテーブルなら、その中のキーも）最後に変えた行。変えた行が無ければ 1 行目。 */
  private static int lastChange(Map<String, Integer> changedAt, String path) {
    int row = 1;
    for (Map.Entry<String, Integer> entry : changedAt.entrySet()) {
      if (entry.getKey().equals(path) || entry.getKey().startsWith(path + ".")) {
        row = Math.max(row, entry.getValue());
      }
    }
    return row;
  }

  /**
   * 反復に固定した行の番号を返す。反復の作業ディレクトリに固定のファイルがあれば、そこから読む。無ければ、閉じた反復の報告から読む。
   *
   * @throws SettingsException その反復をまだ始めていない
   */
  private static int fixedRow(RunDirectory directory, int iteration) throws IOException {
    Path fixed = directory.iterationFile(iteration);
    if (Files.exists(fixed)) {
      return RecordFile.read(IterationFile.class, fixed).value().settingsRow();
    }
    Path reports = directory.report(RunDirectory.ITERATIONS);
    for (String line : DurableFiles.lines(reports)) {
      Node.Table report = Json.read(line, reports.toString());
      if (report.get("iteration") instanceof Node.Numeral number
          && number.integer()
          && number.decimal().intValueExact() == iteration
          && report.get("settingsRow") instanceof Node.Numeral row) {
        return row.decimal().intValueExact();
      }
    }
    throw new SettingsException(List.of("Iteration " + iteration + " of " + directory.root() + " has not started"));
  }
}
