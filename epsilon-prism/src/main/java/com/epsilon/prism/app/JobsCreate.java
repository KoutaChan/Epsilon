package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.eval.Placements;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.run.PrismState;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開したバージョンを、run の最新の設定の行の卓で測るジョブを、範囲のバージョンごとに PENDING で作る。学習の JVM は次の反復の前に、
 * 待っているジョブをバージョンの古い順に回す（止めた run では {@code jobs run} が回す）。同じ名前のジョブが既にあれば、済んだジョブでも
 * 途中のジョブでも触れない。作ったジョブの数と、それらが打つ半荘の合計をログに出す（計画の牌山を全て打たない止め方の卓では上限）。
 *
 * <p>相手が王者で止め方が判定の卓には作らない。その卓の結果は、系譜がバージョンを公開した順に、王者の昇格とリグレッションの判定に
 * 当てるため。相手が決まった番号のバージョンの卓（基準の卓）では、範囲にその番号が入れば作らない。その番号のジョブは、同じバージョン
 * どうしの決闘になるため。
 *
 * @param run run のディレクトリ
 * @param table 卓の名前（最新の設定の行の {@code evaluation.tables} の名前）
 * @param versions 測るバージョンの範囲（a..b）
 */
@Named(value = "jobs create", doc = "Create jobs that measure published versions on a table (they run before the next iteration)")
public record JobsCreate(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Table name (a name in evaluation.tables of the latest settings row)", presence = Presence.REQUIRED) String table,
    @Key(doc = "Range of versions to measure (a..b)", presence = Presence.REQUIRED) String versions)
    implements PrismCommand {

  private static final Logger LOG = LoggerFactory.getLogger(JobsCreate.class);

  public JobsCreate {
    Span.of("versions", versions);
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    SettingsRow row = new SettingsRows(directory).latest();
    PrismSettings settings = row.bind(PrismSettings.class).value();
    TableSpec spec = settings.evaluation().tables().get(table);
    if (spec == null) {
      throw new SettingsException(
          List.of(
              "Unknown table " + table + " in settings row " + row.row() + "; the tables are "
                  + settings.evaluation().tables().keySet()));
    }
    Span span = Span.of("versions", versions);
    requireMeasurable(spec, span, published(directory));
    Node.Table definition = RecordFile.tree(spec);
    int order = new ArrayList<>(settings.evaluation().tables().keySet()).indexOf(table);
    Jobs jobs = new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS));
    int created = 0;
    for (int version = span.first(); version <= span.last(); version++) {
      if (jobs.create(JobFile.pending(version, table, order, row.row(), definition))) {
        created++;
      }
    }
    String games =
        (spec.stop() instanceof StopSettings.Fixed ? "" : "at most ")
            + created * games(spec, settings.run().variant())
            + " games";
    LOG.info(
        "Created {} PENDING job(s) on table {} for versions {} with settings row {} ({} existing job(s) left as they"
            + " are): {}",
        created,
        table,
        span,
        row.row(),
        span.size() - created,
        games);
    return Outcome.ok("created " + created + " job(s), " + games);
  }

  /**
   * 範囲のバージョンを、この卓で測れるかを確かめる。
   *
   * @param latest 公開した最新のバージョン
   * @throws SettingsException 公開していないバージョンがあるか、相手のバージョンが公開していないバージョンになるか、相手が決まった番号の
   *     バージョンの卓で範囲にその番号が入るか、相手が王者で止め方が判定の卓
   */
  private void requireMeasurable(TableSpec spec, Span span, int latest) {
    List<String> problems = new ArrayList<>();
    if (span.last() > latest) {
      problems.add("Version " + span.last() + " is not published: the run has published versions 1.." + latest);
    }
    TableSpec.Opponent opponent = TableSpec.kind(spec.opponent());
    if (opponent == TableSpec.Opponent.VERSION || opponent == TableSpec.Opponent.DISTANCE) {
      int first = spec.opponentVersion(span.first());
      int last = spec.opponentVersion(span.last());
      if (first < 1 || last > latest) {
        problems.add(
            "Table " + table + " plays " + spec.opponent() + ", so versions " + span + " would play versions " + first
                + ".." + last + ", but the run has published versions 1.." + latest);
      }
    }
    if (opponent == TableSpec.Opponent.VERSION) {
      int fixed = spec.opponentVersion(span.first());
      if (span.first() <= fixed && fixed <= span.last()) {
        problems.add("Table " + table + " plays " + spec.opponent() + ", so version " + fixed + " would play itself");
      }
    }
    if (opponent == TableSpec.Opponent.CHAMPION && spec.stop() instanceof StopSettings.Test) {
      problems.add(
          "Table " + table + " duels the champion with a TEST stop: its results promote and roll back the lineage in"
              + " the order the versions are published, so it takes no jobs for past versions");
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
  }

  /**
   * 公開した最新のバージョン（{@link PrismState#next}）。
   *
   * @throws SettingsException state.json が無い（まだバージョンを公開していない）
   */
  private static int published(RunDirectory directory) throws IOException {
    if (!Files.exists(directory.state())) {
      throw new SettingsException(
          List.of("The run has published no version yet: " + directory.state() + " is missing"));
    }
    return RecordFile.read(PrismState.class, directory.state()).value().next();
  }

  /** 卓のジョブ 1 つが打つ半荘の数（計画の牌山を全て打たない止め方の卓では上限）。 */
  private static long games(TableSpec spec, Variant variant) {
    Placements placements = new Placements(variant, spec.seating().seating(), spec.seating().splits(variant));
    return (long) spec.walls() * placements.gamesPerWall();
  }
}
