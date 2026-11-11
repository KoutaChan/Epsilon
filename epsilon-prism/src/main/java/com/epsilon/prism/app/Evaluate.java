package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.run.Halt;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.prism.eval.EvaluationJobs;
import com.epsilon.prism.eval.EvaluationPlan;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 候補の打ち手を、設定にある評価の卓（{@code evaluation.tables.<名前>}）で打たせて測る。系譜の評価と同じジョブ（{@link EvaluationJobs}）を
 * out に作って回し、結果も系譜と同じ形で書く。ジョブごとの {@code jobs/<ジョブ>/} に job.json・games.tsv・looks.tsv・result.json を書き、
 * {@code reports/evaluations.jsonl} に行を足す。王者の卓なら王者の保存先 {@code champion/} にも書く。アラートは
 * {@code reports/alarms.jsonl} に、報告の表は {@code reports} の tsv に書く。止めても、同じ out で起動し直せば続きから打つ。FIXED の卓の
 * 牌山の種は評価の定数なので、系譜の評価や別の候補と同じ牌山どうしで比べられる。推論と卓の設定、ルールは設定の層から読む。
 *
 * <p>候補は {@code --model} で渡す 1 つ（ジョブのバージョンは {@value #MODEL_VERSION}）か、{@code --snapshots} の run にある
 * {@code --versions} の範囲のバージョンにする。後者は測るだけで、系譜には何も書かない。系譜のバージョンや王者を相手にする卓は
 * {@code --snapshots} のときだけ測れる。相手のバージョンも同じ run から決める。同じジョブが既にあれば、卓の値と候補が同じときだけ続きから
 * 回し（済んでいれば打たない）、違えば拒む。
 *
 * @param model 候補の打ち手（バージョンのディレクトリか prism:DIR、nano:DIR、major:DIR）
 * @param snapshots 測るバージョンを持つ run のディレクトリ
 * @param versions 測るバージョンの範囲（a..b）
 * @param table 卓の名前
 * @param out ジョブと報告を書くディレクトリ
 * @param nanoDevice classic の打ち手のデバイス（cpu か gpuN。無ければルールごとの既定値）
 */
@Named(value = "evaluate", doc = "Play a candidate on a configured evaluation table and measure its placement and utility")
public record Evaluate(
    @Key(doc = "Candidate player (version directory, prism:DIR, nano:DIR, major:DIR)", presence = Presence.ABSENT) String model,
    @Key(doc = "Run directory holding the versions to measure (use with --versions)", presence = Presence.ABSENT) Path snapshots,
    @Key(doc = "Range of versions to measure (a..b)", presence = Presence.ABSENT) String versions,
    @Key(doc = "Table name (a name in evaluation.tables)", presence = Presence.REQUIRED) String table,
    @Key(doc = "Directory for the jobs and the report", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Device of the classic players (cpu or gpuN; default: the per-rule default)", presence = Presence.ABSENT) String nanoDevice)
    implements PrismCommand {

  /** {@code --model} で測るジョブのバージョン。 */
  public static final int MODEL_VERSION = 1;

  /** ジョブを作った設定の行（evaluate のジョブは全て同じ設定で打つ）。 */
  private static final int ROW = 1;

  private static final Logger LOG = LoggerFactory.getLogger(Evaluate.class);

  public Evaluate {
    if (model != null) {
      if (snapshots != null || versions != null) {
        throw new InvalidSetting("model", "--model measures one candidate: drop --snapshots and --versions");
      }
      if (model.equals(EvaluationTable.HEURISTIC)) {
        throw new InvalidSetting("model", "the candidate must be a checkpoint, not " + EvaluationTable.HEURISTIC);
      }
    } else if (snapshots == null || versions == null) {
      throw new InvalidSetting("model", "give --model, or --snapshots with --versions");
    }
    if (versions != null) {
      Span.of("versions", versions);
    }
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    PrismSettings settings = context.settings().value();
    TableSpec spec = settings.evaluation().tables().get(table);
    if (spec == null) {
      throw new SettingsException(
          List.of(
              "Unknown evaluation table " + table + "; the settings define " + settings.evaluation().tables().keySet()));
    }
    TableSpec.Opponent opponent = TableSpec.kind(spec.opponent());
    if (model != null && opponent != TableSpec.Opponent.HEURISTIC && opponent != TableSpec.Opponent.CHECKPOINT) {
      throw new SettingsException(
          List.of(
              "Evaluation table " + table + " plays " + spec.opponent()
                  + ", which needs the versions of a run: give --snapshots and --versions instead of --model"));
    }
    settings.runtime().apply();
    settings.diagnostics().apply();
    List<Integer> measured = new ArrayList<>();
    if (model != null) {
      measured.add(MODEL_VERSION);
    } else {
      Span span = Span.of("versions", versions);
      RunDirectory lineage = new RunDirectory(snapshots);
      List<Integer> missing = new ArrayList<>();
      for (int version = span.first(); version <= span.last(); version++) {
        if (Files.isDirectory(lineage.snapshot(version))) {
          measured.add(version);
        } else {
          missing.add(version);
        }
      }
      if (!missing.isEmpty()) {
        throw new SettingsException(List.of("Versions " + missing + " are not published in " + lineage.root()));
      }
    }
    RunDirectory output = new RunDirectory(out);
    Jobs jobs = new Jobs(output.jobs(), output.report(RunDirectory.EVALUATIONS));
    Node.Table definition = RecordFile.tree(spec);
    InferenceSettings inference = settings.inference();
    int ran = 0;
    String last = null;
    List<String> unfinished = new ArrayList<>();
    try (var runtime = new NetworkRuntime(inference.encoders(), inference.slots(), inference.readyBatches());
        var loop = new TableLoop(settings.table())) {
      EvaluationJobs runner =
          EvaluationJobs.measuring(
              output,
              snapshots == null ? null : new RunDirectory(snapshots),
              model == null ? null : EvaluationTable.absolute(model),
              settings,
              nanoDevice,
              runtime,
              loop);
      List<String> names = new ArrayList<>(measured.size());
      for (int version : measured) {
        JobFile fresh = JobFile.pending(version, table, 0, ROW, definition);
        if (!jobs.create(fresh)) {
          requireSame(output, jobs.read(fresh.name()), definition, runner.candidate(version));
        }
        names.add(fresh.name());
      }
      RunControl control = RunControl.detached(context.signals()::requested);
      for (String name : names) {
        JobFile job = jobs.read(name);
        if (job.state() == JobFile.State.DONE) {
          LOG.info("{} is already measured in {}", name, out);
          continue;
        }
        Optional<Halt> halt = jobs.run(job, runner, control);
        if (halt.isPresent()) {
          return PrismRun.outcome(halt.get().ending());
        }
        job = jobs.read(name);
        if (job.state() != JobFile.State.DONE) {
          unfinished.add(name + " is " + job.state() + " after " + job.attempts() + " attempts: " + job.reason());
          continue;
        }
        ran++;
        last = summary(output, job);
      }
    }
    if (!unfinished.isEmpty()) {
      throw new IllegalStateException("Evaluation jobs in " + out + " did not finish: " + String.join("; ", unfinished));
    }
    if (measured.size() == 1) {
      return Outcome.ok(last != null ? last : RunDirectory.jobName(measured.get(0), table) + " is already measured");
    }
    return Outcome.ok(
        "measured " + ran + " of " + measured.size() + " versions on " + table + ", " + (measured.size() - ran)
            + " were already measured");
  }

  /**
   * 既にあるジョブが、同じ卓の値で同じ候補を測るものかを確かめる。
   *
   * @throws SettingsException 卓の値か、計画した候補が違うとき
   */
  private static void requireSame(
      RunDirectory output, JobFile job, Node.Table definition, DuelSetup.Entrant candidate) {
    boolean same =
        job.definition().equals(definition)
            && (job.plan() == null
                || RecordFile.read(EvaluationPlan.class, job.plan()).value().candidate().equals(candidate));
    if (!same) {
      throw new SettingsException(
          List.of(
              "The job " + output.job(job.name()) + " measures another table or candidate; give another --out"));
    }
  }

  /** 済んだジョブの結果を 1 行にまとめる。 */
  private static String summary(RunDirectory output, JobFile job) throws IOException {
    Path home = output.job(job.name());
    if (RecordFile.read(EvaluationPlan.class, job.plan()).value().crowns()) {
      return job.name() + " made " + Jobs.result(home, EvaluationJobs.Crowned.class).champion() + " the first champion";
    }
    return job.name() + " " + Jobs.result(home, DuelResult.class).summary();
  }
}
