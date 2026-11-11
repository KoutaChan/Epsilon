package com.epsilon.classic.campaign;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Sequential;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Halt;
import com.epsilon.agent.run.Halted;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 王者の保存先のジョブ（王者の決闘と確かめの卓）を、段の中から最後まで打つ。ジョブの形は run のジョブと同じで、計画は初めて打つときに渡した
 * 決闘の計画。チャンクの後の安全点で止まると決めたら {@link Halted} を投げ、次に段を回したときに止めた所から続ける。
 */
final class DuelJobs {

  /** 計画の牌山を全て打つ卓の区間の、両側の有意水準（固定の 99%）。 */
  static final double FIXED_ALPHA = 0.01;

  /** ジョブのディレクトリの中の、済んだジョブの結果を 1 行ずつ足すファイル。 */
  static final String RESULTS = "results.jsonl";

  private static final Logger LOG = LoggerFactory.getLogger(DuelJobs.class);

  private DuelJobs() {}

  /**
   * 済んだジョブ。
   *
   * @param setup 計画
   * @param result 結果
   * @param home ジョブのディレクトリ
   */
  record Played(DuelSetup setup, DuelResult result, Path home) {

    /** @return 牌山ごとの、判定に使う量の候補が良い向きの値（牌山の番号の順） */
    double[] values() throws IOException {
      return WallValues.of(setup, DurableFiles.lines(home.resolve(JobOutputs.GAMES))).better(setup.score());
    }
  }

  /** @return ジョブのディレクトリ（済んだジョブの結果は、中の {@value #RESULTS} に足す） */
  static Jobs jobs(Path directory) {
    return new Jobs(directory, directory.resolve(RESULTS));
  }

  /** @return macro の候補の名前（{@code v00010}） */
  static String label(int macro) {
    return String.format(Locale.ROOT, "v%05d", macro);
  }

  /**
   * @return 系譜の打ち手（候補か王者）。書き方は、学習器の系列の接頭辞とチェックポイントのパスで、バージョンで見分けるので SHA-256 は
   *     持たない。持ち主が置いた初めの王者だけは系譜の外のチェックポイントなので、バージョンの代わりに置いたときの SHA-256 を持ち、
   *     決闘を開くときに照合する
   */
  static DuelSetup.Entrant lineage(DuelMacro macro, Champion player) throws IOException {
    String spec = macro.series().prefix() + ":" + player.checkpoint();
    return placed(player)
        ? new DuelSetup.Entrant(spec, player.label(), null, player.sha256())
        : new DuelSetup.Entrant(spec, player.label(), player.version(), null);
  }

  /**
   * @return 持ち主が置いた初めの王者（置いたときの SHA-256 を持つ）なら true。run の外のチェックポイントをそのパスのまま参照するので、
   *     書き換えない run のバージョンと違い、開くときに置いたときの SHA-256 と照合する
   */
  static boolean placed(Champion player) {
    return player.sha256() != null;
  }

  /**
   * 系譜の外の打ち手。チェックポイントを開く打ち手には、今の中身の SHA-256 を添える。
   *
   * @param spec 打ち手の書き方（heuristic か、接頭辞と絶対パス）
   * @throws IOException チェックポイントが読めないとき
   */
  static DuelSetup.Entrant outside(String spec) throws IOException {
    Series.Spec parsed = Series.parse(spec);
    String sha256 = parsed.series().checkpointed() ? CheckpointFiles.sha256(parsed.checkpoint()) : null;
    return new DuelSetup.Entrant(spec, parsed.series().label(parsed.checkpoint()), null, sha256);
  }

  /**
   * 卓の値で、候補と相手を置いた決闘の計画を作る。
   *
   * @param wallSeed 牌山の種の素
   */
  static DuelSetup setup(
      DuelMacro macro, TableSpec table, DuelSetup.Entrant candidate, DuelSetup.Entrant opponent, long wallSeed)
      throws IOException {
    Variant variant = macro.variant();
    return DuelSetup.of(
        variant,
        candidate,
        opponent,
        table.seating().seating(),
        table.seating().splits(variant),
        table.draws(),
        table.wallSet(),
        wallSeed,
        Arena.WALL_BASE,
        table.walls(),
        rule(table.stop()),
        table.score(),
        macro.utility());
  }

  /**
   * ジョブを最後まで打つ。もう済んでいれば打たずに結果を読む。
   *
   * @param directory ジョブのディレクトリ
   * @param job まだ無ければ作るジョブ
   * @param setup ジョブを初めて打つときの計画
   * @return 済んだジョブ。落ちた回数の上限で FAILED になったら空
   * @throws Halted チャンクの後の安全点で止まると決めた
   * @throws IllegalStateException ジョブが、ほかの候補か、ほかの相手・牌山で計画してある
   */
  static Optional<Played> play(DuelMacro macro, Path directory, JobFile job, DuelSetup setup) throws Exception {
    Jobs jobs = jobs(directory);
    jobs.create(job);
    JobFile current = jobs.read(job.name());
    if (current.plan() != null) {
      DuelSetup planned = planned(current);
      if (!planned.candidate().equals(setup.candidate()) || !planned.pairsWith(setup)) {
        throw new IllegalStateException(
            "Job " + job.name() + " in " + directory + " was planned for " + planned.candidate().label() + " against "
                + planned.opponent().label() + ", not for " + setup.candidate().label() + " against "
                + setup.opponent().label() + " on the same walls; rename the table to measure anew");
      }
    }
    if (current.open()) {
      Optional<Halt> halt = jobs.run(current, new Planner(macro, setup), macro.control());
      if (halt.isPresent()) {
        throw new Halted(halt.get());
      }
      current = jobs.read(job.name());
    }
    Path home = directory.resolve(job.name());
    return switch (current.state()) {
      case DONE -> Optional.of(new Played(planned(current), Jobs.result(home, DuelResult.class), home));
      case FAILED -> {
        LOG.warn("{} FAILED after {} attempts: {}", job.name(), current.attempts(), current.reason());
        yield Optional.empty();
      }
      case PENDING, RUNNING -> throw new IllegalStateException("Job " + job.name() + " is still " + current.state());
    };
  }

  /** @return 判定に使った量の、候補が良い向きの平均と区間 */
  static Estimate better(DuelResult result) {
    return result.score() == Score.RANK ? result.rank().negate() : result.utility();
  }

  /** 卓の止め方を、牌山ごとの値の止め方に直す。計画の牌山を全て打つ卓の区間は固定の 99%。 */
  static StopRule rule(StopSettings stop) {
    return switch (stop) {
      case StopSettings.Fixed fixed -> new StopRule.Fixed(FIXED_ALPHA);
      case StopSettings.Precision precision ->
          new StopRule.Precision(precision.error(), precision.alpha(), precision.checkEvery());
      case StopSettings.Test test ->
          new StopRule.Test(
              new Sequential(test.alpha(), test.spending(), test.checkEvery()), test.evenMargin(), test.evenMargin());
    };
  }

  private static DuelSetup planned(JobFile job) {
    return RecordFile.read(DuelSetup.class, job.plan()).value();
  }

  /**
   * 計画に書いた系譜の外の打ち手のチェックポイントが、計画したときの中身のままかを確かめる。系譜の打ち手は書き換えないので確かめない。
   *
   * @throws IOException チェックポイントが読めないか、SHA-256 が計画と違うとき
   */
  private static void verify(DuelSetup.Entrant entrant) throws IOException {
    Series.Spec spec = Series.parse(entrant.spec());
    if (entrant.version() != null || !spec.series().checkpointed()) {
      return;
    }
    String actual = CheckpointFiles.sha256(spec.checkpoint());
    if (!actual.equals(entrant.sha256())) {
      throw new IOException(
          "The checkpoint " + spec.checkpoint() + " has SHA-256 " + actual + ", but the duel was planned with "
              + entrant.sha256());
    }
  }

  /**
   * ジョブの計画を決め、計画の続きを打つ道を開く。
   *
   * @param setup ジョブを初めて打つときの計画
   */
  private record Planner(DuelMacro macro, DuelSetup setup) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) {
      return RecordFile.tree(setup);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws IOException {
      DuelSetup planned = planned(job);
      verify(planned.candidate());
      verify(planned.opponent());
      Policy candidate = macro.open(planned.candidate().spec());
      Policy opponent;
      try {
        opponent = macro.open(planned.opponent().spec());
      } catch (IOException | RuntimeException | Error failure) {
        candidate.close();
        throw failure;
      }
      return Duel.open(job.name(), macro.table(), planned, candidate, opponent, directory);
    }
  }
}
