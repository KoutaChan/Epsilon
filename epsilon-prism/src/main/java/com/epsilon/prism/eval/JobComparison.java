package com.epsilon.prism.eval;

import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.PairedWalls;
import com.epsilon.agent.eval.WallMean;
import com.epsilon.agent.eval.WallValues;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Function;

/**
 * 同じ卓の 2 つの評価のジョブを、同じ牌山どうしで対にして比べた結果。番号が同じ牌山どうしを対にし、牌山ごとの δ と υ の差（2 つ目 − 1 つ目）
 * の平均と、対の標準誤差を求める。2 つの計画は席の方式と方式が同じなので、同じ番号の牌山の半荘は同じ割り当てで打っている（games.tsv の行は、
 * 読むときに計画の割り当てと照合する）。
 *
 * <p>対にするのは、候補のほかの計画（{@link Element}）と、同値な打牌の切り方が同じジョブどうしだけ。牌山の数と止め方は違ってよい。片方の
 * ジョブにしか無い牌山（早く止まったジョブや回っている途中のジョブより後ろの牌山）は対にせず、数だけを残す。計画を決めた jar が違う
 * ジョブどうしも対にする（差には jar の違いも入る）。
 *
 * @param first 1 つ目のジョブ
 * @param second 2 つ目のジョブ
 * @param walls 対にした牌山の数
 * @param rank 牌山ごとの δ の差（2 つ目 − 1 つ目。負なら 2 つ目が良い）
 * @param utility 牌山ごとの υ の差（2 つ目 − 1 つ目。正なら 2 つ目が良い）
 */
public record JobComparison(Side first, Side second, int walls, Value rank, Value utility) {

  /**
   * 比べた 1 つのジョブ。
   *
   * @param run ジョブを持つ run
   * @param job ジョブの記録（状態と済んだ牌山の数）
   * @param plan ジョブの計画
   * @param rank 対にした牌山の δ の平均と標準誤差
   * @param utility 対にした牌山の υ の平均と標準誤差
   */
  public record Side(RunDirectory run, JobFile job, EvaluationPlan plan, Value rank, Value utility) {}

  /**
   * 牌山ごとの値の平均と、平均の標準誤差（不偏分散から求める）。
   *
   * @param mean 平均
   * @param error 標準誤差
   */
  public record Value(double mean, double error) {

    private static Value of(WallMean values) {
      return new Value(values.mean(), values.error());
    }
  }

  /** 候補のほかに、対にする 2 つの計画で同じでなければならない要素。 */
  enum Element {
    /** ルール（四人打ち・三人打ち）。 */
    VARIANT("variant", DuelSetup::variant),
    /**
     * 相手。系譜のバージョンは指定（run のスナップショットのパス）とバージョンで比べるので、別の系譜の同じ番号のバージョンは別の相手に
     * なる。系譜の外のチェックポイントは SHA-256 で、チェックポイントの無い相手は書き方で比べる。
     */
    OPPONENT("opponent", duel -> opponentKey(duel.opponent()), duel -> opponentText(duel.opponent())),
    WALL_SEED("wall seed", DuelSetup::wallSeed),
    FIRST_WALL("first wall", DuelSetup::wallBase),
    /** 席の方式と方式（割り当ての番号の振り方）。 */
    SEATING("seating", duel -> duel.seating() + " " + duel.splits()),
    DRAWS("draw order", DuelSetup::draws),
    UTILITY("rank utility", DuelSetup::utility);

    private final String noun;
    private final Function<DuelSetup, Object> key;
    private final Function<DuelSetup, Object> text;

    Element(String noun, Function<DuelSetup, Object> key) {
      this(noun, key, key);
    }

    Element(String noun, Function<DuelSetup, Object> key, Function<DuelSetup, Object> text) {
      this.noun = noun;
      this.key = key;
      this.text = text;
    }

    /** @return 2 つの計画で違えば、その違いの英語の句（{@code the wall seed differs (1 and 2)}）。同じなら空 */
    Optional<String> mismatch(DuelSetup first, DuelSetup second) {
      if (key.apply(first).equals(key.apply(second))) {
        return Optional.empty();
      }
      return Optional.of("the " + noun + " differs (" + text.apply(first) + " and " + text.apply(second) + ")");
    }
  }

  /**
   * 2 つのジョブを読み、同じ牌山で対にする。
   *
   * @param table 卓の名前
   * @param firstRun 1 つ目のジョブを持つ run
   * @param firstVersion 1 つ目のジョブのバージョン
   * @param secondRun 2 つ目のジョブを持つ run（1 つ目と同じでもよい）
   * @param secondVersion 2 つ目のジョブのバージョン
   * @return 対にして比べた結果
   * @throws SettingsException 同じジョブを 2 度渡したか、ジョブが無いか、まだ計画が無いか、打たずに候補を初めの王者にしたジョブか、候補の
   *     ほかの計画か同値な打牌の切り方が違うか、対にした牌山が 2 つより少ないとき（理由ごとに英語の 1 文）
   * @throws IOException ジョブの記録が読めないか、games.tsv の行が済んだ牌山の分より少ないとき
   * @throws IllegalArgumentException games.tsv の行が計画と合わないとき
   */
  public static JobComparison of(
      String table, RunDirectory firstRun, int firstVersion, RunDirectory secondRun, int secondVersion)
      throws IOException {
    String firstName = RunDirectory.jobName(firstVersion, table);
    String secondName = RunDirectory.jobName(secondVersion, table);
    Path firstHome = firstRun.job(firstName);
    Path secondHome = secondRun.job(secondName);
    if (firstHome.equals(secondHome)) {
      throw new SettingsException(List.of("Both jobs are " + firstHome + "; compare two different jobs"));
    }
    List<String> problems = new ArrayList<>();
    Optional<Planned> firstJob = planned(firstRun, firstName, problems);
    Optional<Planned> secondJob = planned(secondRun, secondName, problems);
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    Planned first = firstJob.orElseThrow();
    Planned second = secondJob.orElseThrow();
    boolean oneRun = firstRun.equals(secondRun);
    String pair =
        "Cannot pair " + (oneRun ? firstName : firstHome.toString()) + " with "
            + (oneRun ? secondName : secondHome.toString());
    DuelSetup firstDuel = first.plan().duel();
    DuelSetup secondDuel = second.plan().duel();
    for (Element element : Element.values()) {
      element.mismatch(firstDuel, secondDuel).ifPresent(mismatch -> problems.add(pair + ": " + mismatch));
    }
    if (first.plan().twinDiscard() != second.plan().twinDiscard()) {
      problems.add(
          pair + ": the twin discard differs (" + first.plan().twinDiscard() + " and " + second.plan().twinDiscard()
              + ")");
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    WallValues firstValues = JobGames.read(firstHome, first.job(), firstDuel);
    WallValues secondValues = JobGames.read(secondHome, second.job(), secondDuel);
    int walls = Math.min(firstValues.walls(), secondValues.walls());
    if (walls < 2) {
      throw new SettingsException(
          List.of(pair + ": a paired standard error needs at least two walls finished in both jobs, not " + walls));
    }
    return new JobComparison(
        side(firstRun, first, firstValues, walls),
        side(secondRun, second, secondValues, walls),
        walls,
        Value.of(PairedWalls.difference(secondValues.rank(), firstValues.rank())),
        Value.of(PairedWalls.difference(secondValues.utility(), firstValues.utility())));
  }

  /** 計画で牌山を打つジョブ。 */
  private record Planned(JobFile job, EvaluationPlan plan) {}

  /**
   * ジョブの記録と計画を読む。
   *
   * @param problems 対にできない理由を足す先
   * @return 計画で牌山を打つジョブ。ジョブが無いか、まだ計画が無いか、打たずに候補を初めの王者にしたジョブなら空
   */
  private static Optional<Planned> planned(RunDirectory run, String name, List<String> problems) throws IOException {
    Path file = run.job(name).resolve(Jobs.JOB);
    if (!Files.exists(file)) {
      problems.add("There is no job " + name + " in " + run.root());
      return Optional.empty();
    }
    JobFile job = RecordFile.read(JobFile.class, file).value();
    if (job.plan() == null) {
      problems.add("The job " + name + " in " + run.root() + " has not started, so it has no plan yet");
      return Optional.empty();
    }
    EvaluationPlan plan = RecordFile.read(EvaluationPlan.class, job.plan()).value();
    if (plan.crowns()) {
      problems.add(
          "The job " + name + " in " + run.root() + " played no walls: it made " + plan.candidate().label()
              + " the first champion");
      return Optional.empty();
    }
    return Optional.of(new Planned(job, plan));
  }

  private static Side side(RunDirectory run, Planned planned, WallValues values, int walls) {
    return new Side(
        run,
        planned.job(),
        planned.plan(),
        Value.of(head(values.rank(), walls)),
        Value.of(head(values.utility(), walls)));
  }

  /** 最初の牌山から walls 個の値の平均と標準誤差（牌山の番号の順に足す）。 */
  private static WallMean head(double[] values, int walls) {
    WallMean mean = new WallMean();
    for (int index = 0; index < walls; index++) {
      mean.add(values[index]);
    }
    return mean;
  }

  /**
   * 相手を比べる値。系譜のバージョンは指定とバージョン（{@link DuelSetup#pairsWith} と同じ）、系譜の外のチェックポイントは SHA-256、
   * チェックポイントの無い相手は書き方。
   */
  private static String opponentKey(DuelSetup.Entrant opponent) {
    if (opponent.version() != null) {
      return "version " + opponent.version() + " " + opponent.spec();
    }
    return opponent.sha256() != null ? "sha256 " + opponent.sha256() : opponent.spec();
  }

  private static String opponentText(DuelSetup.Entrant opponent) {
    if (opponent.version() != null) {
      return opponent.label() + " " + opponent.spec();
    }
    return opponent.sha256() != null ? opponent.label() + " sha256 " + opponent.sha256() : opponent.spec();
  }
}
