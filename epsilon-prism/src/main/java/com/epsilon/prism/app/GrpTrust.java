package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.grp.train.GateFailure;
import com.epsilon.grp.train.GateReport;
import com.epsilon.grp.train.GrpTrained;
import com.epsilon.grp.train.TrustGames;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.grp.TrustTrial;
import com.epsilon.prism.settings.PrismSettings;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;

/**
 * 点数だけの Φ から初めて GRP に切り替える関門で、2 SE の判定に要る半荘の数を測る（{@link TrustTrial}）。設定は変えない。測れるのは、
 * Φ を自分の GRP から作る run（SELF_GRP）だけ。学び直しは範囲の半荘を全て学習に使い、関門は範囲の次のコホートの半荘で比べる。そのため、
 * 判定に要る半荘の数は 1 コホートの半荘の数と比べる（{@link TrustTrial#cohortHasPower}）。提案する trustGames は、設定のキーと同じく
 * 学び直しの範囲の半荘の数で、候補を学習した半荘の数以上の、最小の 2 の冪にする。1 コホートの半荘が判定に足りないときと、学習した候補が
 * 関門用の半荘で最初の切り替えのほかの条件（有限、単調）を満たさなかったときは、それを出して、提案はしない。
 *
 * @param run run のディレクトリ
 * @param cohorts 測るコホートの範囲（a..b。無ければ、点数だけの Φ で集めて半荘の記録がある、最新の確定済みのコホート 1 つ）
 * @param resamples ブートストラップの回数
 */
@Named(value = "grp-trust", doc = "Measure how many games the first switch away from the score-only Φ needs for a 2 SE decision")
public record GrpTrust(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Range of cohorts to measure (a..b; default: the latest sealed cohort collected with the score-only Φ)", presence = Presence.ABSENT)
        String cohorts,
    @Key(doc = "Number of bootstrap samples", value = "2000") @Range(min = 2) int resamples)
    implements PrismCommand {

  /** 提案する trustGames の上限（int に収まる最大の 2 の冪）。 */
  private static final int MAX_SUGGESTION = 1 << 30;

  public GrpTrust {
    if (cohorts != null) {
      Span.of("cohorts", cohorts);
    }
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    PrismSettings settings = PrismApp.latest(run);
    if (!(settings.phi() instanceof PhiSettings.SelfGrp self)) {
      throw new SettingsException(
          List.of("The run " + run + " has the Φ source " + settings.phi().source() + ", not SELF_GRP"));
    }
    GrpFiles files = new GrpFiles(run);
    Span span;
    if (cohorts != null) {
      span = Span.of("cohorts", cohorts);
    } else {
      OptionalInt latest = TrustTrial.latestCohort(files);
      if (latest.isEmpty()) {
        throw new SettingsException(List.of("The run " + run + " has no sealed cohort recorded with a score-only Φ"));
      }
      span = new Span(latest.getAsInt(), latest.getAsInt());
    }
    TrustTrial trial =
        TrustTrial.run(
            files,
            new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed()),
            span.first(),
            span.last(),
            resamples);
    TrustGames trust = trial.trust();
    GateReport gate = trial.gate();
    GrpTrained trained = trial.trained();
    PrintStream out = context.out();
    out.println(
        String.format(
            Locale.ROOT,
            "cohorts %s of %s collected with %s (%s): %d games learned from scratch (%d rows, %d stop rows,"
                + " %d half epochs, stop NLL %.6f), %d games held out",
            span,
            settings.run().variant(),
            trial.used().id(),
            trial.used().kind(),
            trial.learnedGames(),
            trial.rows(),
            trial.stopRows(),
            trained.halfEpochs(),
            trained.stopNll(),
            trial.heldGames()));
    out.println(
        String.format(
            Locale.ROOT,
            "held out: %d games with hand ends, %.3f hand ends per game, paired MSE candidate - used %+.6f"
                + " (SE %.6f, z %+.2f), bootstrap SD %.6f over %d resamples",
            trust.games(),
            trust.handEndsPerGame(),
            trust.difference().mean(),
            trust.difference().standardError(),
            trust.difference().z(),
            trust.bootstrapSd(),
            trust.resamples()));
    out.println(
        String.format(
            Locale.ROOT,
            "replacement gate on the held-out games: %s (monotone violations %.4f, line %.4f)",
            gate.passed() ? "PASSED" : "REJECTED " + gate.failures(),
            gate.monotoneRate(),
            gate.monotoneLine()));
    List<GateFailure> others = new ArrayList<>(gate.failures());
    others.remove(GateFailure.NOT_BETTER);
    if (!others.isEmpty()) {
      out.println(
          "the candidate learned from " + trial.learnedGames() + " games fails " + others
              + ", so the first switch needs a larger window than the 2 SE judgment alone: measure more cohorts");
    }
    int now = self.grp().trustGames();
    if (Double.isInfinite(trust.withPower())) {
      out.println(
          "the candidate is not better on the held-out games, so no number of games passes the 2 SE judgment"
              + " (the run has trustGames " + now + ")");
      return Outcome.ok(
          String.format(
              Locale.ROOT,
              "not better: gate=%s difference=%+.6f sd=%.6f",
              verdict(gate),
              trust.difference().mean(),
              trust.bootstrapSd()));
    }
    if (!trial.cohortHasPower()) {
      out.println(
          "the gate of a refit compares only the games of the next cohort, and a cohort of the range has "
              + trial.cohortGames() + " games with hand ends, fewer than the 2 SE judgment needs: a larger trustGames"
              + " adds none, so measure more cohorts or collect larger ones");
    }
    boolean enough = others.isEmpty() && trial.cohortHasPower();
    OptionalInt suggested = enough ? suggestion(trial.learnedGames()) : OptionalInt.empty();
    String suggestion =
        suggested.isPresent() ? Integer.toString(suggested.getAsInt()) : enough ? "over " + MAX_SUGGESTION : "none";
    out.println(
        String.format(
            Locale.ROOT,
            "2 SE judgment: %d games at the line, %d games with 95%% power, counted in the cohort after the refit"
                + " range (a cohort of the range has %d); the candidate learned from %d games of the range;"
                + " suggested trustGames %s (the run has %d)",
            (long) Math.ceil(trust.atLine()),
            (long) Math.ceil(trust.withPower()),
            trial.cohortGames(),
            trial.learnedGames(),
            suggestion,
            now));
    return Outcome.ok(
        String.format(
            Locale.ROOT,
            "trustGames=%s learned=%d cohort=%d line=%d power=%d gate=%s difference=%+.6f sd=%.6f",
            suggested.isPresent() ? Integer.toString(suggested.getAsInt()) : "none",
            trial.learnedGames(),
            trial.cohortGames(),
            (long) Math.ceil(trust.atLine()),
            (long) Math.ceil(trust.withPower()),
            verdict(gate),
            trust.difference().mean(),
            trust.bootstrapSd()));
  }

  /** 半荘の数以上で最小の 2 の冪を返す。{@value #MAX_SUGGESTION} を超えれば空を返す。 */
  static OptionalInt suggestion(double games) {
    if (!(games <= MAX_SUGGESTION)) {
      return OptionalInt.empty();
    }
    int needed = Math.max(1, (int) Math.ceil(games));
    return OptionalInt.of(needed == 1 ? 1 : Integer.highestOneBit(needed - 1) << 1);
  }

  private static String verdict(GateReport gate) {
    return gate.passed() ? "PASSED" : "REJECTED" + gate.failures();
  }
}
