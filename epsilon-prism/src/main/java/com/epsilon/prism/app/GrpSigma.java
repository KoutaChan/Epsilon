package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Span;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.grp.ScoreSmooth;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.SmoothSigma;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.settings.PrismSettings;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.OptionalDouble;

/**
 * Φ（点数のなめらかな関数）が使う σ の係数を、run の半荘の記録から測る（{@link SmoothSigma}）。設定も、コードにある係数も変えない。
 * ルールは記録のファイルから読む。
 *
 * @param run run のディレクトリ
 * @param cohorts 測るコホートの範囲（a..b。無ければ、半荘の記録がある確定済みのコホートの全て）
 */
@Named(value = "grp-sigma", doc = "Measure the σ coefficient of the smooth score-only Φ from the game records")
public record GrpSigma(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Range of cohorts to measure (a..b; default: every sealed cohort with game records)", presence = Presence.ABSENT)
        String cohorts)
    implements PrismCommand {

  /** 局の名前に使う場風の頭文字（東・南・西）。 */
  private static final String WINDS = "ESW";

  public GrpSigma {
    if (cohorts != null) {
      Span.of("cohorts", cohorts);
    }
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    context.requireNoLayers();
    GrpFiles files = new GrpFiles(run);
    Span span = cohorts == null ? null : Span.of("cohorts", cohorts);
    List<Integer> chosen = new ArrayList<>();
    for (int cohort : files.recordedCohorts()) {
      if (files.recorded(cohort) && (span == null || (cohort >= span.first() && cohort <= span.last()))) {
        chosen.add(cohort);
      }
    }
    if (chosen.isEmpty()) {
      String where = span == null ? "" : " in " + span;
      throw new SettingsException(List.of("The run " + run + " has no sealed cohort with game records" + where));
    }
    Variant variant = null;
    SmoothSigma.Builder builder = null;
    long games = 0;
    for (int cohort : chosen) {
      GameCohort recorded = GameRecordCodec.read(files.records(cohort));
      if (builder == null) {
        variant = recorded.variant();
        builder = SmoothSigma.builder(variant);
      } else if (recorded.variant() != variant) {
        throw new IllegalStateException(
            "Variant mismatch: expected " + variant + " but found " + recorded.variant() + " in "
                + files.records(cohort));
      }
      builder.add(recorded.games());
      games += recorded.games().size();
    }
    SmoothSigma sigma = builder.build();
    PrintStream out = context.out();
    for (SmoothSigma.Round round : sigma.rounds()) {
      out.println(
          String.format(
              Locale.ROOT,
              "next %s: remaining %d, seat-hands %d, spread %.1f, per hand %.1f",
              name(variant, round.round()),
              round.remaining(),
              round.seatHands(),
              round.spread(),
              round.perHand()));
    }
    OptionalDouble inUse = ScoreSmooth.sigmaPerHand(variant);
    out.println(
        String.format(
            Locale.ROOT,
            "%s over %d cohorts (%d..%d), %d games, %d seat-hands: sigma per hand %.1f (in use: %s)",
            variant,
            chosen.size(),
            chosen.getFirst(),
            chosen.getLast(),
            games,
            sigma.seatHands(),
            sigma.perHand(),
            inUse.isPresent() ? String.format(Locale.ROOT, "%.0f", inUse.getAsDouble()) : "none"));
    return Outcome.ok(
        String.format(
            Locale.ROOT, "variant=%s sigmaPerHand=%.1f seatHands=%d", variant, sigma.perHand(), sigma.seatHands()));
  }

  /** 局の名前（東 1 局なら E1）を返す。 */
  private static String name(Variant variant, int round) {
    return WINDS.charAt(round / variant.seats()) + Integer.toString(round % variant.seats() + 1);
  }
}
