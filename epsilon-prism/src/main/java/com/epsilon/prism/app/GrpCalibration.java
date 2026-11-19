package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.grp.RankCalibration;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

@Named(
    value = "grp-calibration",
    doc = "Compare predicted and realized final ranks by next round, rank and score gap")
public record GrpCalibration(
    @Key(doc = "Run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Recorded cohort", presence = Presence.REQUIRED) @Range(min = 1) int cohort)
    implements PrismCommand {
  @Override
  public Outcome run(Context<PrismSettings> context) throws IOException {
    context.requireNoLayers();
    var settings = PrismApp.latest(run);
    var setup = new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed());
    var lines = RankCalibration.read(new GrpFiles(run), setup, cohort);
    var out = context.out();
    var header = new StringBuilder("nextRound\tdirection\tgap\trank\tseatHands");
    int seats = settings.run().variant().seats();
    for (String kind : List.of("predicted", "observed")) {
      for (int r = 1; r <= seats; r++) header.append('\t').append(kind).append(r);
    }
    out.println(header.append("\tpredictedUtility\tobservedUtility"));
    for (var line : lines) {
      out.printf(
          "%s\t%s\t%s\t%d\t%d",
          line.round(), line.direction(), line.gap(), line.rank(), line.seatHands());
      for (Double predicted : line.predicted()) out.print("\t" + predicted);
      for (Double observed : line.observed()) out.print("\t" + observed);
      out.printf("\t%s\t%s%n", line.predictedUtility(), line.observedUtility());
    }
    return Outcome.ok("Measured cohort " + cohort);
  }
}
