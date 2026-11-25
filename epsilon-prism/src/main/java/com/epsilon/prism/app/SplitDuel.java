package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.StepReport;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.collect.SplitTrial;
import com.epsilon.prism.grp.HandEndPhi;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.luck.LedgerReport;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.run.PhiReport;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.stream.LongStream;

@Named(
    value = "split-duel",
    doc = "Compare two greedy actions on sampled decisions of a frozen reference policy")
public record SplitDuel(
    @Key(doc = "First version directory", presence = Presence.REQUIRED) Path a,
    @Key(doc = "Second version directory", presence = Presence.REQUIRED) Path b,
    @Key(doc = "Reference version directory", presence = Presence.REQUIRED) Path mu,
    @Key(doc = "First wall ID (inclusive)", value = "0") @Range(min = 0) long first,
    @Key(doc = "Number of consecutive walls", presence = Presence.REQUIRED) @Range(min = 1)
        int games,
    @Key(doc = "Decision sampling probability", presence = Presence.REQUIRED)
        @Range(min = 0, minInclusive = false, max = 1)
        double q,
    @Key(
            doc = "Frozen PHI.json step report (default: smooth score utility)",
            presence = Presence.ABSENT)
        Path phi,
    @Key(
            doc =
                "続きを打つ方策の独立した半荘で当てはめた"
                    + " ledger.json。選択が割れた判断では両方の局末効用から各経路の帳簿補正を引いて差を集計する。省略時は生の効用差を使う。",
            presence = Presence.ABSENT)
        Path ledger,
    @Key(doc = "Report directory for JSON and TSV", presence = Presence.ABSENT) Path out)
    implements PrismCommand {

  @Override
  public Outcome run(Context<PrismSettings> context) throws IOException {
    var settings = context.settings().value();
    settings.runtime().apply();
    var variant = settings.run().variant();
    Phi value =
        phi == null
            ? Phi.scoreSmooth(variant, settings.utility())
            : HandEndPhi.openSingleSlot(
                ((PhiReport) StepReport.read(phi, PhiReport.class).metrics()).value(),
                settings.utility());
    var fit = ledger == null ? null : RecordFile.read(LedgerReport.Full.class, ledger).value();
    var recording =
        fit == null
            ? null
            : recording(fit.recording(), settings.noise().ledger(), context.layers());
    var correction = fit == null ? null : LedgerCorrection.restore(fit, recording, variant, null);
    long seed = settings.run().seed();
    var cohort =
        new Collector.Cohort(
            variant,
            seed,
            seed,
            LongStream.range(first, Math.addExact(first, games)).boxed().toList(),
            DrawOrder.SEAT_STREAMS,
            Sampling.OFF,
            recording,
            new UraSettings.Realized(0),
            null,
            settings.behaviour().twinDiscard(),
            null);
    for (Path version : List.of(a, b, mu)) {
      if (PrismNetwork.readFormat(Snapshot.model(version)).variant() != variant)
        throw new IOException("Expected " + variant + " network in " + version);
    }
    try (var table = new TableLoop(settings.table());
        var networks = new ToolNetworks(settings.inference());
        var reference = networks.openDuel(mu);
        var left = networks.openDuel(a);
        var right = networks.openDuel(b)) {
      var firstNetwork = networks.execution(ToolNetworks.Role.A, a, left);
      var secondNetwork = networks.execution(ToolNetworks.Role.B, b, right);
      var referenceNetwork = networks.execution(ToolNetworks.Role.MU, mu, reference);
      var results = SplitTrial.play(table, cohort, reference, value, q, left, right, correction);
      var report = new Report(firstNetwork, secondNetwork, referenceNetwork, results);
      StringBuilder tsv =
          new StringBuilder(
              "group\tpositions\tdisagreements\tdisagreementMean\tdisagreementSE\tmean\tse"
                  + "\taVersion\taBackend\taPrecision\tbVersion\tbBackend\tbPrecision\tmuVersion"
                  + "\tmuBackend\tmuPrecision\n");
      for (var result : results) {
        var diff = result.disagreements();
        var all = result.all();
        tsv.append(result.group())
            .append('\t')
            .append(all.positions())
            .append('\t')
            .append(diff.positions())
            .append('\t')
            .append(diff.mean())
            .append('\t')
            .append(diff.se())
            .append('\t')
            .append(all.mean())
            .append('\t')
            .append(all.se());
        for (var network : List.of(report.a(), report.b(), report.mu())) {
          tsv.append('\t')
              .append(network.version())
              .append('\t')
              .append(network.backend())
              .append('\t')
              .append(network.precision());
        }
        tsv.append('\n');
      }
      context.out().print(tsv);
      if (out != null) {
        DurableFiles.write(
            out.resolve("split-duel.json"),
            new GsonBuilder().serializeNulls().setPrettyPrinting().create().toJson(report));
        DurableFiles.write(out.resolve("split-duel.tsv"), tsv.toString());
      }
    }
    return Outcome.ok("Compared " + games + " walls");
  }

  record Report(
      ToolNetworks.Execution a,
      ToolNetworks.Execution b,
      ToolNetworks.Execution mu,
      List<SplitTrial.Result> results) {}

  static LedgerRecording recording(
      LedgerRecording provenance, LedgerSettings settings, List<Layer> layers) throws IOException {
    var keys = new HashSet<String>();
    for (var layer : layers) {
      if (layer.tree().get("noise") instanceof Node.Table noise
          && noise.get("ledger") instanceof Node.Table ledger)
        keys.addAll(ledger.entries().keySet());
    }
    var supplied = LedgerRecording.of(settings);
    if (provenance == null) {
      if (supplied == null || !keys.containsAll(List.of("compute", "pointClip", "policyClip")))
        throw new IOException(
            "Ledger fit lacks recording provenance. Supply noise.ledger.mode, compute, pointClip"
                + " and policyClip");
      return supplied;
    }
    if (!keys.isEmpty()
        && (supplied == null
            || keys.contains("compute") && !supplied.compute().equals(provenance.compute())
            || keys.contains("pointClip") && supplied.pointClip() != provenance.pointClip()
            || keys.contains("policyClip") && supplied.policyClip() != provenance.policyClip()))
      throw new IOException("Ledger recording settings differ from the fit provenance");
    return provenance;
  }
}
