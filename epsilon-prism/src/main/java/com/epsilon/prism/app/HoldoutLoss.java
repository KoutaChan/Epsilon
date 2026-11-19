package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.runtime.RuntimeSettings;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.GameEstimate;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

@Named(
    value = "holdout-loss",
    doc = "Measure raw label value loss and game-clustered standard errors")
public record HoldoutLoss(
    @Key(doc = "Run containing the cohort", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Cohort number", presence = Presence.REQUIRED) @Range(min = 1) int cohort,
    @Key(doc = "Version directories", presence = Presence.REQUIRED) List<Path> versions,
    @Key(doc = "Report directory", presence = Presence.REQUIRED) Path out)
    implements PrismCommand {

  record Difference(String reference, Double meanDifference, Double se, Double z) {
    Difference(Path reference, GameEstimate estimate) {
      this(
          reference.toString(),
          estimate.mean(),
          estimate.se(),
          estimate.se() == null || estimate.se() == 0 ? null : estimate.mean() / estimate.se());
    }
  }

  record Line(
      String version,
      int cohort,
      String group,
      GameEstimate loss,
      InferencePath backend,
      Precision precision,
      Difference difference) {}

  record Scores(Map<Long, GameEstimate.Game> games, byte[] digest) {
    List<GameEstimate.Game> differences(Scores reference) throws IOException {
      if (!Arrays.equals(digest, reference.digest)
          || !games.keySet().equals(reference.games.keySet()))
        throw new IOException("Paired versions must score identical cohort rows");
      var differences = new ArrayList<GameEstimate.Game>(games.size());
      games.forEach((id, game) -> differences.add(game.difference(reference.games.get(id))));
      return differences;
    }
  }

  @Override
  public Outcome run(Context<PrismSettings> context) throws IOException {
    var recorded =
        new SettingsRows(new RunDirectory(run))
            .read().reversed().stream()
                .filter(row -> row.from() <= cohort)
                .findFirst()
                .orElseThrow(() -> new IOException("No settings for cohort " + cohort));
    var layers = new ArrayList<Layer>();
    layers.add(recorded.layer());
    layers.addAll(context.layers());
    Settings.section(RuntimeSettings.class, "runtime", layers).value().apply();
    var inference = Settings.section(InferenceSettings.class, "inference", layers).value();
    var temperatures = temperatures(recorded.settings());
    var lines = new ArrayList<Line>();
    if (versions.isEmpty()) throw new IllegalArgumentException("At least one version is required");
    try (var store = new CohortStore();
        var networks = new ToolNetworks(inference)) {
      store.addCohort(CohortWriter.directory(run, cohort));
      int[] labels = store.select(cohort, cohort, i -> i.source().label() && i.measured() >= 2);
      if (labels.length == 0) throw new IOException("No contributing labels in cohort " + cohort);
      Scores reference = null;
      for (Path version : versions) {
        try (NetworkPolicy policy = networks.open(version)) {
          var execution = networks.execution(ToolNetworks.Role.HOLDOUT, version, policy);
          var scores = score(store, labels, policy, temperatures, versions.size() > 1);
          var differences = reference == null ? null : scores.differences(reference);
          for (DecisionClass group : temperatures.keySet()) {
            lines.add(line(execution, group, scores.games().values(), differences));
          }
          lines.add(line(execution, null, scores.games().values(), differences));
          if (reference == null) reference = scores;
        }
      }
    }
    StringBuilder tsv =
        new StringBuilder("version\tcohort\tgroup\trows\thanchan\tmean\tse\tbackend\tprecision\n");
    var diffTsv =
        new StringBuilder("version\treference\tgroup\trows\thanchan\tmeanDifference\tse\tz\n");
    for (Line line : lines) {
      var loss = line.loss();
      tsv.append(line.version())
          .append('\t')
          .append(cohort)
          .append('\t')
          .append(line.group())
          .append('\t')
          .append(loss.positions())
          .append('\t')
          .append(loss.hanchan())
          .append('\t')
          .append(loss.mean())
          .append('\t')
          .append(loss.se())
          .append('\t')
          .append(line.backend())
          .append('\t')
          .append(line.precision())
          .append('\n');
      var diff = line.difference();
      if (diff != null) {
        diffTsv
            .append(line.version())
            .append('\t')
            .append(diff.reference())
            .append('\t')
            .append(line.group())
            .append('\t')
            .append(loss.positions())
            .append('\t')
            .append(loss.hanchan())
            .append('\t')
            .append(diff.meanDifference())
            .append('\t')
            .append(diff.se())
            .append('\t')
            .append(diff.z())
            .append('\n');
      }
    }
    DurableFiles.write(
        out.resolve("holdout-loss.json"),
        new GsonBuilder().serializeNulls().setPrettyPrinting().create().toJson(lines));
    DurableFiles.write(out.resolve("holdout-loss.tsv"), tsv.toString());
    DurableFiles.write(out.resolve("holdout-loss-diff.tsv"), diffTsv.toString());
    return Outcome.ok("Wrote holdout loss to " + out);
  }

  private Line line(
      ToolNetworks.Execution network,
      DecisionClass group,
      Collection<GameEstimate.Game> games,
      Collection<GameEstimate.Game> differences) {
    return new Line(
        network.version(),
        cohort,
        group == null ? "ALL" : group.name(),
        GameEstimate.of(games, group),
        network.backend(),
        network.precision(),
        differences == null
            ? null
            : new Difference(versions.getFirst(), GameEstimate.of(differences, group)));
  }

  private static Map<DecisionClass, Float> temperatures(Node.Table recorded) throws IOException {
    if (!(recorded.get("fit") instanceof Node.Table fit)
        || !(fit.get("temperature") instanceof Node.Table table))
      throw new IOException("Missing fit.temperature in recorded settings");
    var result = new EnumMap<DecisionClass, Float>(DecisionClass.class);
    for (var entry : table.entries().entrySet()) {
      if (!(entry.getValue() instanceof Node.Numeral number))
        throw new IOException("Expected numeric temperature");
      float lambda = number.decimal().floatValue();
      if (!(lambda > 0 && Float.isFinite(lambda)))
        throw new IOException("Invalid temperature for " + entry.getKey());
      result.put(DecisionClass.valueOf(entry.getKey()), lambda);
    }
    return result;
  }

  private static Scores score(
      CohortStore store,
      int[] labels,
      NetworkPolicy policy,
      Map<DecisionClass, Float> temperatures,
      boolean paired)
      throws IOException {
    if (store.variant() != policy.format().variant())
      throw new IOException("Cohort and network variants differ");
    var games = new TreeMap<Long, GameEstimate.Game>();
    MessageDigest digest;
    try {
      digest = paired ? MessageDigest.getInstance("SHA-256") : null;
    } catch (NoSuchAlgorithmException unavailable) {
      throw new IllegalStateException("SHA-256 unavailable", unavailable);
    }
    int capacity = policy.maxBatch(0);
    var rows = new Row[capacity];
    var batch = new ObservationBatch(store.variant(), capacity);
    var reader = new Observation(store.variant());
    var requests = new ArrayList<PolicyRequest>(capacity);
    for (int from = 0; from < labels.length; ) {
      batch.clear();
      requests.clear();
      int size = Math.min(capacity, labels.length - from);
      for (int i = 0; i < size; i++) {
        Row row = store.read(labels[from++], digest);
        rows[i] = row;
        reader.bind(row.observation(), 0);
        int record = batch.add(reader);
        requests.add(
            new PolicyRequest(
                policy,
                row.gameId(),
                i,
                null,
                row.decisionClass(),
                false,
                0,
                batch,
                record,
                reader));
      }
      var admission = policy.tryAcquireIngress(0);
      while (!admission.acquired()) {
        admission.available().join();
        admission = policy.tryAcquireIngress(0);
      }
      policy.submit(requests, admission.ingress()).join();
      for (int i = 0; i < size; i++) {
        Row row = rows[i];
        double loss =
            PrismLoss.value(
                requests.get(i).scores, row.values(), temperatures.get(row.decisionClass()));
        games
            .computeIfAbsent(row.gameId(), id -> new GameEstimate.Game())
            .add(row.decisionClass(), loss);
      }
    }
    return new Scores(games, digest == null ? null : digest.digest());
  }
}
