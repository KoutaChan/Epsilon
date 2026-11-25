package com.epsilon.prism.app;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerCell;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.luck.LedgerReport;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.SlopeRefresh;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SplitDuelTest {
  @Test
  public void missingClassesUseThePooledParityCoefficients() throws Exception {
    var root = Files.createTempDirectory("split-ledger");
    try {
      var report =
          new LedgerReport.Full(
              10,
              20,
              0,
              segment(1),
              Map.of("THREAT", segment(3)),
              new LedgerReport.Bias(List.of()),
              0,
              new LedgerRecording(LedgerGroup.computed(), 12, 2));
      var path = root.resolve("ledger.json");
      Files.writeString(path, RecordFile.json(report));
      var saved = RecordFile.read(LedgerReport.Full.class, path).value();
      var beta =
          ((LedgerCorrection.ByClass)
                  LedgerCorrection.restore(saved, report.recording(), Variant.TENHOU_4P, null))
              .beta();
      int column = LedgerColumn.OPP_IMPROVE_1.ordinal();
      Assert.assertEquals(beta[0][DecisionClass.CALL.ordinal()][column], 1.0);
      Assert.assertEquals(beta[1][DecisionClass.CALL.ordinal()][column], -1.0);
      Assert.assertEquals(beta[1][DecisionClass.THREAT.ordinal()][column], -3.0);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void recordingMustBeKnownAndMatchTheFittedColumns() throws Exception {
    var fit = new LedgerRecording(LedgerGroup.computed(), 12, 2);
    var settings = new LedgerSettings.RecordOnly(fit.compute(), 12, 2, 1000, SlopeRefresh.DEPOSIT);
    var off = new LedgerSettings.Off();
    Assert.assertEquals(SplitDuel.recording(fit, off, List.of()), fit);
    Assert.expectThrows(IOException.class, () -> SplitDuel.recording(null, off, List.of()));
    var clips = Layer.parse("clips", Layer.Source.FILE, "[noise.ledger]\npointClip = 9");
    var mismatch = new LedgerSettings.RecordOnly(fit.compute(), 9, 2, 1000, SlopeRefresh.DEPOSIT);
    Assert.expectThrows(
        IOException.class, () -> SplitDuel.recording(fit, mismatch, List.of(clips)));
    Assert.expectThrows(
        IOException.class, () -> SplitDuel.recording(null, settings, List.of(clips)));
    var explicit =
        Layer.parse(
            "fit",
            Layer.Source.FILE,
            "[noise.ledger]\n"
                + "mode='RECORD'\n"
                + "compute=['IMPROVE','POINTS','DORA','URA','POLICY']\n"
                + "pointClip=12\n"
                + "policyClip=2");
    Assert.assertEquals(SplitDuel.recording(null, settings, List.of(explicit)), fit);
    var report =
        new LedgerReport.Full(
            10, 20, 0, segment(1), Map.of(), new LedgerReport.Bias(List.of()), 0, null);
    var missing = new LedgerRecording(Set.of(LedgerGroup.POINTS), 12, 2);
    Assert.expectThrows(
        IOException.class,
        () -> LedgerCorrection.restore(report, missing, Variant.TENHOU_4P, null));
  }

  private static LedgerReport.Segment segment(double scale) {
    var even = new LinkedHashMap<String, Double>();
    var odd = new LinkedHashMap<String, Double>();
    for (var column : LedgerColumn.values()) {
      even.put(column.reportName(), scale);
      odd.put(column.reportName(), -scale);
    }
    return new LedgerReport.Segment(
        10,
        20,
        null,
        null,
        null,
        null,
        Map.of(),
        even,
        List.of(new LedgerReport.Fold(null, even, false), new LedgerReport.Fold(null, odd, false)));
  }

  @Test
  public void eagerFrozenNetworksRepeatAcrossWorkerCounts() throws Exception {
    frozenComparison(
        "cpu", "FLOAT32", "EAGER", new PrismShape.V3(32, 128, 64, 128, 128, 128, true));
  }

  static void frozenComparison(String device, String precision, String backend, PrismShape shape)
      throws Exception {
    var root = Files.createTempDirectory("split-workers");
    try {
      var a = root.resolve("a");
      var b = root.resolve("b");
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(a, HipFixtures.network(manager, shape, 3), PolicyParams.of(1));
        Snapshot.write(b, HipFixtures.network(manager, shape, 7), PolicyParams.of(2));
      }
      int games = device.equals("cpu") ? 2 : 3;
      String serial = compare(root, a, b, device, precision, 1, games);
      Assert.assertEquals(compare(root, a, b, device, precision, 3, games), serial);
      var fields = overall(serial);
      Assert.assertTrue(Long.parseLong(fields.get("disagreements")) > 0);
      for (String role : List.of("a", "b", "mu")) {
        Assert.assertEquals(fields.get(role + "Backend"), backend);
        Assert.assertEquals(fields.get(role + "Precision"), precision);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void ledgerFlagChangesThePairedEstimateWithTheSameDisagreements() throws Exception {
    var root = Files.createTempDirectory("split-cells");
    try {
      var a = root.resolve("a");
      var b = root.resolve("b");
      var shape = new PrismShape.V2(32, 128, 64, 2);
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(a, HipFixtures.network(manager, shape, 3), PolicyParams.of(1));
        Snapshot.write(b, HipFixtures.network(manager, shape, 7), PolicyParams.of(2));
      }
      var raw = overall(compare(root, a, b, "cpu", "FLOAT32", 1));
      var cells = new LinkedHashMap<String, List<LedgerReport.Fold>>();
      for (DecisionClass group : DecisionClass.values()) {
        var partition = LedgerCell.of(Variant.TENHOU_4P);
        for (int i = 0; i < partition.count(); i++)
          cells.put(partition.name(group, i), segment(1 + i).folds());
      }
      var report =
          new LedgerReport.Full(
              10,
              20,
              0,
              segment(0),
              Map.of(),
              new LedgerReport.Bias(List.of()),
              0,
              new LedgerRecording(LedgerGroup.computed(), 32, 8),
              cells);
      var path = root.resolve("ledger.json");
      Files.writeString(path, RecordFile.json(report));
      var adjusted = overall(compare(root, a, b, "cpu", "FLOAT32", 1, "--ledger", path.toString()));
      Assert.assertEquals(adjusted.get("positions"), raw.get("positions"));
      Assert.assertEquals(adjusted.get("disagreements"), raw.get("disagreements"));
      Assert.assertTrue(Long.parseLong(raw.get("disagreements")) > 0);
      Assert.assertNotEquals(adjusted.get("mean"), raw.get("mean"));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  static Map<String, String> overall(String tsv) {
    String[] header = tsv.lines().findFirst().orElseThrow().split("\t");
    String[] values =
        tsv.lines().filter(s -> s.startsWith("ALL\t")).findFirst().orElseThrow().split("\t");
    var byName = new LinkedHashMap<String, String>();
    for (int i = 0; i < header.length; i++) byName.put(header[i], values[i]);
    return byName;
  }

  static String compare(
      Path root, Path a, Path b, String device, String precision, int workers, String... options)
      throws IOException {
    return compare(root, a, b, device, precision, workers, 3, options);
  }

  private static String compare(
      Path root,
      Path a,
      Path b,
      String device,
      String precision,
      int workers,
      int games,
      String... options)
      throws IOException {
    String[] args = {
      "split-duel",
      "--a",
      a.toString(),
      "--b",
      b.toString(),
      "--mu",
      a.toString(),
      "--out",
      root.toString(),
      "--q",
      "0.025",
      "--games",
      Integer.toString(games),
      "--set",
      "run.variant=TENHOU_4P",
      "--set",
      "run.seed=17",
      "--set",
      "inference.maxBatch=" + (workers == 1 ? 1 : 8),
      "--set",
      "table.workers=" + workers,
      "--set",
      "table.activeGames=" + games,
      "--set",
      "inference.devices=[" + device + "]",
      "--set",
      "inference.precision=" + precision
    };
    var command = new ArrayList<>(List.of(args));
    command.addAll(List.of(options));
    Assert.assertEquals(
        App.run(PrismApp.application(), command.toArray(String[]::new), System.out),
        Exit.OK.code());
    return Files.readString(root.resolve("split-duel.tsv"));
  }

  @Test
  public void identicalFrozenVersionsProduceZero() throws Exception {
    var root = Files.createTempDirectory("split-duel");
    try {
      var snapshot = root.resolve("version");
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        var shape = new PrismShape.V3(32, 128, 64, 128, 128, 128, true);
        Snapshot.write(snapshot, HipFixtures.network(manager, shape, 3), PolicyParams.of(1));
      }
      String version = snapshot.toString();
      var bytes = new ByteArrayOutputStream();
      try (var out = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
        String[] args = {
          "split-duel",
          "--a",
          version,
          "--b",
          version,
          "--mu",
          version,
          "--q",
          "0.02",
          "--games",
          "2",
          "--set",
          "run.variant=TENHOU_4P",
          "--set",
          "run.seed=17"
        };
        Assert.assertEquals(App.run(PrismApp.application(), args, out), Exit.OK.code());
      }
      String overall =
          bytes
              .toString(StandardCharsets.UTF_8)
              .lines()
              .filter(s -> s.startsWith("ALL\t"))
              .findFirst()
              .orElseThrow();
      String[] fields = overall.split("\t");
      Assert.assertTrue(Long.parseLong(fields[1]) > 0);
      Assert.assertEquals(fields[2], "0");
      Assert.assertEquals(Double.parseDouble(fields[5]), 0.0);
      Assert.assertEquals(Double.parseDouble(fields[6]), 0.0);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
