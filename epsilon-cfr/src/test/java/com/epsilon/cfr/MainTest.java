package com.epsilon.cfr;

import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.config.settings.EpsilonSettings;
import com.epsilon.cfr.io.CfrFiles;
import com.epsilon.cfr.training.CfrRunState;
import com.epsilon.cfr.training.CfrRunStore;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MainTest {
  @Test
  public void configurationCommandCreatesValidatedDefaultsAndRejectsOverwrite() throws Exception {
    Path directory = Files.createTempDirectory("cfr-main-");
    Path settings = settings(directory);
    Path target = directory.resolve("config.toml");
    try (PrintStream output = new PrintStream(new ByteArrayOutputStream())) {
      Assert.assertEquals(
          run(settings, new String[] {"init-config", "--out", target.toString()}, output), 0);
      Assert.assertEquals(
          CfrConfig.load(EpsilonSettings.load(target, Map.of())).collection().mode(), "HAND_GRP");
      String original = Files.readString(target);
      Assert.assertTrue(original.contains("[epsilon.cfr"));
      Assert.assertEquals(
          run(settings, new String[] {"init-config", "--out", target.toString()}, output), 1);
      Assert.assertEquals(Files.readString(target), original);
    } finally {
      delete(directory);
    }
  }

  @Test
  public void helpUsesRegisteredCommandsAndTheCommonExitCodes() throws Exception {
    Path directory = Files.createTempDirectory("cfr-main-help-");
    Path settings = settings(directory);
    try {
      for (String[] arguments :
          List.of(
              new String[0],
              new String[] {"help"},
              new String[] {"--help"},
              new String[] {"-h"},
              new String[] {"unknown"})) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (PrintStream output = new PrintStream(bytes, true, StandardCharsets.UTF_8)) {
          int expected = arguments.length > 0 && arguments[0].equals("unknown") ? 2 : 0;
          Assert.assertEquals(run(settings, arguments, output), expected);
          Assert.assertEquals(
              bytes.toString(StandardCharsets.UTF_8).lines().map(String::strip).toList(),
              List.of(
                  "Common option: --settings <file> <command> [arguments]",
                  "train --run DIR",
                  "resume --run DIR",
                  "init-config --out FILE",
                  "status --run DIR",
                  "verify --run DIR",
                  "export --run DIR --out DIR [--selection champion|average]"));
        }
      }
    } finally {
      delete(directory);
    }
  }

  @Test
  public void invalidOptionsFailBeforeCreatingOrOpeningTheRun() throws Exception {
    Path directory = Files.createTempDirectory("cfr-main-options-");
    Path settings = settings(directory);
    Path run = directory.resolve("run");
    Path config = directory.resolve("config.json");
    Path destination = directory.resolve("deployment");
    List<String[]> invalid =
        List.of(
            new String[] {"train", "--run", run.toString(), "--unexpected", config.toString()},
            new String[] {"train", "--run", run.toString(), "--run", run.toString()},
            new String[] {"train", "--run", run.toString(), "--config", config.toString()},
            new String[] {"train", "--run", run.toString(), "--nano-checkpoint", "nano"},
            new String[] {"resume", "--run", ""},
            new String[] {"init-config", "--out", config.toString(), "--unexpected", "1"},
            new String[] {"status", "run", run.toString()},
            new String[] {"verify", "--unexpected", run.toString()},
            new String[] {
              "export", "--run", run.toString(), "--out", destination.toString(), "--selection"
            },
            new String[] {
              "export",
              "--run",
              run.toString(),
              "--out",
              destination.toString(),
              "--unexpected",
              "1"
            },
            new String[] {
              "export",
              "--run",
              run.toString(),
              "--out",
              destination.toString(),
              "--selection",
              "invalid"
            });
    try (PrintStream output = new PrintStream(new ByteArrayOutputStream())) {
      for (String[] arguments : invalid) {
        Assert.assertEquals(run(settings, arguments, output), 1, String.join(" ", arguments));
      }
      Assert.assertFalse(Files.exists(run));
      Assert.assertFalse(Files.exists(config));
      Assert.assertFalse(Files.exists(destination));
    } finally {
      delete(directory);
    }
  }

  @Test
  public void savedRunsIgnoreUnrelatedStartupTrainingValues() throws Exception {
    Path directory = Files.createTempDirectory("cfr-main-saved-");
    Path settings = settings(directory);
    Files.writeString(settings, "[epsilon.cfr.network]\nhidden = 1\n");
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> CfrConfig.load(EpsilonSettings.load(settings, Map.of())));
    Path run = directory.resolve("run");
    try {
      CfrRunState expected;
      try (CfrRunStore store = new CfrRunStore(run)) {
        CfrConfig.defaults().write(run.resolve("config.json"));
        Path model = store.newArtifact("model", 0);
        Files.write(model.resolve("weights.bin"), new byte[] {1, 2, 3});
        CfrRunStore.sealArtifact(model);
        expected =
            new CfrRunState(
                2,
                "existing",
                0,
                1,
                CfrRunState.Phase.COMPLETE,
                0,
                0,
                store.relative(model),
                store.relative(model),
                null,
                0,
                null,
                "disabled",
                CfrFiles.sha256(run.resolve("config.json")));
        store.publish(expected);
      }
      byte[] original = Files.readAllBytes(run.resolve("config.json"));
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      try (PrintStream output = new PrintStream(bytes)) {
        for (String command : List.of("status", "verify", "resume")) {
          Assert.assertEquals(
              run(settings, new String[] {command, "--run", run.toString()}, output), 0, command);
          Assert.assertEquals(CfrRunStore.inspect(run), expected);
          Assert.assertEquals(Files.readAllBytes(run.resolve("config.json")), original);
        }
      }
      Assert.assertTrue(
          bytes.toString(StandardCharsets.UTF_8).contains("\"solveId\": \"existing\""));
    } finally {
      delete(directory);
    }
  }

  private static Path settings(Path directory) throws IOException {
    return Files.writeString(directory.resolve("startup.toml"), "# Use bundled defaults.\n");
  }

  private static int run(Path settings, String[] arguments, PrintStream output) throws IOException {
    String[] startup = new String[arguments.length + 2];
    startup[0] = "--settings";
    startup[1] = settings.toString();
    System.arraycopy(arguments, 0, startup, 2, arguments.length);
    return Main.run(startup, output);
  }

  private static void delete(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
