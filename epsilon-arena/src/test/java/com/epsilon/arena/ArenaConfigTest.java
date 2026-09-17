package com.epsilon.arena;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 対戦設定の既定値、部分上書き、ファイルの扱い、系列別のパス解決と入力検証を確認する。 */
public class ArenaConfigTest {
  private static final String SEATS =
      """
      [seat0]
      series = "epsilon-nano"
      checkpoint = "nano/decision"
      [seat1]
      series = "epsilon-pico"
      checkpoint = "pico/decision"
      [seat2]
      series = "epsilon"
      checkpoint = "major/decision"
      [seat3]
      series = "epsilon"
      checkpoint = "major/decision"
      """;

  @Test
  public void partialArenaOverridesRetainDefaultsWithoutRewritingTheSelectedFile()
      throws Exception {
    String text = "[arena]\ngames = 12\n" + SEATS;
    Path config = writeConfig(text);
    try {
      RunSettings run = ArenaConfig.load(config).run();
      Assert.assertEquals(run.games(), 12);
      Assert.assertEquals(run.concurrentGames(), 128);
      Assert.assertEquals(
          run.workers(), Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
      Assert.assertEquals(Files.readString(config), text);
    } finally {
      Files.delete(config);
    }
  }

  @Test
  public void missingSeatIsNotFilledFromTheBundledExample() throws Exception {
    Path config = writeConfig(SEATS.substring(0, SEATS.indexOf("[seat3]")));
    try {
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> ArenaConfig.load(config));
      Assert.assertTrue(failure.getMessage().contains("seat3.series"));
    } finally {
      Files.delete(config);
    }
  }

  @Test
  public void unknownArenaKeyIsRejectedInsteadOfUsingTheDefault() throws Exception {
    Path config = writeConfig("[arena]\nconcurentGames = 4\n" + SEATS);
    try {
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> ArenaConfig.load(config));
      Assert.assertTrue(failure.getMessage().contains("arena.concurentGames"));
    } finally {
      Files.delete(config);
    }
  }

  @Test
  public void matchConfigKeepsSeriesSeparateAndResolvesOnlyKnownRelativePaths() throws Exception {
    Path config =
        writeConfig(
            SEATS
                + """
                [seat0.options]
                settings = "settings/../nano.toml"
                device = "cpu"
                epsilon.decision.inference.computePrecision = "BFLOAT16"
                custom.path = "raw/../left"
                """);
    try {
      ArenaConfig loaded = ArenaConfig.load(config);
      ModelSpec nano = loaded.seats().getFirst();
      Assert.assertEquals(nano.series(), "epsilon-nano");
      Assert.assertEquals(loaded.seats().get(1).series(), "epsilon-pico");
      Assert.assertEquals(nano.checkpoint(), config.getParent().resolve("nano/decision"));
      Assert.assertEquals(
          nano.options(),
          Map.of(
              "settings", config.getParent().resolve("nano.toml").toString(),
              "device", "cpu",
              "epsilon.decision.inference.computePrecision", "BFLOAT16",
              "custom.path", "raw/../left"));
      Assert.assertEquals(loaded.seats().get(2), loaded.seats().get(3));
    } finally {
      Files.delete(config);
    }
  }

  @Test
  public void missingExplicitFileFailsWithoutGeneratingIt() throws Exception {
    Path directory = Files.createTempDirectory("arena-missing-config");
    Path config = directory.resolve("arena.toml");
    try {
      Assert.expectThrows(IOException.class, () -> ArenaConfig.load(config));
      Assert.assertFalse(Files.exists(config));
    } finally {
      Files.deleteIfExists(config);
      Files.delete(directory);
    }
  }

  @DataProvider
  public Object[][] invalidRunSettings() {
    return new Object[][] {
      {"games = 6", "games"},
      {"workers = 0", "workers"},
      {"slotsPerDevice = 3\nreadyBatchesPerDevice = 2", "slotsPerDevice"}
    };
  }

  @Test(dataProvider = "invalidRunSettings")
  public void typedBindingPreservesRunConstraints(String settings, String invalidField)
      throws Exception {
    Path config = writeConfig("[arena]\n" + settings + "\n" + SEATS);
    try {
      IllegalArgumentException failure =
          Assert.expectThrows(IllegalArgumentException.class, () -> ArenaConfig.load(config));
      Assert.assertTrue(failure.getMessage().contains(invalidField));
    } finally {
      Files.delete(config);
    }
  }

  private static Path writeConfig(String text) throws IOException {
    Path config = Files.createTempFile("arena-config", ".toml");
    Files.writeString(config, text);
    return config;
  }
}
