package com.epsilon.config.settings;

import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 設定の部分上書き、スナップショットの独立性、接頭辞別の読み込みと不正値のエラー報告を検証する。 */
public class SettingsSnapshotTest {
  @SettingsPrefix("sample")
  record Options(
      @Default("32") @Positive @MultipleOf(8) int width,
      @Default("0.1") @Range(min = 0, max = 1) float rate,
      @Default("true") boolean enabled,
      @Default("checkpoint") String path) {}

  @Test
  public void capturesOverridesWithoutSharingTheCallersMutableMap() {
    var values = new HashMap<>(Map.of("sample.width", "64"));
    var snapshot = SettingsLoader.of(values);
    values.put("sample.width", "128");
    Assert.assertEquals(snapshot.bind(Options.class).width(), 64);
    Assert.expectThrows(
        UnsupportedOperationException.class, () -> snapshot.values().put("sample.width", "256"));
    Assert.assertEquals(SettingsLoader.of(values).bind(Options.class).width(), 128);
  }

  @Test
  public void overlaysPartialFilesAndKeepsQuotedWindowsPathsIntact() throws Exception {
    var defaults =
        SettingsLoader.parse(
            "bundled", List.of("[sample]", "width = 48", "rate = 0.2", "enabled = true"), Map.of());
    var selected = Files.createTempFile("epsilon-settings-", ".toml");
    try {
      Files.writeString(
          selected, "[sample]\nwidth = 9_6 # explicit width\npath = 'C:\\models\\test#1'\n");
      var overrides = new HashMap<>(Map.of("sample.rate", "0.3"));
      var snapshot = SettingsLoader.load(defaults, selected, overrides);
      overrides.put("sample.rate", "0.4");
      Files.writeString(selected, "[sample]\nwidth = 128\n");
      Assert.assertEquals(
          snapshot.bind(Options.class), new Options(96, .3f, true, "C:\\models\\test#1"));
      Assert.assertSame(snapshot.bind(Options.class), snapshot.bind(Options.class));
      Assert.assertEquals(defaults.bind(Options.class), new Options(48, .2f, true, "checkpoint"));
    } finally {
      Files.delete(selected);
    }
  }

  @Test
  public void programmaticOverridesPreserveTheDefaultsSnapshot() throws Exception {
    var defaults = SettingsLoader.of(Map.of("sample.width", "48", "sample.rate", "0.2"));
    var overrides = new HashMap<>(Map.of("sample.width", "96"));
    var snapshot = SettingsLoader.load(defaults, null, overrides);
    overrides.put("sample.width", "128");
    Assert.assertEquals(snapshot.bind(Options.class), new Options(96, .2f, true, "checkpoint"));
    Assert.assertEquals(defaults.bind(Options.class), new Options(48, .2f, true, "checkpoint"));
  }

  @Test
  public void bindsTheSameRecordAtSeparatePrefixesWithoutChangingTheDefaultCache() {
    var snapshot =
        SettingsLoader.of(
            Map.of(
                "sample.width", "48",
                "seat0.width", "64",
                "seat0.rate", "0.2",
                "seat1.width", "96",
                "seat1.rate", "0.3"));
    Options cached = snapshot.bind(Options.class);
    Options first = snapshot.bind(Options.class, "seat0");
    Options second = snapshot.bind(Options.class, "seat1");

    Assert.assertEquals(first, new Options(64, .2f, true, "checkpoint"));
    Assert.assertEquals(second, new Options(96, .3f, true, "checkpoint"));
    Assert.assertEquals(cached, new Options(48, .1f, true, "checkpoint"));
    Assert.assertSame(snapshot.bind(Options.class), cached);
    Assert.assertNotSame(snapshot.bind(Options.class, "seat0"), first);
  }

  @Test
  public void explicitPrefixReportsInvalidValuesUnderTheSelectedPrefix() {
    var snapshot = SettingsLoader.of(Map.of("seat1.width", "17"));
    var failure =
        Assert.expectThrows(
            IllegalArgumentException.class, () -> snapshot.bind(Options.class, "seat1"));
    Assert.assertTrue(failure.getMessage().contains("seat1.width"));
  }

  @DataProvider(name = "invalid")
  public Object[][] invalid() {
    return new Object[][] {
      {"sample.width", "0"},
      {"sample.width", "17"},
      {"sample.rate", "NaN"},
      {"sample.rate", "1.1"},
      {"sample.enabled", "maybe"}
    };
  }

  @Test(dataProvider = "invalid")
  public void rejectsInvalidExternalValuesAtBinding(String key, String value) {
    var failure =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> SettingsLoader.of(Map.of(key, value)).bind(Options.class));
    Assert.assertTrue(failure.getMessage().contains(key));
  }

  @Test
  public void reportsFileAndLineForMalformedInput() {
    var failure =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> SettingsLoader.parse("tiny.toml", List.of("[sample]", "invalid"), Map.of()));
    Assert.assertTrue(failure.getMessage().contains("tiny.toml at line 2"));
  }
}
