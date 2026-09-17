package com.epsilon.workflow;

import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 明示指定した設定ファイルを、既定ファイルで暗黙に置き換えないことを検証する。 */
public class StartupSettingsTest {
  @Test
  public void explicitMissingPathIsSelectedWithoutGeneratingADefault() throws Exception {
    Path directory = Files.createTempDirectory("startup-settings-");
    Path selected = directory.resolve("selected.toml");
    try {
      StartupSettings startup =
          StartupSettings.read(
              "epsilon-test",
              "missing-default-settings.toml",
              new String[] {"--settings", selected.toString(), "status"});
      Assert.assertEquals(startup.path(), selected);
      Assert.assertEquals(startup.arguments(), new String[] {"status"});
      Assert.assertFalse(Files.exists(selected));
    } finally {
      Files.deleteIfExists(selected);
      Files.delete(directory);
    }
  }
}
