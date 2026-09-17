package com.epsilon.config.settings;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 既定ファイルの初回生成と、利用者が編集したファイルの保持を検証する。 */
public class SettingsFilesTest {
  private static final String RESOURCE = "config/settings/default-settings.toml";

  @DataProvider
  public Object[][] targetLocations() {
    return new Object[][] {{true}, {false}};
  }

  @Test(dataProvider = "targetLocations")
  public void createsExactDefaultAndKeepsUserEdits(boolean nestedDirectory) throws Exception {
    Path directory = Files.createTempDirectory("settings-files-");
    Path parent = directory.resolve("config");
    Path target =
        nestedDirectory
            ? parent.resolve("settings.toml")
            : Path.of(directory.getFileName() + ".toml");
    try (InputStream resource =
        SettingsFilesTest.class.getClassLoader().getResourceAsStream(RESOURCE)) {
      byte[] expected = resource.readAllBytes();
      SettingsFiles.createDefault(target, RESOURCE);
      Assert.assertEquals(Files.readAllBytes(target), expected);

      byte[] edited = "[sample]\nwidth = 96 # user override\n".getBytes(StandardCharsets.UTF_8);
      Files.write(target, edited);
      SettingsFiles.createDefault(target, "missing-default-settings.toml");
      Assert.assertEquals(Files.readAllBytes(target), edited);
    } finally {
      Files.deleteIfExists(target);
      Files.deleteIfExists(parent);
      Files.delete(directory);
    }
  }

  @Test
  public void missingResourceDoesNotCreateTheDestination() throws Exception {
    Path directory = Files.createTempDirectory("missing-settings-");
    Path parent = directory.resolve("config");
    Path target = parent.resolve("settings.toml");
    String missing = "config/settings/not-present.toml";
    try {
      IOException failure =
          Assert.expectThrows(
              IOException.class, () -> SettingsFiles.createDefault(target, missing));
      Assert.assertTrue(failure.getMessage().contains(missing));
      Assert.assertFalse(Files.exists(target));
      Assert.assertFalse(Files.exists(parent));
    } finally {
      Files.deleteIfExists(target);
      Files.deleteIfExists(parent);
      Files.delete(directory);
    }
  }
}
