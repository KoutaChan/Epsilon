package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsLoader;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class WorkflowOptionsTest {
  @Test
  public void namedOptionsPreserveValuesAndAllowEitherOrder() {
    WorkflowOptions options =
        arguments("--selection", "average", "--run", "path with spaces").namedOptions();
    Assert.assertEquals(options.path("run"), Path.of("path with spaces"));
    Assert.assertEquals(options.optionalText("selection", "champion"), "average");
    Assert.assertEquals(options.optionalText("missing", "default"), "default");
    options.requireEmpty();
  }

  @Test
  public void missingAndUnknownOptionsFailBeforeExecution() {
    WorkflowOptions missing = arguments("--selection", "average").namedOptions();
    IllegalArgumentException missingFailure =
        Assert.expectThrows(IllegalArgumentException.class, () -> missing.path("run"));
    Assert.assertTrue(missingFailure.getMessage().contains("requires --run"));

    WorkflowOptions unknown = arguments("--run", "known", "--unexpected", "value").namedOptions();
    Assert.assertEquals(unknown.path("run"), Path.of("known"));
    IllegalArgumentException unknownFailure =
        Assert.expectThrows(IllegalArgumentException.class, unknown::requireEmpty);
    Assert.assertTrue(unknownFailure.getMessage().contains("unexpected"));
  }

  @Test
  public void malformedNamedOptionsAreRejected() {
    String[][] malformed = {
      {"--run", "first", "--run", "second"},
      {"--run"},
      {"--run", "--selection", "average"},
      {"--run", " "},
      {"run", "value"},
      {"--", "value"},
      {"-- ", "value"}
    };
    for (String[] values : malformed) {
      Assert.expectThrows(IllegalArgumentException.class, () -> arguments(values).namedOptions());
    }
  }

  @Test
  public void namedParsingDoesNotMutatePositionalArguments() {
    WorkflowArguments arguments = arguments("--run", "checkpoint");
    arguments.namedOptions().path("run");
    Assert.assertEquals(arguments.text(0, "option"), "--run");
    Assert.assertEquals(arguments.path(1, "path"), Path.of("checkpoint"));
    Assert.assertEquals(arguments.namedOptions().path("run"), Path.of("checkpoint"));
  }

  private static WorkflowArguments arguments(String... values) {
    return new WorkflowArguments("test", values, SettingsLoader.of(Map.of()));
  }
}
