package com.epsilon.workflow;

import com.epsilon.config.settings.SettingsLoader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CommandLineTest {
  @Test
  public void helpAndUnknownCommandsUseTheProvidedOutputWithoutSettingsInstructions() {
    AtomicInteger calls = new AtomicInteger();
    WorkflowRegistry workflows = registry(arguments -> calls.incrementAndGet() > 0);
    for (String[] arguments :
        List.of(
            new String[0], new String[] {"help"}, new String[] {"--help"}, new String[] {"-h"})) {
      ByteArrayOutputStream bytes = new ByteArrayOutputStream();
      int status = CommandLine.run(arguments, workflows, new PrintStream(bytes));
      Assert.assertEquals(status, 0);
      Assert.assertEquals(bytes.toString(StandardCharsets.UTF_8).strip(), "test [value]");
    }
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    Assert.assertEquals(
        CommandLine.run(new String[] {"unknown"}, workflows, new PrintStream(bytes)), 2);
    Assert.assertTrue(bytes.toString(StandardCharsets.UTF_8).contains("test [value]"));
    Assert.assertEquals(calls.get(), 0);
  }

  @Test
  public void settingsFreeExecutionUsesTheRegistryAndDoesNotWriteHelp() {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    WorkflowRegistry workflows =
        registry(
            arguments -> {
              Assert.assertTrue(arguments.settings().values().isEmpty());
              Assert.assertEquals(arguments.text(0, "value"), "kept");
              return true;
            });
    Assert.assertEquals(
        CommandLine.run(new String[] {"test", "kept"}, workflows, new PrintStream(bytes)), 0);
    Assert.assertEquals(bytes.size(), 0);
  }

  @Test
  public void existingSettingsArePassedThroughUnchanged() {
    SettingsLoader settings = SettingsLoader.of(Map.of("marker", "configured"));
    WorkflowRegistry workflows =
        registry(
            arguments -> {
              Assert.assertSame(arguments.settings(), settings);
              return true;
            });
    Assert.assertEquals(CommandLine.run(new String[] {"test"}, workflows, settings), 0);
  }

  @Test
  public void explicitOutputRetainsSettingsAndTheirHelpInstructions() {
    SettingsLoader settings = SettingsLoader.of(Map.of("marker", "configured"));
    WorkflowRegistry workflows =
        registry(
            arguments -> {
              Assert.assertSame(arguments.settings(), settings);
              return true;
            });
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (PrintStream output = new PrintStream(bytes)) {
      Assert.assertEquals(CommandLine.run(new String[] {"test"}, workflows, settings, output), 0);
      Assert.assertEquals(bytes.size(), 0);
      Assert.assertEquals(CommandLine.run(new String[] {"help"}, workflows, settings, output), 0);
      Assert.assertEquals(
          bytes.toString(StandardCharsets.UTF_8).lines().map(String::strip).toList(),
          List.of("Common option: --settings <file> <command> [arguments]", "test [value]"));
    }
  }

  @Test
  public void unsuccessfulResultsAndExceptionsUseTheCommonFailureCode() {
    PrintStream output = new PrintStream(new ByteArrayOutputStream());
    Assert.assertEquals(
        CommandLine.run(new String[] {"test"}, registry(arguments -> false), output), 1);
    Assert.assertEquals(
        CommandLine.run(
            new String[] {"test"},
            registry(
                arguments -> {
                  throw new IOException("test failure");
                }),
            output),
        1);
    AtomicInteger calls = new AtomicInteger();
    Assert.assertEquals(
        CommandLine.run(
            new String[] {"test", "extra", "extra"},
            registry(arguments -> calls.incrementAndGet() > 0),
            output),
        1);
    Assert.assertEquals(calls.get(), 0);
  }

  @Test
  public void interruptionUsesCode130AndRestoresTheInterruptFlag() {
    try {
      Assert.assertEquals(
          CommandLine.run(
              new String[] {"test"},
              registry(
                  arguments -> {
                    throw new InterruptedException("test interruption");
                  }),
              new PrintStream(new ByteArrayOutputStream())),
          130);
      Assert.assertTrue(Thread.currentThread().isInterrupted());
    } finally {
      Thread.interrupted();
    }
  }

  private static WorkflowRegistry registry(Execution execution) {
    return WorkflowRegistry.of(
        List.of(
            new CommandWorkflow<Boolean>() {
              @Override
              public WorkflowDefinition definition() {
                return WorkflowDefinition.range("test", WorkflowKind.AUDIT, "test [value]", 0, 1);
              }

              @Override
              public Boolean execute(WorkflowArguments arguments) throws Exception {
                return execution.run(arguments);
              }

              @Override
              public String summarize(Boolean result) {
                return "result=" + result;
              }

              @Override
              public boolean successful(Boolean result) {
                return result;
              }
            }));
  }

  @FunctionalInterface
  private interface Execution {
    boolean run(WorkflowArguments arguments) throws Exception;
  }
}
