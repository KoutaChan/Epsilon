package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.testng.annotations.Test;

/** 学習のロック。同じ JVM の中で確かめてもロックは外れず、ほかのプロセスからは取られているように見える。外した後は、ほかのプロセスが取れる。 */
public class TrainLockTest {

  @Test
  public void checkingInsideTheJvmKeepsTheLockForOtherProcesses() throws Exception {
    Path file = Files.createTempDirectory("train-lock").resolve(".train.lock");
    try (TrainLock held = TrainLock.hold(file)) {
      for (int check = 0; check < 3; check++) {
        assertTrue(TrainLock.running(file));
      }
      assertEquals(probe(file, "running"), "true");
      assertEquals(probe(file, "hold"), "refused: Another JVM holds " + file.toAbsolutePath().normalize());
    }
    assertEquals(probe(file, "running"), "false");
    assertEquals(probe(file, "hold"), "held");
  }

  /** 別の JVM でロックを確かめる（{@link Probe}）。 */
  private static String probe(Path file, String mode) throws IOException, InterruptedException {
    String java = ProcessHandle.current().info().command().orElseThrow();
    Process process =
        new ProcessBuilder(
                List.of(
                    java,
                    "-Xmx64m",
                    "-cp",
                    System.getProperty("java.class.path"),
                    Probe.class.getName(),
                    file.toString(),
                    mode))
            .redirectErrorStream(true)
            .start();
    if (!process.waitFor(60, TimeUnit.SECONDS)) {
      process.destroyForcibly();
      throw new AssertionError("The probe did not finish");
    }
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    String[] lines = output.strip().split("\n");
    return lines[lines.length - 1].strip();
  }

  /** 別の JVM で動かす確かめ。結果を最後の行に書く。 */
  public static final class Probe {

    private Probe() {}

    public static void main(String[] args) throws IOException {
      Path file = Path.of(args[0]);
      if (args[1].equals("running")) {
        System.out.println(TrainLock.running(file));
        return;
      }
      try (TrainLock lock = TrainLock.hold(file, Duration.ofMillis(300))) {
        System.out.println("held");
      } catch (IOException refused) {
        System.out.println("refused: " + refused.getMessage());
      }
    }
  }
}
