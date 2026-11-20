package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.TimeUnit;
import org.testng.annotations.Test;

/** 同じ JVM からロックを確認した後も、解放するまで他のプロセスから取得できないことを確かめる。 */
public class TrainLockTest {
  private static final int AVAILABLE = 0;
  private static final int IN_USE = 10;
  private static final int REFUSED = 11;

  @Test
  public void checkingInsideTheJvmKeepsTheLockForOtherProcesses() throws Exception {
    Path directory = Files.createTempDirectory("train-lock");
    Path file = directory.resolve(".train.lock");
    Path reason = directory.resolve("refusal.txt");
    try {
      try (TrainLock held = TrainLock.hold(file)) {
        for (int check = 0; check < 3; check++) {
          assertTrue(TrainLock.running(file));
        }
        assertEquals(probe(file, "running", reason), IN_USE);
        assertEquals(probe(file, "hold", reason), REFUSED);
        assertEquals(
            Files.readString(reason, StandardCharsets.UTF_8),
            "Another JVM holds " + file.toAbsolutePath().normalize());
      }
      assertEquals(probe(file, "running", reason), AVAILABLE);
      assertEquals(probe(file, "hold", reason), AVAILABLE);
    } finally {
      Files.deleteIfExists(reason);
      Files.deleteIfExists(file);
      Files.delete(directory);
    }
  }

  /** 別の JVM でロックを調べ、終了コードを返す。 */
  private static int probe(Path file, String mode, Path reason)
      throws IOException, InterruptedException {
    String java = ProcessHandle.current().info().command().orElseThrow();
    Process process =
        new ProcessBuilder(
                java,
                "-Xmx64m",
                "-cp",
                System.getProperty("java.class.path"),
                Probe.class.getName(),
                file.toString(),
                mode,
                reason.toString())
            .redirectOutput(ProcessBuilder.Redirect.DISCARD)
            .redirectError(ProcessBuilder.Redirect.INHERIT)
            .start();
    try {
      assertTrue(process.waitFor(60, TimeUnit.SECONDS), "The probe did not finish");
      return process.exitValue();
    } finally {
      if (process.isAlive()) {
        process.destroyForcibly().waitFor();
      }
    }
  }

  /** 別の JVM からロックの状態か取得結果を終了コードで返す。 */
  public static final class Probe {
    private Probe() {}

    public static void main(String[] args) throws IOException {
      Path file = Path.of(args[0]);
      int result;
      if (args[1].equals("running")) {
        result = TrainLock.running(file) ? IN_USE : AVAILABLE;
      } else {
        try (TrainLock lock = TrainLock.hold(file, Duration.ofMillis(300))) {
          result = AVAILABLE;
        } catch (IOException failure) {
          if (!("Another JVM holds " + file.toAbsolutePath().normalize())
              .equals(failure.getMessage())) {
            throw failure;
          }
          Files.writeString(Path.of(args[2]), failure.getMessage(), StandardCharsets.UTF_8);
          result = REFUSED;
        }
      }
      System.exit(result);
    }
  }
}
