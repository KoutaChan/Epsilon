package com.epsilon.classic;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotEquals;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * テスト用の乱数のチェックポイントがシードだけで決まり（同じシードなら中身の SHA-256 が同じ）、系列から開けることを確かめる。
 */
public class RandomCheckpointsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final PolicyOptions CPU = PolicyOptions.defaults();

  @Test(groups = "native")
  public void aSeedDeterminesTheCheckpoint() throws IOException {
    Path root = Files.createTempDirectory("random-checkpoints");
    try {
      for (Architecture architecture : Architecture.values()) {
        Path first = root.resolve(architecture + "-first");
        Path again = root.resolve(architecture + "-again");
        Path other = root.resolve(architecture + "-other");
        RandomCheckpoints.write(first, architecture, VARIANT, 5);
        RandomCheckpoints.write(again, architecture, VARIANT, 5);
        RandomCheckpoints.write(other, architecture, VARIANT, 6);
        assertEquals(CheckpointFiles.sha256(again), CheckpointFiles.sha256(first), architecture.toString());
        assertNotEquals(CheckpointFiles.sha256(other), CheckpointFiles.sha256(first), architecture.toString());
        try (Policy policy = ModelFamily.of(architecture.series()).open(first, VARIANT, CPU)) {
          assertEquals(policy.label(), architecture.series().prefix() + ":" + first.getFileName());
        }
      }
    } finally {
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }
}
