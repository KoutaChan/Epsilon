package com.epsilon.classic.learn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 途中の状態: 書いて読むと進み具合とテンソルが元に戻り、新しい区切りを書くと古いテンソルのファイルは消える。状態の無いディレクトリは null。
 */
public class PretrainStateTest {

  @Test
  public void writtenStateReadsBackAndOldTensorsGo() throws Exception {
    Path directory = Files.createTempDirectory("pretrain-state");
    try {
      Assert.assertNull(PretrainState.progress(directory));
      PretrainState.Progress first = progress(3, List.of());
      PretrainState.Tensors withoutBest = tensors(11, null);
      PretrainState.write(directory, first, withoutBest);
      Assert.assertEquals(PretrainState.progress(directory), first);
      assertSame(PretrainState.tensors(directory, first), withoutBest);

      PretrainState.Progress second =
          progress(4, List.of(new Pretrainer.Epoch(0, 17, 1.25, 0.5, 0.875, 2.0 / 3, 1.5)));
      PretrainState.Tensors withBest = tensors(17, new float[][] {{0.25f, -0.5f, 1e-7f}, {Float.MIN_VALUE}});
      PretrainState.write(directory, second, withBest);
      Assert.assertEquals(PretrainState.progress(directory), second);
      assertSame(PretrainState.tensors(directory, second), withBest);
      try (Stream<Path> listed = Files.list(directory)) {
        Assert.assertEquals(
            listed.map(path -> path.getFileName().toString()).sorted().toList(),
            List.of("progress.json", "tensors-00004.bin"));
      }
    } finally {
      try (Stream<Path> paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
          Files.deleteIfExists(path);
        }
      }
    }
  }

  private static PretrainState.Progress progress(int segment, List<Pretrainer.Epoch> epochs) {
    return new PretrainState.Progress(
        new Pretrainer.Settings(3, 1_024, 2e-4, 0.01, 0.5, 2.0, 10, 7), "NANO", 1_000, 100, new Pretrainer.Baselines(0.2, 1.7, 0.55), epochs,
        epochs.size() - 1, epochs.size(), 5, 0.1 + 0.2, 5, 17 + segment, segment);
  }

  private static PretrainState.Tensors tensors(long adamSteps, float[][] best) {
    return new PretrainState.Tensors(
        List.of("a_weight", "b_gamma"), adamSteps,
        new float[][] {{1f, 2f, -3f}, {Float.NaN}},
        new float[][] {{0.1f, 0.2f, 0.3f}, {-0f}},
        new float[][] {{1e-30f, 4f, 5f}, {Float.MAX_VALUE}},
        best);
  }

  private static void assertSame(PretrainState.Tensors actual, PretrainState.Tensors expected) {
    Assert.assertEquals(actual.names(), expected.names());
    Assert.assertEquals(actual.adamSteps(), expected.adamSteps());
    assertBits(actual.weights(), expected.weights());
    assertBits(actual.means(), expected.means());
    assertBits(actual.variances(), expected.variances());
    if (expected.best() == null) {
      Assert.assertNull(actual.best());
    } else {
      assertBits(actual.best(), expected.best());
    }
  }

  private static void assertBits(float[][] actual, float[][] expected) {
    Assert.assertEquals(actual.length, expected.length);
    for (int index = 0; index < expected.length; index++) {
      Assert.assertEquals(actual[index].length, expected[index].length);
      for (int value = 0; value < expected[index].length; value++) {
        Assert.assertEquals(
            Float.floatToRawIntBits(actual[index][value]), Float.floatToRawIntBits(expected[index][value]),
            "tensor " + index + " value " + value);
      }
    }
  }
}
