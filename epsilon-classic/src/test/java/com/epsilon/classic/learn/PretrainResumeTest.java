package com.epsilon.classic.learn;

import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomCheckpoints;
import com.epsilon.classic.data.ClassicRows;
import com.epsilon.classic.data.LogRows;
import com.epsilon.classic.data.RowChunks;
import com.epsilon.classic.data.RuleGames;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 区切って学ぶ（CPU の FP32）。初めの区切りは最初のステップの前に止まり、基準の値だけを書く。その後は数ステップずつ区切って学び、
 * 1 エポックで終えた状態を 2 エポックに延ばす。それでも、区切らずに 2 エポック学んだときと、エポックごとの値（秒数を除く）・ステップ数・
 * 最後の重みがビットで同じになる。区切りで止めた呼び出しは finished が false。設定（エポック数を除く）か行が違う状態から続けようと
 * すると拒む。
 */
public class PretrainResumeTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int SEGMENT_STEPS = 9;

  /** 学習の結果と、学び終えたネットワークの重み。 */
  private record Outcome(Pretrainer.Result result, float[][] weights) {}

  @Test(groups = "native", timeOut = 1_200_000)
  public void segmentedTrainingMatchesAnUnbrokenRun() throws Exception {
    Path root = Files.createTempDirectory("pretrain-resume");
    try {
      Path checkpoint = root.resolve("start");
      RandomCheckpoints.write(checkpoint, Architecture.NANO, VARIANT, 11);
      List<ClassicRows> chunks = chunks();
      RowChunks rows = RowChunks.of(chunks);
      Pretrainer.Settings two = new Pretrainer.Settings(2, 64, 1e-3, 0.01, 0.5, 2.0, 10, 5);
      Pretrainer.Settings one = two.withEpochs(1);

      Outcome unbroken = train(checkpoint, rows, two, null, () -> false);
      Assert.assertTrue(unbroken.result().finished());
      Assert.assertEquals(unbroken.result().epochs().size(), 2);

      Path state = root.resolve("state");
      Outcome first = train(checkpoint, rows, one, state, pauseAfter(0));
      Assert.assertFalse(first.result().finished());
      Assert.assertEquals(first.result().steps(), 0);
      Assert.assertEquals(first.result().epochs(), List.of());
      Assert.assertEquals(first.result().best(), -1);
      Assert.assertEquals(first.result().baselines(), unbroken.result().baselines());

      int segments = 1;
      Outcome last = first;
      for (Pretrainer.Settings settings : List.of(one, two)) {
        do {
          last = train(checkpoint, rows, settings, state, pauseAfter(SEGMENT_STEPS));
          segments++;
          Assert.assertTrue(segments < 1_000, "no progress");
        } while (!last.result().finished());
        Assert.assertEquals(last.result().epochs().size(), settings.epochs());
      }
      Assert.assertTrue(segments > 4, "segments " + segments);

      Pretrainer.Result expected = unbroken.result();
      Pretrainer.Result actual = last.result();
      Assert.assertEquals(actual.steps(), expected.steps());
      Assert.assertEquals(actual.best(), expected.best());
      Assert.assertEquals(actual.trainRows(), expected.trainRows());
      Assert.assertEquals(actual.validationRows(), expected.validationRows());
      Assert.assertEquals(actual.baselines(), expected.baselines());
      for (int epoch = 0; epoch < expected.epochs().size(); epoch++) {
        Assert.assertEquals(withoutSeconds(actual.epochs().get(epoch)), withoutSeconds(expected.epochs().get(epoch)));
      }
      Assert.assertEquals(last.weights().length, unbroken.weights().length);
      for (int index = 0; index < unbroken.weights().length; index++) {
        Assert.assertTrue(Arrays.equals(last.weights()[index], unbroken.weights()[index]), "tensor " + index);
      }

      Pretrainer.Settings otherBatch = new Pretrainer.Settings(2, 32, 1e-3, 0.01, 0.5, 2.0, 10, 5);
      Assert.assertThrows(
          IllegalArgumentException.class, () -> train(checkpoint, rows, otherBatch, state, () -> false));
      RowChunks fewer = RowChunks.of(chunks.subList(0, 1));
      Assert.assertThrows(IllegalArgumentException.class, () -> train(checkpoint, fewer, two, state, () -> false));
    } finally {
      delete(root);
    }
  }

  /** 1 回の呼び出し。checkpoint の重みから始め、途中の状態があれば続きから学ぶ。 */
  private static Outcome train(
      Path checkpoint, RowChunks rows, Pretrainer.Settings settings, Path state, BooleanSupplier pause)
      throws IOException {
    try (NDManager manager = NDManager.newBaseManager("PyTorch")) {
      ClassicNetwork network = ClassicCheckpoint.read(checkpoint, VARIANT, manager);
      Pretrainer.Result result = new Pretrainer(network, manager, Precision.FLOAT32, rows, null, settings).train(state, pause);
      List<float[]> weights = new ArrayList<>();
      for (Pair<String, Parameter> pair : network.getParameters()) {
        weights.add(pair.getValue().getArray().toFloatArray());
      }
      return new Outcome(result, weights.toArray(float[][]::new));
    }
  }

  /** 各ステップの前に止めるかを聞かれると、初めの {@code steps} 回は「続ける」と答え、その後は「止める」と答える。 */
  private static BooleanSupplier pauseAfter(int steps) {
    int[] asked = {0};
    return () -> asked[0]++ >= steps;
  }

  private static Pretrainer.Epoch withoutSeconds(Pretrainer.Epoch epoch) {
    return new Pretrainer.Epoch(
        epoch.epoch(), epoch.steps(), epoch.trainLoss(), epoch.agreement(), epoch.nll(), epoch.valueLoss(), 0);
  }

  /** 学習用の半荘 2 つと検証用の半荘 1 つを、2 つの行のファイルに分けたもの（初めのファイルに両方の行がある）。 */
  private static List<ClassicRows> chunks() {
    List<RecordedGame> train = new ArrayList<>();
    RecordedGame validation = null;
    for (int index = 0; train.size() < 2 || validation == null; index++) {
      String id = "resume-" + index;
      boolean held = LogRows.validation(id, 0.1);
      if (held && validation == null) {
        validation = RuleGames.play(VARIANT, 3_000 + index, id);
      } else if (!held && train.size() < 2) {
        train.add(RuleGames.play(VARIANT, 3_000 + index, id));
      }
    }
    LogRows logs = new LogRows(VARIANT, 0.1);
    ClassicRows.Builder first = ClassicRows.builder(VARIANT);
    logs.add(train.get(0), first);
    logs.add(validation, first);
    ClassicRows.Builder second = ClassicRows.builder(VARIANT);
    logs.add(train.get(1), second);
    return List.of(first.build(), second.build());
  }

  private static void delete(Path root) throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
