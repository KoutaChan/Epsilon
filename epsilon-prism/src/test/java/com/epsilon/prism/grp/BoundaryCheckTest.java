package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 境目の点検: 見本は種で決まる最大 {@value BoundaryCheck#SAMPLES} 個の局末で、ファイルの往復はビットで同じ。同じ Φ は合い、{@value
 * BoundaryCheck#TOLERANCE} を越えるずれと有限でない値は合わない。学習の JVM の GrpPhi の経路は、maxBatch 行ずつ直に通した値と合う。
 */
public class BoundaryCheckTest {

  @Test
  public void samplesAreSeededAndRoundTripBitwise() throws Exception {
    Path directory = Files.createTempDirectory("prism-boundary");
    try {
      List<RecordedGame> games = GrpFixtures.games(0, 900, 11);
      int total = games.stream().mapToInt(RecordedGame::continuingHandEnds).sum();
      Assert.assertTrue(total > BoundaryCheck.SAMPLES);
      HandEndPhi score = ScorePhi.offline(PhiEntryKind.SCORE_RANK, GrpFixtures.PROFILE);
      BoundaryCheck sample = BoundaryCheck.sample(games, score, 5);
      Assert.assertEquals(sample.inputs().size(), BoundaryCheck.SAMPLES);
      BoundaryCheck again = BoundaryCheck.sample(games, score, 5);
      for (int i = 0; i < BoundaryCheck.SAMPLES; i++)
        Assert.assertEquals(again.inputs().get(i).prefix(), sample.inputs().get(i).prefix());
      List<HandEndInput> other = BoundaryCheck.inputs(games, 6);
      int differ = 0;
      for (int i = 0; i < BoundaryCheck.SAMPLES; i++)
        if (!Arrays.equals(other.get(i).prefix(), sample.inputs().get(i).prefix())) differ++;
      Assert.assertTrue(differ > 0, "another seed picks other hand ends");
      // 局末が見本の数より少なければ全てを使う
      List<RecordedGame> few = games.subList(0, 3);
      Assert.assertEquals(
          BoundaryCheck.inputs(few, 1).size(), few.stream().mapToInt(RecordedGame::continuingHandEnds).sum());
      Path file = directory.resolve("boundary").resolve("g009-a1.bin");
      sample.write(file);
      BoundaryCheck read = BoundaryCheck.read(file);
      Assert.assertEquals(read.inputs().size(), sample.inputs().size());
      for (int i = 0; i < sample.inputs().size(); i++) {
        Assert.assertEquals(read.inputs().get(i).prefix(), sample.inputs().get(i).prefix());
        Assert.assertEquals(read.inputs().get(i).next().scores(), sample.inputs().get(i).next().scores());
        Assert.assertEquals(PhiResolverTest.bits(read.utilities()[i]), PhiResolverTest.bits(sample.utilities()[i]));
      }
      Assert.assertTrue(read.matches(PhiEntry.scoreRank(), GrpFixtures.PROFILE));
      read.utilities()[17][2] += 2 * BoundaryCheck.TOLERANCE;
      Assert.assertFalse(read.matches(PhiEntry.scoreRank(), GrpFixtures.PROFILE));
      read.utilities()[17][2] = Double.NaN;
      Assert.assertFalse(read.matches(PhiEntry.scoreRank(), GrpFixtures.PROFILE));
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  @Test(timeOut = 300_000)
  public void theLabelPathReproducesADirectSample() throws Exception {
    Path directory = Files.createTempDirectory("prism-boundary");
    try {
      Path checkpoint = GrpFixtures.checkpoint(directory.resolve("decision"));
      List<RecordedGame> games = GrpFixtures.games(0, 200, 13);
      try (Model model = EpsilonGrpCheckpointManager.load(checkpoint, Device.cpu())) {
        var inference = new EpsilonGrpInference(model, PhiResolverTest.MAX_BATCH);
        BoundaryCheck sample = BoundaryCheck.sample(games, GrpFixtures.direct(inference), 3);
        Assert.assertTrue(sample.matches(inference, GrpFixtures.PROFILE));
        sample.utilities()[0][0] += 1e-3;
        Assert.assertFalse(sample.matches(inference, GrpFixtures.PROFILE));
      }
    } finally {
      Snapshot.deleteTree(directory);
    }
  }
}
