package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 半荘の記録のファイルを確かめる。書いて読めば元に戻る。途中で切れたファイルと、後ろに余計なバイトがあるファイルは、読むと
 * 例外を投げる。ヘッダーの半荘の数が中身より多いファイルも、読むと IOException を投げる。フィールドに入らない値は書かずに例外を
 * 投げ、元のファイルも変えない。局末から、終わった局と実際の順列を求められる。
 */
public class GameRecordCodecTest {

  private Path root;

  @BeforeMethod
  public void createRoot() throws IOException {
    root = Files.createTempDirectory("grp-records");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRoot() throws IOException {
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(path);
      }
    }
  }

  @Test
  public void cohortsRoundTrip() throws IOException {
    for (Variant variant : Variant.values()) {
      List<GameRecord> games = TrainFixtures.simulated(variant, 300, 1_000, 7);
      GameCohort cohort = new GameCohort(variant, 41, "g039-a1", games);
      Path file = root.resolve(variant + "/c41.bin.gz");
      GameRecordCodec.write(file, cohort);
      GameCohort read = GameRecordCodec.read(file);
      Assert.assertEquals(read, cohort);
      Assert.assertEquals(read.games(), games);
      Assert.assertFalse(Files.exists(file.resolveSibling("c41.bin.gz.tmp")));
      GameRecordCodec.write(file, cohort);
      Assert.assertEquals(GameRecordCodec.read(file), cohort);
    }
  }

  @Test
  public void emptyGamesAndCohortsRoundTrip() throws IOException {
    GameRecord flying = new GameRecord(Variant.TENHOU_3P, 5, List.of(), new int[] {70_000, 36_000, -1_000});
    GameCohort cohort = new GameCohort(Variant.TENHOU_3P, 3, "", List.of(flying));
    Path file = root.resolve("one.bin.gz");
    GameRecordCodec.write(file, cohort);
    Assert.assertEquals(GameRecordCodec.read(file), cohort);
    GameCohort empty = new GameCohort(Variant.TENHOU_4P, 0, "SCORE_RANK", List.of());
    GameRecordCodec.write(file, empty);
    Assert.assertEquals(GameRecordCodec.read(file), empty);
  }

  @Test
  public void truncatedFilesAndTrailingBytesAreRefused() throws IOException {
    GameCohort cohort = new GameCohort(
        Variant.TENHOU_4P, 9, "id", TrainFixtures.simulated(Variant.TENHOU_4P, 200, 0, 11));
    Path file = root.resolve("whole.bin.gz");
    GameRecordCodec.write(file, cohort);
    byte[] bytes = Files.readAllBytes(file);
    for (int cut : new int[] {bytes.length - 1, bytes.length - 8, bytes.length / 2, 20}) {
      Path truncated = root.resolve("cut-" + cut + ".bin.gz");
      Files.write(truncated, Arrays.copyOf(bytes, cut));
      Assert.assertThrows(IOException.class, () -> GameRecordCodec.read(truncated));
    }
    Path padded = root.resolve("padded.bin.gz");
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    try (GZIPOutputStream out = new GZIPOutputStream(raw)) {
      out.write(gunzip(bytes));
      out.write(0);
    }
    Files.write(padded, raw.toByteArray());
    Assert.assertThrows(IOException.class, () -> GameRecordCodec.read(padded));
  }

  @Test
  public void gameCountBeyondTheFileIsRefused() throws IOException {
    GameCohort cohort = new GameCohort(
        Variant.TENHOU_4P, 9, "id", TrainFixtures.simulated(Variant.TENHOU_4P, 20, 0, 11));
    Path file = root.resolve("whole.bin.gz");
    GameRecordCodec.write(file, cohort);
    byte[] content = gunzip(Files.readAllBytes(file));
    int countOffset = 2 + Variant.TENHOU_4P.name().length() + 4 + 2 + "id".length();
    Assert.assertEquals(ByteBuffer.wrap(content).getInt(countOffset), 20);
    ByteBuffer.wrap(content).putInt(countOffset, Integer.MAX_VALUE);
    Path inflated = root.resolve("inflated.bin.gz");
    try (GZIPOutputStream out = new GZIPOutputStream(Files.newOutputStream(inflated))) {
      out.write(content);
    }
    Assert.assertThrows(IOException.class, () -> GameRecordCodec.read(inflated));
  }

  @Test
  public void valuesOutsideTheRecordAreRefusedWithoutTouchingTheFile() throws IOException {
    Path file = root.resolve("kept.bin.gz");
    GameCohort kept = new GameCohort(Variant.TENHOU_4P, 1, "id", List.of());
    GameRecordCodec.write(file, kept);
    GrpState deep = new GrpState(Variant.TENHOU_4P, 4, 70_000, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    GameRecord game = new GameRecord(Variant.TENHOU_4P, 1, List.of(deep), new int[] {40_000, 30_000, 20_000, 10_000});
    GameCohort wide = new GameCohort(Variant.TENHOU_4P, 1, "id", List.of(game));
    Assert.assertThrows(IllegalArgumentException.class, () -> GameRecordCodec.write(file, wide));
    Assert.assertEquals(GameRecordCodec.read(file), kept);
    Assert.assertFalse(Files.exists(root.resolve("kept.bin.gz.tmp")));
  }

  @Test
  public void handEndsKnowTheirEndedRoundAndRealizedPermutation() {
    for (GameRecord game : TrainFixtures.simulated(Variant.TENHOU_4P, 100, 0, 13)) {
      RankPermutations table = RankPermutations.of(Variant.TENHOU_4P);
      for (int handEnd = 0; handEnd < game.handEnds().size(); handEnd++) {
        GrpState state = game.handEnds().get(handEnd);
        Assert.assertEquals(game.endedRound(handEnd), handEnd == 0 ? 0 : game.handEnds().get(handEnd - 1).round());
        int permutation = game.realizedPermutation(handEnd);
        for (int relative = 0; relative < 4; relative++) {
          int seat = (state.dealer() + relative) % 4;
          Assert.assertEquals(table.rank(permutation, relative), game.finalRank(seat));
          Assert.assertEquals(game.finalRank(seat), Placement.rank(game.finalScores(), seat));
        }
      }
    }
  }

  @Test
  public void recordsCheckTheirVariant() {
    GrpState three = new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_000, 35_000});
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new GameRecord(Variant.TENHOU_4P, 1, List.of(three), new int[] {25_000, 25_000, 25_000, 25_000}));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new GameRecord(Variant.TENHOU_4P, 1, List.of(), new int[3]));
    GameRecord four = new GameRecord(Variant.TENHOU_4P, 1, List.of(), new int[4]);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> new GameCohort(Variant.TENHOU_3P, 1, "id", List.of(four)));
  }

  private static byte[] gunzip(byte[] bytes) throws IOException {
    try (GZIPInputStream in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
      return in.readAllBytes();
    }
  }
}
