package com.epsilon.prism.grp;

import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 対局の記録の形（{@link GameLog}）: 書いて読んだ記録が全ての欄で元と同じ（局末のない半荘・負の点数・大きな本場を含む）。書き直しは前の記録を
 * 差し替え、一時ファイルを残さない。途中で切れた記録・印のない記録・範囲外の欄は拒む。
 */
public class GameLogTest {

  @Test
  public void roundTripKeepsEveryField() throws IOException {
    Path directory = Files.createTempDirectory("prism-gamelog");
    try {
      CohortGames games = synthetic(17, 200, 7);
      Path file = new GrpFiles(directory).games(17);
      GameLog.write(file, games);
      assertSame(GameLog.read(file), games);
      // 書き直しは差し替え（一時ファイルは残らない）
      CohortGames other = synthetic(17, 3, 8);
      GameLog.write(file, other);
      assertSame(GameLog.read(file), other);
      try (var listed = Files.list(file.getParent())) {
        Assert.assertEquals(listed.count(), 1L);
      }
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  @Test
  public void rejectsTruncatedAndMalformedLogs() throws IOException {
    Path directory = Files.createTempDirectory("prism-gamelog");
    try {
      Path file = directory.resolve("c00001.bin.gz");
      GameLog.write(file, synthetic(1, 50, 3));
      byte[] bytes = Files.readAllBytes(file);
      Path cut = directory.resolve("cut.bin.gz");
      Files.write(cut, Arrays.copyOf(bytes, bytes.length / 2));
      Assert.assertThrows(IOException.class, () -> GameLog.read(cut));
      Path empty = directory.resolve("empty.bin.gz");
      Files.write(empty, new byte[0]);
      Assert.assertThrows(IOException.class, () -> GameLog.read(empty));
      // 符号なしの欄に入らない値は書かない（前の記録はそのまま）
      var negative =
          new CohortGames(
              header(1),
              List.of(
                  new RecordedGame(
                      1, List.of(new HandStart(1, -1, 0, new int[] {25000, 25000, 25000, 25000})), new int[4])));
      Assert.assertThrows(IllegalArgumentException.class, () -> GameLog.write(file, negative));
      Assert.assertEquals(GameLog.read(file).games().size(), 50);
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  /** 乱数の半荘の記録（半荘 ID は昇順、局末の数は 0 から 15）。 */
  static CohortGames synthetic(int cohort, int count, long seed) {
    var random = new SplittableRandom(seed);
    List<RecordedGame> games = new ArrayList<>();
    for (int g = 0; g < count; g++) {
      int hands = g == 0 ? 0 : random.nextInt(16);
      List<HandStart> starts = new ArrayList<>();
      for (int k = 0; k < hands; k++)
        starts.add(
            new HandStart(
                Math.min(11, k),
                random.nextInt(g == 1 ? 300 : 6),
                random.nextInt(5),
                new int[] {
                  random.nextInt(-20000, 80000),
                  random.nextInt(-20000, 80000),
                  random.nextInt(-20000, 80000),
                  random.nextInt(-20000, 80000)
                }));
      games.add(
          new RecordedGame(
              1_000_000L * cohort + 3L * g,
              starts,
              new int[] {random.nextInt(-30000, 90000), 25000, 25000, random.nextInt(100)}));
    }
    return new CohortGames(header(cohort), games);
  }

  static GameLogHeader header(int cohort) {
    return new GameLogHeader(cohort, "g003-a1", "ab".repeat(32), 0.005, 0.0, 123_456_789L, 4321L);
  }

  static void assertSame(CohortGames actual, CohortGames expected) {
    Assert.assertEquals(actual.header(), expected.header());
    Assert.assertEquals(actual.games().size(), expected.games().size());
    for (int g = 0; g < expected.games().size(); g++) {
      RecordedGame a = actual.games().get(g), e = expected.games().get(g);
      Assert.assertEquals(a.gameId(), e.gameId());
      Assert.assertEquals(a.finalScores(), e.finalScores());
      Assert.assertEquals(a.starts().size(), e.starts().size());
      for (int k = 0; k < e.starts().size(); k++) {
        HandStart as = a.starts().get(k), es = e.starts().get(k);
        Assert.assertEquals(as.kyokuIndex(), es.kyokuIndex());
        Assert.assertEquals(as.honba(), es.honba());
        Assert.assertEquals(as.kyotaku(), es.kyotaku());
        Assert.assertEquals(as.scores(), es.scores());
      }
    }
    Assert.assertEquals(actual.handEnds(), expected.handEnds());
  }
}
