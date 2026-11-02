package com.epsilon.prism.grp;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * audit-grp の CSV の組み直し: 次局の局と本場は同じ半荘の次の行、供託は (100000 − Σ s)/1000、半荘の最後の行には次局がなく最終の
 * 点数になる。p と Φ は半荘が続く局末だけを {@link HandEndPhi#gameValues} の順に並べる。
 */
public class AuditCsvGamesTest {
  private static final String HEADER;

  static {
    StringBuilder header = new StringBuilder("game,hand,kyoku,honba,sticks,dealer,s0,s1,s2,s3,ended");
    for (int k = 0; k < 16; k++) header.append(",p").append(k);
    for (String prefix : new String[] {"phi", "rank", "u"})
      for (int seat = 0; seat < 4; seat++) header.append(',').append(prefix).append(seat);
    HEADER = header.toString();
  }

  @Test
  public void rebuildsTheNextHandOfEachRow() throws IOException {
    Path csv =
        write(
            row(8, 0, 0, 0, 1, new int[] {24000, 26000, 25000, 24000}, false, "1,0,2,3"),
            row(8, 1, 0, 1, 0, new int[] {20000, 30000, 26000, 24000}, false, "1,0,2,3"),
            row(8, 2, 1, 0, 0, new int[] {14000, 36000, 26000, 24000}, true, "3,0,1,2"),
            row(9, 0, 0, 0, 0, new int[] {33000, 25000, 25000, 17000}, false, "0,1,2,3"),
            row(9, 1, 1, 0, 0, new int[] {41000, 25000, 25000, 9000}, true, "0,1,2,3"));
    AuditCsvGames audit = AuditCsvGames.read(csv);
    List<RecordedGame> games = audit.games();
    Assert.assertEquals(games.size(), 2);
    RecordedGame first = games.getFirst();
    Assert.assertEquals(first.gameId(), 8);
    Assert.assertEquals(first.continuingHandEnds(), 2);
    // 局末 0 の次局は行 1 の局・本場（東 1 局 1 本場）、点数の合計 99000 から供託 1 本
    assertStart(first.starts().get(0), 0, 1, 1, new int[] {24000, 26000, 25000, 24000});
    assertStart(first.starts().get(1), 1, 0, 0, new int[] {20000, 30000, 26000, 24000});
    Assert.assertEquals(first.finalScores(), new int[] {14000, 36000, 26000, 24000});
    Assert.assertEquals(games.get(1).continuingHandEnds(), 1);
    Assert.assertEquals(audit.marginals().length, 3);
    Assert.assertEquals(audit.phi().length, 3);
    Assert.assertEquals(audit.marginals()[1][5], 0.0125f, 1e-7);
    Assert.assertEquals(audit.phi()[2][3], 0.0003, 1e-12);
    // 次局 root の公開系列は始めの局頭 + 次局 0..k
    Assert.assertEquals(first.prefix(1).length, 3 * 7);
  }

  @Test
  public void rejectsRowsOutOfOrder() throws IOException {
    Path csv =
        write(
            row(8, 0, 0, 0, 0, new int[] {25000, 25000, 25000, 25000}, false, "0,1,2,3"),
            row(8, 2, 1, 0, 0, new int[] {25000, 25000, 25000, 25000}, true, "0,1,2,3"));
    Assert.assertThrows(IOException.class, () -> AuditCsvGames.read(csv));
  }

  @Test
  public void rejectsScoresThatDoNotLeaveWholeSticks() throws IOException {
    Path csv =
        write(
            row(8, 0, 0, 0, 0, new int[] {25000, 25000, 25000, 24500}, false, "0,1,2,3"),
            row(8, 1, 1, 0, 0, new int[] {25000, 25000, 25000, 25000}, true, "0,1,2,3"));
    Assert.assertThrows(IOException.class, () -> AuditCsvGames.read(csv));
  }

  private static void assertStart(HandStart start, int kyoku, int honba, int kyotaku, int[] scores) {
    Assert.assertEquals(start.kyokuIndex(), kyoku);
    Assert.assertEquals(start.honba(), honba);
    Assert.assertEquals(start.kyotaku(), kyotaku);
    Assert.assertEquals(start.scores(), scores);
  }

  /** 1 行（p_i = 0.01·hand + 0.0005·i、phi_s = 0.003·hand + 0.0001·s で行を見分ける。ended の行は p が空）。 */
  private static String row(
      long game, int hand, int kyoku, int honba, int sticks, int[] scores, boolean ended, String ranks) {
    StringBuilder line = new StringBuilder();
    line.append(game).append(',').append(hand).append(',').append(kyoku).append(',').append(honba).append(',')
        .append(sticks).append(",0");
    for (int score : scores) line.append(',').append(score);
    line.append(',').append(ended ? 1 : 0);
    for (int k = 0; k < 16; k++) line.append(',').append(ended ? "" : String.valueOf(0.01 * hand + 0.0005 * k));
    for (int seat = 0; seat < 4; seat++) line.append(',').append(0.003 * hand + 0.0001 * seat);
    line.append(',').append(ranks);
    for (int seat = 0; seat < 4; seat++) line.append(",0");
    return line.toString();
  }

  private static Path write(String... rows) throws IOException {
    Path csv = Files.createTempFile("audit-grp", ".csv.gz");
    try (Writer writer =
        new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(csv)), StandardCharsets.UTF_8)) {
      writer.write(HEADER + "\n");
      for (String row : rows) writer.write(row + "\n");
    }
    csv.toFile().deleteOnExit();
    return csv;
  }
}
