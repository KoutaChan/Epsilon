package com.epsilon.prism.run;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.rows.CohortWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * コホートを確定した後の仕上げ（{@link CollectStep#finish}）と、GRP のバックグラウンドの処理が見る確定の判定
 * （{@link GrpFiles#sealed}）のテスト。収集の段は、半荘の記録と数のファイルをコホートのディレクトリ（rows/&lt;t&gt;）の中に書く。
 * 仕上げで、この 2 つは中身を変えずに grp/records と grp/counts の c&lt;t&gt;.bin へ移る。どちらか 1 つでも中に残っていれば、
 * コホートはまだ確定していないとみなす。仕上げは何度呼んでも結果が同じ。
 */
public class CollectFinishTest {
  private Path root;
  private RunDirectory run;
  private GrpFiles grp;

  @BeforeMethod
  public void createRun() throws IOException {
    root = Files.createTempDirectory("prism-finish");
    run = new RunDirectory(root);
    grp = new GrpFiles(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(root);
  }

  @Test
  public void theFilesLiveInTheCohortAndMoveUnderGrp() {
    Assert.assertEquals(grp.cohort(3), run.rows(3), "the collect step seals its part directory as rows/<t>");
    Assert.assertEquals(grp.unmovedRecords(3), run.rows(3).resolve(GrpFiles.UNMOVED_RECORDS));
    Assert.assertEquals(grp.unmovedCounts(3), run.rows(3).resolve(GrpFiles.UNMOVED_COUNTS));
    Assert.assertNotEquals(GrpFiles.UNMOVED_COUNTS, GrpFiles.UNMOVED_RECORDS);
    Assert.assertEquals(grp.counts(3).getFileName().toString(), "c00003.bin");
    Assert.assertEquals(grp.counts(3).getParent().getFileName().toString(), "counts");
    Assert.assertEquals(grp.counts(3).getParent().getParent(), grp.records(3).getParent().getParent());
    Assert.assertNotEquals(grp.counts(3), grp.records(3));
  }

  @Test
  public void eitherUnmovedFileKeepsTheCohortUnsealed() throws IOException {
    Assert.assertFalse(grp.sealed(3), "no manifest");
    seal(3);
    Assert.assertTrue(grp.sealed(3), "a cohort without records is sealed");
    write(grp.unmovedCounts(3), "counts");
    Assert.assertFalse(grp.sealed(3), "the count file is still inside");
    Files.delete(grp.unmovedCounts(3));
    write(grp.unmovedRecords(3), "records");
    Assert.assertFalse(grp.sealed(3), "the game log is still inside");
    write(grp.unmovedCounts(3), "counts");
    Assert.assertFalse(grp.sealed(3), "both are still inside");
  }

  @Test
  public void finishMovesBothFilesOnceAndKeepsTheirBytes() throws IOException {
    seal(3);
    write(grp.unmovedRecords(3), "records of 3");
    write(grp.unmovedCounts(3), "counts of 3");
    CollectStep.finish(run, 3);
    Assert.assertTrue(grp.sealed(3));
    Assert.assertFalse(Files.exists(grp.unmovedRecords(3)));
    Assert.assertFalse(Files.exists(grp.unmovedCounts(3)));
    Assert.assertEquals(read(grp.records(3)), "records of 3");
    Assert.assertEquals(read(grp.counts(3)), "counts of 3");
    // もう一度呼んでも何も変えない
    CollectStep.finish(run, 3);
    Assert.assertTrue(grp.sealed(3));
    Assert.assertEquals(read(grp.records(3)), "records of 3");
    Assert.assertEquals(read(grp.counts(3)), "counts of 3");
  }

  /** 半荘の記録を移した後で落ちたコホートでは、残った数のファイルだけを移す。前の試行で残った同じ名前のファイルは差し替える。 */
  @Test
  public void finishMovesWhatIsLeft() throws IOException {
    seal(4);
    write(grp.records(4), "records of 4");
    write(grp.counts(4), "stale counts");
    write(grp.unmovedCounts(4), "counts of 4");
    Assert.assertFalse(grp.sealed(4));
    CollectStep.finish(run, 4);
    Assert.assertTrue(grp.sealed(4));
    Assert.assertEquals(read(grp.records(4)), "records of 4");
    Assert.assertEquals(read(grp.counts(4)), "counts of 4");
    // 数のファイルだけのコホート（半荘の記録が書けなかったとき）も移す
    seal(5);
    write(grp.unmovedCounts(5), "counts of 5");
    CollectStep.finish(run, 5);
    Assert.assertTrue(grp.sealed(5));
    Assert.assertFalse(Files.exists(grp.records(5)));
    Assert.assertEquals(read(grp.counts(5)), "counts of 5");
    // 何も無いコホートでは何もしない
    seal(6);
    CollectStep.finish(run, 6);
    Assert.assertTrue(grp.sealed(6));
    Assert.assertFalse(Files.exists(grp.records(6)));
    Assert.assertFalse(Files.exists(grp.counts(6)));
  }

  private void seal(int t) throws IOException {
    Files.createDirectories(grp.cohort(t));
    Files.writeString(grp.cohort(t).resolve(CohortWriter.MANIFEST), "{}");
  }

  private static void write(Path file, String text) throws IOException {
    Files.createDirectories(file.getParent());
    Files.writeString(file, text, StandardCharsets.UTF_8);
  }

  private static String read(Path file) throws IOException {
    return Files.readString(file, StandardCharsets.UTF_8);
  }
}
