package com.epsilon.prism.grp;

import com.epsilon.agent.app.Exit;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.PhiTable;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 設定の確かめ（settings check）での Φ のテスト。正しい設定なら終了コードは 0。読み込みで落ちる設定（古い形の値、知らないキー、自前の GRP
 * 以外での GRP の運用のキー、記録しない自前の GRP）と、開けない GRP、GruSinkhorn でない GRP に重ねた表は 4。設定ファイルが無ければ 2。
 * どの場合も何も書かない。固定の GRP の項目はチェックポイントと表の sha を持つ。点数の順位から始める自前の GRP では、g000 を決めるのに
 * GRP を読まない。
 */
public class SettingsCheckTest {

  private Path directory;

  @BeforeMethod
  public void createDirectory() throws IOException {
    directory = Files.createTempDirectory("prism-settings-check");
  }

  @AfterMethod(alwaysRun = true)
  public void deleteDirectory() throws IOException {
    Snapshot.deleteTree(directory);
  }

  @Test
  public void validSettingsPassAndNameTheirPhi() throws Exception {
    Assert.assertEquals(check(write("score-rank.toml", "[phi]\nsource = \"SCORE_RANK\"\n")), Exit.OK.code());
    Assert.assertEquals(check(write("self.toml", "[phi]\nsource = \"SELF_GRP\"\n")), Exit.OK.code());
    try (var listed = Files.list(directory)) {
      Assert.assertEquals(listed.count(), 2L, "settings check writes nothing");
    }
    // 点数の順位から始める自前の GRP では、g000 は GRP を読まずに決まる
    PhiEntry initial = PhiHistory.initialEntry(GrpFixtures.scoreStart());
    Assert.assertEquals(initial.id(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(initial.kind(), PhiEntryKind.SCORE_RANK);
    // 固定の GRP（GruSinkhorn）に表を重ねる項目は、チェックポイントと表の sha を持つ
    Path gru = GrpFixtures.gru(directory.resolve("gru"), 1);
    Path table = directory.resolve("table.json");
    PhiTable.of(GrpFixtures.VARIANT, new double[] {1.1}, -1, 2.0 / 3, new double[][] {{0.01}}).write(table);
    Assert.assertEquals(check(write("fixed.toml", fixed(gru, table))), Exit.OK.code());
    PhiEntry entry = PhiScorer.original(GrpFixtures.setup("phi.source=FIXED_GRP", "phi.grp=" + gru, "phi.table=" + table));
    Assert.assertEquals(entry.id(), PhiEntry.FIXED_ID);
    Assert.assertEquals(entry.kind(), PhiEntryKind.NETWORK);
    Assert.assertEquals(entry.sha256(), GrpCheckpoint.sha256(gru));
    Assert.assertEquals(entry.tableSha256(), PhiTable.sha256(table));
    // 表を重ねられるのは GruSinkhorn だけ
    Path network = GrpFixtures.network(directory.resolve("network"), 2);
    Assert.assertEquals(check(write("overlaid.toml", fixed(network, table))), Exit.PROBLEMS.code());
    Assert.assertEquals(check(write("network.toml", fixed(network, null))), Exit.OK.code());
  }

  @Test
  public void brokenSettingsReportProblems() throws Exception {
    Assert.assertEquals(check(write("old.toml", "phi = \"grp\"\n")), Exit.PROBLEMS.code());
    Assert.assertEquals(check(write("unknown.toml", "grpRefitEvery = 48\n")), Exit.PROBLEMS.code());
    // 自前の GRP は記録が要る
    Assert.assertEquals(
        check(write("unrecorded.toml", "[phi]\nsource = \"SELF_GRP\"\n\n[phi.grp]\nmonitor = \"OFF\"\n")),
        Exit.PROBLEMS.code());
    // GRP の運用のキーは、自前の GRP の設定でしか使えない
    Assert.assertEquals(
        check(write("monitored.toml", "[phi]\nsource = \"SCORE_RANK\"\n\n[phi.grp]\nmonitor = \"SCORE\"\n")),
        Exit.PROBLEMS.code());
    // 開けない GRP と、無い設定ファイル
    Assert.assertEquals(check(write("missing.toml", fixed(directory.resolve("no-grp"), null))), Exit.PROBLEMS.code());
    Assert.assertEquals(check(directory.resolve("absent.toml")), Exit.USAGE.code());
  }

  /** {@code settings check --settings <ファイル>} を実行し、終了コードを返す。 */
  private static int check(Path file) {
    return PhiControlTest.prism("settings", "check", "--settings", file.toString());
  }

  /** 固定の GRP を使う Φ のセクションを返す。表が null なら表を重ねない。 */
  private static String fixed(Path grp, Path table) {
    return "[phi]\nsource = \"FIXED_GRP\"\ngrp = \"" + grp + "\"\n" + (table == null ? "" : "table = \"" + table + "\"\n");
  }

  /** 渡した本文（いちばん外のキーとセクション）に、テスト用のルールと乱数の種を足して、設定ファイルを書く。 */
  private Path write(String name, String body) throws IOException {
    return Files.writeString(directory.resolve(name), body + "\n[run]\nvariant = \"TENHOU_4P\"\nseed = 1\n");
  }
}
