package com.epsilon.agent.policy;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.expectThrows;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 系列の名前（models.json の series に書く値）と接頭辞（卓の相手の指定と結果の名前に使う）は、保存した設定と結果に残るので、
 * 定数で固定する。結果の名前は、接頭辞とチェックポイントのディレクトリ名から作る。卓の相手は「接頭辞:パス」で書く。
 * チェックポイントを取らない系列は、接頭辞だけで書く。
 */
public class SeriesTest {

  @Test
  public void idsAndPrefixesAreThePinnedNames() {
    assertEquals(
        Arrays.stream(Series.values()).map(Series::id).toList(),
        List.of("epsilon-prism", "epsilon-nano", "epsilon", "rulebot"));
    assertEquals(
        Arrays.stream(Series.values()).map(Series::prefix).toList(), List.of("prism", "nano", "major", "heuristic"));
  }

  @Test
  public void labelsUseTheLastDirectoryName() {
    assertEquals(Series.NANO.label(Path.of("checkpoints", "nano", "ppo17")), "nano:ppo17");
    assertEquals(Series.PRISM.label(Path.of("runs", "4p", "snapshots", "00760")), "prism:00760");
    assertEquals(Series.MAJOR.label(Path.of("macro614")), "major:macro614");
    assertEquals(Series.RULE_BOT.label(null), "heuristic");
  }

  @Test
  public void checkpointsFollowTheSeries() {
    expectThrows(IllegalArgumentException.class, () -> Series.NANO.label(null));
    expectThrows(IllegalArgumentException.class, () -> Series.RULE_BOT.label(Path.of("ppo17")));
    expectThrows(IllegalArgumentException.class, () -> new Series.Spec(Series.MAJOR, null));
    expectThrows(IllegalArgumentException.class, () -> new Series.Spec(Series.RULE_BOT, Path.of("ppo17")));
  }

  @Test
  public void seriesAreFoundByIdAndByPrefix() {
    for (Series series : Series.values()) {
      assertSame(Series.ofId(series.id()), series);
      assertSame(Series.ofPrefix(series.prefix()), series);
    }
    expectThrows(IllegalArgumentException.class, () -> Series.ofId("nano"));
    expectThrows(IllegalArgumentException.class, () -> Series.ofPrefix("epsilon-nano"));
    expectThrows(IllegalArgumentException.class, () -> Series.ofPrefix("snapshot"));
  }

  @Test
  public void opponentsAreReadFromTheirSpec() {
    assertEquals(Series.parse("heuristic"), new Series.Spec(Series.RULE_BOT, null));
    assertEquals(
        Series.parse("nano:/models/nano/ppo17"), new Series.Spec(Series.NANO, Path.of("/models/nano/ppo17")));
    assertEquals(Series.parse("major:macro614"), new Series.Spec(Series.MAJOR, Path.of("macro614")));
    Series.Spec snapshot = Series.parse("prism:runs/4p/snapshots/00760");
    assertEquals(snapshot, new Series.Spec(Series.PRISM, Path.of("runs/4p/snapshots/00760")));
    assertEquals(snapshot.series().label(snapshot.checkpoint()), "prism:00760");
    for (String spec : List.of("", "nano", "nano:", "heuristic:ppo17", "rulebot", "snapshot:00760", ":ppo17")) {
      expectThrows(IllegalArgumentException.class, () -> Series.parse(spec));
    }
  }
}
