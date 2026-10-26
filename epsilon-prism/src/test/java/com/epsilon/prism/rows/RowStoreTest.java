package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class RowStoreTest {
  /**
   * id ごとに中身を変えた行。種類は id % 3、組は id の 2 の位、id % 5 == 0 なら保留。測った手は played と、id が偶数なら隣の手も（根なら隣の手が
   * 逸脱 d、奇数なら d = played）。
   */
  private static PrismRow row(int iteration, long id) {
    int n = 2 + (int) (id % 4);
    int played = (int) (id % n);
    PrismRow.Source source = PrismRow.Source.values()[(int) (id % 3)];
    float[] values = new float[n];
    Arrays.fill(values, Float.NaN);
    values[played] = (float) (id % 7) * 0.1f - 0.3f;
    if (id % 2 == 0) values[(played + 1) % n] = (float) (id % 5) * 0.2f - 0.5f;
    float[] prior = new float[n];
    Arrays.fill(prior, 1f / n);
    byte[] support = new byte[Math.min(n, 3)];
    for (int i = 0; i < support.length; i++) support[i] = (byte) i;
    int deviation = -1;
    float probability = Float.NaN, propensity = Float.NaN;
    long rootDecisionId = -1;
    if (source == PrismRow.Source.ROOT) {
      deviation = id % 2 == 0 ? (played + 1) % n : played;
      probability = 0.25f;
      propensity = 0.03f * 0.25f;
    }
    if (source == PrismRow.Source.ROLLIN) rootDecisionId = id * 7;
    return new PrismRow(
        iteration,
        id,
        id * 31,
        (int) (id & 3),
        Kind.values()[(int) (id % Kind.values().length)],
        Stage.values()[(int) (id % Stage.values().length)],
        DecisionClass.values()[(int) (id % DecisionClass.COUNT)],
        id % 5 == 0 ? PrismRow.Purpose.HOLDOUT : PrismRow.Purpose.TRAIN,
        ((id / 2) & 1) == 0 ? PrismRow.Group.A : PrismRow.Group.B,
        source,
        played,
        n,
        values,
        prior,
        support,
        deviation,
        probability,
        propensity,
        rootDecisionId,
        ObservationPayload.takeOwnership(new byte[] {1, 2, (byte) id}));
  }

  private static long id(int t, int i) {
    return (long) t * 1_000_000 + i;
  }

  private static Path cohort(Path root, int t, int rows) throws Exception {
    Path dir = root.resolve(String.format("%05d", t));
    try (var writer = new RowWriter(dir, t, 2)) {
      for (int i = 0; i < rows; i++) writer.accept(row(t, id(t, i)));
      writer.seal();
    }
    return dir;
  }

  /** 全ての欄がシャードを通って元の行に戻り、索引は行から作った索引と同じ。 */
  @Test
  public void framesRoundTripThroughShards() throws Exception {
    Path root = Files.createTempDirectory("prism-rows");
    try (var store = new RowStore()) {
      for (int t = 1; t <= 3; t++) store.addCohort(cohort(root, t, 300));
      Assert.assertEquals(store.size(), 900);
      Assert.assertEquals(store.cohorts(), List.of(1, 2, 3));
      int checked = 0;
      for (int t = 1; t <= 3; t++)
        for (int index : store.select(t, entry -> true)) {
          PrismRow read = store.read(index);
          PrismRow expected = row(t, read.gameId());
          Assert.assertEquals(RowFormat.encodeRaw(read), RowFormat.encodeRaw(expected));
          Assert.assertEquals(read.payload(), expected.payload());
          for (int a = 0; a < expected.actions(); a++)
            Assert.assertEquals(
                Float.floatToIntBits(read.values()[a]),
                Float.floatToIntBits(expected.values()[a]));
          RowFormat.Index entry = store.index(index);
          RowFormat.Index made = RowFormat.index(expected, entry.frameLength());
          Assert.assertEquals(entry, made);
          Assert.assertEquals(entry.measured(), expected.measuredCount());
          if (expected.source() == PrismRow.Source.ROOT)
            Assert.assertEquals(read.deviationValue(), expected.values()[expected.deviation()]);
          checked++;
        }
      Assert.assertEquals(checked, 900);
    }
  }

  /** データの記憶の間引き: keep = 1 は全行、keep &lt; 1 は同じ種なら何度読んでも同じ行で、割合はおよそ keep。 */
  @Test
  public void keepSubsamplesDeterministically() throws Exception {
    Path root = Files.createTempDirectory("prism-rows");
    Path dir = cohort(root, 2, 2000);
    try (var all = new RowStore();
        var first = new RowStore();
        var second = new RowStore();
        var other = new RowStore()) {
      all.addCohort(dir, 1, 7);
      first.addCohort(dir, 0.3, 7);
      second.addCohort(dir, 0.3, 7);
      other.addCohort(dir, 0.3, 8);
      Assert.assertEquals(all.size(), 2000);
      Assert.assertTrue(first.size() > 500 && first.size() < 700, "kept " + first.size());
      Assert.assertEquals(second.size(), first.size());
      for (int row = 0; row < first.size(); row++)
        Assert.assertEquals(first.read(row).gameId(), second.read(row).gameId());
      boolean differs = other.size() != first.size();
      for (int row = 0; !differs && row < first.size(); row++)
        differs = other.read(row).gameId() != first.read(row).gameId();
      Assert.assertTrue(differs, "another salt picks other rows");
      Assert.assertThrows(IllegalArgumentException.class, () -> all.addCohort(dir, 0, 7));
    }
  }

  /** 索引の条件で選び、コホートの範囲はコホートの順に並べる。読み込んでいないコホートは空で、封印していないコホートは読まない。 */
  @Test
  public void selectsByIndexAndCohortRange() throws Exception {
    Path root = Files.createTempDirectory("prism-rows");
    try (var store = new RowStore()) {
      store.addCohort(cohort(root, 4, 101));
      store.addCohort(cohort(root, 5, 57));
      int[] roots = store.select(4, entry -> entry.source() == PrismRow.Source.ROOT);
      for (int index : roots) Assert.assertEquals(store.read(index).source(), PrismRow.Source.ROOT);
      long expectedRoots = 0;
      for (int i = 0; i < 101; i++) if (id(4, i) % 3 == 2) expectedRoots++;
      Assert.assertEquals(roots.length, expectedRoots);
      int[] both = store.select(3, 6, entry -> entry.group() == PrismRow.Group.B);
      int[] four = store.select(4, entry -> entry.group() == PrismRow.Group.B);
      int[] five = store.select(5, entry -> entry.group() == PrismRow.Group.B);
      Assert.assertEquals(both.length, four.length + five.length);
      for (int i = 0; i < four.length; i++) Assert.assertEquals(both[i], four[i]);
      for (int i = 0; i < five.length; i++) Assert.assertEquals(both[four.length + i], five[i]);
      Assert.assertEquals(store.select(6, entry -> true).length, 0);
      Path unsealed = root.resolve("00007");
      try (var writer = new RowWriter(unsealed, 7, 1)) {
        writer.accept(row(7, id(7, 0)));
      }
      Assert.assertThrows(IOException.class, () -> store.addCohort(unsealed));
    }
  }
}
