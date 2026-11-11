package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.noise.LedgerGroup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 帳簿の列（{@link LedgerColumn}）と、ルールごとの列の並び（{@link LedgerCatalog}）のテスト。列の名前と順は、運の精算を luck に
 * まとめる前の帳簿（EventLedger の NAMES）と同じ。組は 17 列を重ならずに分ける。どのルールでも 17 列を宣言の順に並べる。
 */
public class LedgerColumnTest {

  /** まとめる前の帳簿が行に書いていた列の名前（行に書く順）。 */
  private static final List<String> FORMER_NAMES =
      List.of(
          "ownImprove1",
          "ownImprove2",
          "ownImprove3",
          "ownTsumo",
          "ownForcedDealIn",
          "ownDrawnDanger",
          "oppImprove1",
          "oppImprove2",
          "oppTsumo",
          "oppForcedToSelf",
          "oppForcedToOther",
          "oppDrawnChance",
          "doraSelf",
          "doraOpponents",
          "ura",
          "policySelf",
          "policyOpponent");

  /** 列の名前と順は、まとめる前の帳簿と同じ。名前は重ならない。 */
  @Test
  public void namesAndOrderAreTheFormerOnes() {
    Assert.assertEquals(LedgerColumn.COUNT, FORMER_NAMES.size());
    Assert.assertEquals(Arrays.stream(LedgerColumn.values()).map(LedgerColumn::reportName).toList(), FORMER_NAMES);
    Assert.assertEquals(new TreeSet<>(FORMER_NAMES).size(), FORMER_NAMES.size());
  }

  /**
   * 組は 17 列を重ならずに分け、使う組は {@link LedgerGroup#computed()} の 5 つちょうど。向聴の列は IMPROVE、点数の列は POINTS で、
   * IMPROVE と POINTS を合わせると、まとめる前の報告の draw の範囲 [0, 12) になる。dora は [12, 14)、ura は [14, 15)、policy は
   * [15, 17)。
   */
  @Test
  public void groupsSplitTheColumns() {
    Map<LedgerGroup, TreeSet<Integer>> byGroup = new EnumMap<>(LedgerGroup.class);
    for (LedgerColumn column : LedgerColumn.values())
      byGroup.computeIfAbsent(column.group(), group -> new TreeSet<>()).add(column.ordinal());
    Assert.assertEquals(byGroup.keySet(), LedgerGroup.computed());
    Assert.assertEquals(byGroup.get(LedgerGroup.IMPROVE), new TreeSet<>(List.of(0, 1, 2, 6, 7)));
    Assert.assertEquals(byGroup.get(LedgerGroup.POINTS), new TreeSet<>(List.of(3, 4, 5, 8, 9, 10, 11)));
    Assert.assertEquals(byGroup.get(LedgerGroup.DORA), range(12, 14));
    Assert.assertEquals(byGroup.get(LedgerGroup.URA), range(14, 15));
    Assert.assertEquals(byGroup.get(LedgerGroup.POLICY), range(15, 17));
    TreeSet<Integer> draw = new TreeSet<>(byGroup.get(LedgerGroup.IMPROVE));
    draw.addAll(byGroup.get(LedgerGroup.POINTS));
    Assert.assertEquals(draw, range(0, 12));
    int columns = byGroup.values().stream().mapToInt(TreeSet::size).sum();
    Assert.assertEquals(columns, LedgerColumn.COUNT);
    for (LedgerColumn column : LedgerColumn.values()) {
      String name = column.reportName();
      if (name.contains("Improve")) Assert.assertEquals(column.group(), LedgerGroup.IMPROVE, name);
      if (name.contains("Tsumo") || name.contains("DealIn") || name.contains("Danger") || name.contains("Forced")
          || name.contains("Chance")) Assert.assertEquals(column.group(), LedgerGroup.POINTS, name);
    }
  }

  /** どのルールの並びも、全ての列を宣言の順に置いた 17 列で、並びは書き換えられない。 */
  @Test
  public void everyRuleCatalogHoldsTheSeventeenColumnsInOrder() {
    for (Variant variant : Variant.values()) {
      LedgerCatalog catalog = LedgerCatalog.of(variant);
      Assert.assertEquals(catalog.columns(), List.of(LedgerColumn.values()), variant.name());
      Assert.assertEquals(catalog.width(), LedgerColumn.COUNT, variant.name());
      Assert.assertThrows(UnsupportedOperationException.class, () -> catalog.columns().add(LedgerColumn.URA));
    }
    Assert.assertThrows(NullPointerException.class, () -> LedgerCatalog.of(null));
  }

  /** 渡した一覧を後で書き換えても、並びは変わらない。 */
  @Test
  public void catalogCopiesItsColumns() {
    var columns = new ArrayList<>(List.of(LedgerColumn.OWN_TSUMO, LedgerColumn.URA));
    LedgerCatalog catalog = new LedgerCatalog(columns);
    columns.add(LedgerColumn.POLICY_SELF);
    Assert.assertEquals(catalog.columns(), List.of(LedgerColumn.OWN_TSUMO, LedgerColumn.URA));
    Assert.assertEquals(catalog.width(), 2);
  }

  private static TreeSet<Integer> range(int from, int to) {
    TreeSet<Integer> range = new TreeSet<>();
    for (int i = from; i < to; i++) range.add(i);
    return range;
  }
}
