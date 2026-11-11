package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.LedgerSettings;
import com.epsilon.prism.noise.LossWeightSettings;
import com.epsilon.prism.noise.SlopeRefresh;
import com.epsilon.prism.noise.UraSettings;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * 持ち越しのキーのうち、運の精算の部分。帳簿の列と教師の意味を決める設定の値を持つ。この値が前の反復と違えば、前の反復の補正の R² は
 * 同じ量を測っていない。帳簿のモードが持たないキーは、組の集合なら空、ほかの値なら null にする。
 *
 * @param columns 行に書く帳簿の列の並び（帳簿を書かないモードでは空）
 * @param mode 帳簿のモード（設定に書く名前）
 * @param compute 計算する組
 * @param pointClip 点の列を切る幅（千点）
 * @param policyClip 方策の列で手の値を切る幅（効用）
 * @param slopeStep 傾きの列で点をずらす幅（点）
 * @param slopeRefresh 傾きの列を作り直す時点
 * @param apply 補正に使う組
 * @param cells 係数を当てはめる区分
 * @param weighting 係数を当てはめるときの行の重み
 * @param ura 立直の和了で終わった局で、局末の値をどう作るか（設定に書く名前）
 */
public record LedgerMeaning(
    List<LedgerColumn> columns,
    String mode,
    Set<LedgerGroup> compute,
    @Key(doc = "点の列を切る幅（千点）", presence = Presence.ABSENT) Double pointClip,
    @Key(doc = "方策の列で手の値を切る幅（効用）", presence = Presence.ABSENT) Double policyClip,
    @Key(doc = "傾きの列で点をずらす幅（点）", presence = Presence.ABSENT) Integer slopeStep,
    @Key(doc = "傾きの列を作り直す時点", presence = Presence.ABSENT) SlopeRefresh slopeRefresh,
    Set<LedgerGroup> apply,
    @Key(doc = "係数を当てはめる区分", presence = Presence.ABSENT) LedgerCells cells,
    LossWeightSettings.Weighting weighting,
    String ura) {

  public LedgerMeaning {
    columns = List.copyOf(columns);
    Objects.requireNonNull(mode, "mode");
    compute = LedgerRecording.groups(compute);
    apply = LedgerRecording.groups(apply);
    Objects.requireNonNull(weighting, "weighting");
    Objects.requireNonNull(ura, "ura");
  }

  /**
   * 設定とルールから作る。
   *
   * @param catalog ルールで決まる帳簿の列の並び
   */
  static LedgerMeaning of(
      LedgerSettings ledger, LedgerCatalog catalog, LossWeightSettings lossWeight, UraSettings ura) {
    LedgerSettings.Recording recording = ledger instanceof LedgerSettings.Recording r ? r : null;
    LedgerSettings.Fitting fitting = ledger instanceof LedgerSettings.Fitting f ? f : null;
    return new LedgerMeaning(
        recording == null ? List.of() : catalog.columns(),
        caseName(ledger),
        recording == null ? Set.of() : recording.compute(),
        recording == null ? null : recording.pointClip(),
        recording == null ? null : recording.policyClip(),
        recording == null ? null : recording.slopeStep(),
        recording == null ? null : recording.slopeRefresh(),
        fitting == null ? Set.of() : fitting.apply(),
        fitting == null ? null : fitting.cells(),
        lossWeight.fit(),
        caseName(ura));
  }

  /**
   * 前の反復から変わった項目を、英語の 1 文で返す。変わった項目が複数あれば、最初の 1 つだけを書く。
   *
   * @param previous 前の反復の値
   * @return 全て同じなら null
   */
  public String change(LedgerMeaning previous) {
    record Item(String name, Object before, Object after) {}
    List<Item> items =
        List.of(
            new Item("columns", previous.columns, columns),
            new Item("mode", previous.mode, mode),
            new Item("compute", previous.compute, compute),
            new Item("pointClip", previous.pointClip, pointClip),
            new Item("policyClip", previous.policyClip, policyClip),
            new Item("slopeStep", previous.slopeStep, slopeStep),
            new Item("slopeRefresh", previous.slopeRefresh, slopeRefresh),
            new Item("apply", previous.apply, apply),
            new Item("cells", previous.cells, cells),
            new Item("weighting", previous.weighting, weighting),
            new Item("ura", previous.ura, ura));
    for (Item item : items)
      if (!Objects.equals(item.before(), item.after()))
        return "the ledger " + item.name() + " changed from " + item.before() + " to " + item.after();
    return null;
  }

  /** 設定の選択肢を、設定に書く名前で返す。 */
  private static String caseName(Object choice) {
    return choice.getClass().getAnnotation(Case.class).value();
  }
}
