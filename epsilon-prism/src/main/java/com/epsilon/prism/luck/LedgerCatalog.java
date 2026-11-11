package com.epsilon.prism.luck;

import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Objects;

/**
 * ルールごとの帳簿の列の並び。行の帳簿は、測った手ごとにこの並びで列を書く。
 *
 * @param columns 列（宣言の順）
 */
public record LedgerCatalog(List<LedgerColumn> columns) {

  /** 全ての列を宣言の順に置いたもの。四人打ちも三人打ちもこれを使う。 */
  private static final LedgerCatalog CLASSIC = new LedgerCatalog(List.of(LedgerColumn.values()));

  public LedgerCatalog {
    columns = List.copyOf(columns);
  }

  /** ルールの列の並びを返す。 */
  public static LedgerCatalog of(Variant variant) {
    Objects.requireNonNull(variant, "variant");
    return CLASSIC;
  }

  /** 測った手 1 つあたりの列の数。 */
  public int width() {
    return columns.size();
  }
}
