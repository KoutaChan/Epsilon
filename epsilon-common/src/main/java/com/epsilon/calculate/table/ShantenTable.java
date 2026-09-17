package com.epsilon.calculate.table;

/**
 * 通常形のシャンテン計算に使う圧縮テーブルを、新しいオブジェクトを生成せずに参照する。
 *
 * <p>通常の呼び出し元は{@code com.epsilon.calculate.shape.HandShapeAnalyzer}を使う。
 * このクラスの検証を省いたメソッドは、牌種ごとの枚数を5進数で符号化した検証済みの値だけを受け取る。
 */
public final class ShantenTable {

  private ShantenTable() {}

  public static int suitKey(int[] tileCounts, int offset) {
    return ShapeTableLoader.suitKey(tileCounts, offset);
  }

  public static int honorKey(int[] tileCounts) {
    return ShapeTableLoader.honorKey(tileCounts);
  }

  public static int suitCountCode(int[] tileCounts, int offset) {
    return ShapeTableLoader.suitCountCode(tileCounts, offset);
  }

  public static int honorCountCode(int[] tileCounts) {
    return ShapeTableLoader.honorCountCode(tileCounts);
  }

  public static int suitKey(int countCode) {
    return ShapeTableLoader.suitKey(countCode);
  }

  public static int honorKey(int countCode) {
    return ShapeTableLoader.honorKey(countCode);
  }

  public static int suitKeyUnchecked(int countCode) {
    return ShapeTableLoader.suitKeyUnchecked(countCode);
  }

  public static int honorKeyUnchecked(int countCode) {
    return ShapeTableLoader.honorKeyUnchecked(countCode);
  }

  public static int suitKeyAfterAdding(int countCode, int rankIndex) {
    return ShapeTableLoader.suitKeyAfterAdding(countCode, rankIndex);
  }

  public static int honorKeyAfterAdding(int countCode, int honorIndex) {
    return ShapeTableLoader.honorKeyAfterAdding(countCode, honorIndex);
  }

  /** 変更しない数牌2色を、既存の合成設定キーへまとめる。 */
  public static int mergeTwoSuitKeys(int firstKey, int secondKey) {
    ShapeTableLoader table = ShapeTableLoader.table();
    return table.mergeTwo[firstKey * table.suitKeyCount + secondKey] & 0xff;
  }

  /** 数牌2色の合成キーへ、残る1色のキーを加える。 */
  public static int mergeThirdSuitKey(int mergedTwoKey, int thirdKey) {
    ShapeTableLoader table = ShapeTableLoader.table();
    return table.mergeThree[mergedTwoKey * table.suitKeyCount + thirdKey] & 0xff;
  }

  /** 合成済み数牌3色と字牌から、指定面子数の通常形向聴数を返す。 */
  public static int standardShantenFromMergedSuits(
      int mergedSuitKey, int honorKey, int meldTarget) {
    ShapeTableLoader table = ShapeTableLoader.table();
    return table.distances[(mergedSuitKey * table.honorKeyCount + honorKey) * 5 + meldTarget] - 1;
  }

  public static int standardShanten(
      int manKey, int pinKey, int souKey, int honorKey, int meldTarget) {
    return ShapeTableLoader.standardShanten(manKey, pinKey, souKey, honorKey, meldTarget);
  }
}
