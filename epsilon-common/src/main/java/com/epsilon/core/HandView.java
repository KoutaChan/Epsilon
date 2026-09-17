package com.epsilon.core;

import com.epsilon.calculate.shape.HandShapeView;

/**
 * 実際の手牌と候補行動を適用した仮想的な手牌に共通する、読み取り専用のインターフェース。
 *
 * <p>候補の比較では配列を複製せず、牌種ごとの枚数、手牌形の符号、確定面子を直接参照する。
 */
public interface HandView extends HandShapeView {

  /** 手牌（副露・暗槓を除く）に指定牌種の赤5があるかを返す。 */
  boolean hasAkaTile(int tileType);

  /** 手牌（副露・暗槓を除く）に指定牌種の通常牌があるかを返す。 */
  default boolean hasNonAkaTile(int tileType) {
    int tileCount = count(tileType);
    return hasAkaTile(tileType) ? tileCount > 1 : tileCount > 0;
  }

  /** 手牌（副露・暗槓を除く）に指定牌種の通常牌が2枚以上あるかを返す。 */
  default boolean hasMultipleNonAkaTiles(int tileType) {
    int tileCount = count(tileType);
    return hasAkaTile(tileType) ? tileCount > 2 : tileCount > 1;
  }

  /** 手牌（副露・暗槓を除く）の赤5所有状態を返す。 */
  int concealedAkaMask();

  /** 手牌（副露・暗槓を除く）と確定面子を合わせた赤5所有状態を返す。 */
  int ownedAkaMask();

  /** 指定位置の共有の面子を返す。 */
  Meld meld(int meldIndex);

  /** チー・ポン成立時の本人の河サイズを返す。不明または対象外は-1。 */
  int meldCallAfterRiverIndex(int meldIndex);

  /** 槓成立時の本人の河サイズを返す。不明または対象外は-1。 */
  int meldKanAfterRiverIndex(int meldIndex);

  /** 暗槓以外の副露がなく門前ならtrue。 */
  boolean isMenzen();

  /** 確定槓子数を返す。 */
  int kanCount();

  /** 面子の識別子を成立順に詰めたキャッシュ照合用の値を返す。 */
  long canonicalMeldSignature();
}
