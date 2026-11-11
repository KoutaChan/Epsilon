package com.epsilon.prism.luck;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * ルールごとの牌の表と、プール（誰にも見えない牌）や席が持つ牌の枚数を数える処理。帳簿の計算器と裏ドラの期待値の計算で共有する。表は
 * 作った後に変えないので、どのスレッドから読んでもよい。数えた枚数を書く配列と手牌の写しは、呼び出し元が自分の作業領域に持つ。
 */
final class LedgerTables {

  private static final LedgerTables[] BY_VARIANT = build();

  final Variant variant;
  final int seats;

  /** ルールで使う牌の種類のビット集合。 */
  final long kinds;

  /** 赤牌がある種類のビット集合。 */
  final long redKinds;

  /** 表示牌の種類 → ドラの種類（ルールで使わない種類は −1）。 */
  final int[] doraOf = new int[Tiles.KINDS];

  /** 種類ごとの牌の枚数。 */
  final int[] copies = new int[Tiles.KINDS];

  private LedgerTables(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    TileSet set = variant.tiles();
    kinds = set.kinds();
    redKinds = set.redKinds();
    Arrays.fill(doraOf, -1);
    for (long bits = kinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      doraOf[kind] = set.doraOf(kind);
      copies[kind] = set.copies(kind);
    }
  }

  /** ルールの表を返す。 */
  static LedgerTables of(Variant variant) {
    return BY_VARIANT[variant.ordinal()];
  }

  private static LedgerTables[] build() {
    Variant[] variants = Variant.values();
    LedgerTables[] tables = new LedgerTables[variants.length];
    for (Variant variant : variants) tables[variant.ordinal()] = new LedgerTables(variant);
    return tables;
  }

  /**
   * 判断の時点で誰にも見えない牌の枚数（次のイベントの前のプール）を、種類ごとに書く。ルールで使わない種類は 0。
   *
   * @param full 判断の時点の局面
   * @param pool 種類ごとの枚数を書く先
   * @return プールの枚数の和
   */
  int pool(FullView full, int[] pool) {
    Arrays.fill(pool, 0);
    int total = 0;
    for (long bits = kinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      int left = full.unseen(kind);
      pool[kind] = left;
      total += left;
    }
    return total;
  }

  /**
   * 席が持つ牌（純手牌・面子・抜いた北）の枚数を、種類ごとに書く。
   *
   * @param hand 純手牌を写す作業領域。呼んだ後は席の純手牌を持つ
   * @param counts 種類ごとの枚数を書く先
   */
  void holdings(FullView full, int seat, HandShape hand, int[] counts) {
    Arrays.fill(counts, 0);
    full.copyHand(seat, hand);
    for (long bits = hand.kinds(); bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      counts[kind] = hand.count(kind);
    }
    SeatView view = full.seat(seat);
    for (int m = 0; m < view.meldCount(seat); m++) {
      Meld meld = Meld.of(view.meld(seat, m));
      for (int k = 0; k < meld.size(); k++) counts[meld.tile(k)]++;
    }
    counts[Tiles.PEI] += view.nuki(seat);
  }

  /**
   * 和了した席について、和了形の牌の枚数を種類ごとに書く。純手牌が 13 枚相当なら、栄和した牌（直前の打牌・加槓の牌・抜いた北）を足す。
   *
   * @param hand 純手牌を写す作業領域
   * @param counts 種類ごとの枚数を書く先
   * @throws IllegalStateException 純手牌が 13 枚相当でも 14 枚相当でもないとき、13 枚相当で栄和の牌が無いとき
   */
  void winningHand(FullView full, int winner, HandShape hand, int[] counts) {
    holdings(full, winner, hand, counts);
    int size = hand.size();
    if (size == HandShape.MAX_SIZE) return;
    int tile = full.seat(winner).last().tile();
    if (size != HandShape.MAX_SIZE - 1 || tile < 0)
      throw new IllegalStateException("Seat " + winner + " won with " + size + " tiles and the last tile " + tile);
    counts[tile]++;
  }
}
