package com.epsilon.agent.eval;

import com.epsilon.agent.policy.Policy;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Objects;

/**
 * 評価の卓の計画。牌山の番号は、最初の番号から牌山の数だけ続く。どの割り当ての半荘でも、局の牌山と手の抽選に使う乱数は、
 * 牌山の種と番号だけから作る。そのため牌山の種と番号が同じなら、スロットの打ち手や席の割り当てを替えても、同じ牌山で打つ。割り当ての番号は
 * {@link Placements} の決まりで数える。
 *
 * @param slots スロットごとの方策（2 つ以上）。スロットは卓に参加する打ち手の番号で、決闘では 0 が候補、1 が相手。卓は方策を閉じない
 * @param splits 席の分け方（スロットごとの席の数）。どれも {@link Placements} の決まりに合い、スロットの数が方策の数と同じ
 * @param drawOrder 全ての半荘で使うツモの順
 * @param wallSet 牌山の集合の決め方（牌山の種の素を作る側が使う）
 * @param wallSeed 牌山の種の素（卓は {@link Arena#EVALUATION_SALT} を混ぜて使う）
 * @param wallBase 最初の牌山の番号
 * @param walls 牌山の数（1 以上）
 * @param utility 順位の効用
 */
public record ArenaPlan(
    Variant variant,
    List<Policy> slots,
    Seating seating,
    List<Split> splits,
    DrawOrder drawOrder,
    WallSet wallSet,
    long wallSeed,
    long wallBase,
    int walls,
    RankUtility utility) {

  public ArenaPlan {
    Objects.requireNonNull(variant, "variant");
    slots = List.copyOf(slots);
    Objects.requireNonNull(seating, "seating");
    splits = List.copyOf(splits);
    Objects.requireNonNull(drawOrder, "drawOrder");
    Objects.requireNonNull(wallSet, "wallSet");
    Objects.requireNonNull(utility, "utility");
    if (slots.size() < 2) {
      throw new IllegalArgumentException("An arena seats at least two slots, not " + slots.size());
    }
    for (int slot = 0; slot < slots.size(); slot++) {
      if (slots.get(slot).variant() != variant) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + variant + " but found " + slots.get(slot).variant() + " in slot " + slot);
      }
    }
    if (walls < 1) {
      throw new IllegalArgumentException("An arena plays at least one wall: " + walls);
    }
    if (wallBase < 0 || wallBase > Long.MAX_VALUE - walls) {
      throw new IllegalArgumentException("Walls " + wallBase + " + " + walls + " are outside the wall numbers");
    }
    Placements placements = new Placements(variant, seating, splits);
    if (placements.slots() != slots.size()) {
      throw new IllegalArgumentException(
          "Split " + splits.get(0) + " does not seat " + slots.size() + " slots at the " + variant.seats() + " seats");
    }
    long perWall = placements.gamesPerWall();
    if (walls * perWall > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(walls + " walls of " + perWall + " games are too many for one arena");
    }
  }

  /** @return 席の割り当て */
  public Placements layout() {
    return new Placements(variant, seating, splits);
  }

  /**
   * @param done 打ち終えた牌山の数（0 以上で、牌山の数より小さい）
   * @return 打ち終えた牌山より後ろだけを打つ計画。牌山の番号・割り当て・乱数は元の計画と同じ
   */
  public ArenaPlan rest(int done) {
    Objects.checkIndex(done, walls);
    return new ArenaPlan(
        variant, slots, seating, splits, drawOrder, wallSet, wallSeed, wallBase + done, walls - done, utility);
  }

  /** @return 牌山 1 つで打つ半荘の数（{@link Seating#BY_GAME} なら 1、ほかは全ての割り当ての数） */
  public int gamesPerWall() {
    return layout().gamesPerWall();
  }

  /**
   * @param wall 牌山の番号
   * @param game 牌山の中の半荘の番号（0 から {@link #gamesPerWall} − 1 まで）
   * @return その半荘の割り当ての番号
   */
  public int placement(long wall, int game) {
    return layout().placement(wall, game);
  }

  /** @return 割り当てごと・席ごとのスロットの番号（[割り当て][席]。呼ぶたびに新しく作る） */
  public int[][] slotTable() {
    return layout().slotTable();
  }
}
