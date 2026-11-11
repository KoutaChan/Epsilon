package com.epsilon.agent.eval;

import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Objects;

/**
 * 卓の席の割り当ての一覧。割り当ての番号は分け方の順に通しで振り、分け方の中は {@link Split} の番号の順にする。打ち手に依らないので、
 * 打ち手のモデルを読み込まずにジョブの記録を読むときにも使う。
 *
 * @param seating 席の決め方
 * @param splits 分け方（1 つ以上で重複なし。どれもスロットの数が同じで、卓の全ての席を埋める）。{@link Seating#BY_GAME} では、スロット 0 が
 *     1 席でスロット 1 が残りの席の分け方 1 つだけ
 */
public record Placements(Variant variant, Seating seating, List<Split> splits) {

  /** @throws IllegalArgumentException 分け方が上の条件に合わないとき */
  public Placements {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(seating, "seating");
    splits = List.copyOf(splits);
    if (splits.isEmpty()) {
      throw new IllegalArgumentException("An arena needs a split");
    }
    int slots = splits.get(0).slots();
    for (int index = 0; index < splits.size(); index++) {
      Split split = splits.get(index);
      if (split.slots() != slots || split.seatCount() != variant.seats()) {
        throw new IllegalArgumentException(
            "Split " + split + " does not seat " + slots + " slots at the " + variant.seats() + " seats");
      }
      for (int earlier = 0; earlier < index; earlier++) {
        if (splits.get(earlier).equals(split)) {
          throw new IllegalArgumentException("Repeated split in " + splits);
        }
      }
    }
    if (seating == Seating.BY_GAME && (splits.size() != 1 || slots != 2 || splits.get(0).seatsOf(0) != 1)) {
      throw new IllegalArgumentException(
          "A " + seating + " arena plays the split " + Split.oneAgainstRest(variant) + " only, not " + splits);
    }
  }

  /** @return スロットの数 */
  public int slots() {
    return splits.get(0).slots();
  }

  /** @return 全ての分け方の割り当ての数の和 */
  public int count() {
    int placements = 0;
    for (Split split : splits) {
      placements += split.placements();
    }
    return placements;
  }

  /** @return 1 つの牌山で打つ半荘の数（{@link Seating#BY_GAME} は 1、ほかは全ての割り当ての数） */
  public int gamesPerWall() {
    return seating == Seating.BY_GAME ? 1 : count();
  }

  /**
   * @param wall 牌山の番号
   * @param game 牌山の中の半荘の番号（0 から {@link #gamesPerWall} − 1）
   * @return その半荘の割り当ての番号
   */
  public int placement(long wall, int game) {
    Objects.checkIndex(game, gamesPerWall());
    return seating == Seating.BY_GAME ? Math.floorMod(wall, variant.seats()) : game;
  }

  /**
   * @param placement 割り当ての番号
   * @return その割り当ての分け方の番号
   */
  public int splitOf(int placement) {
    Objects.checkIndex(placement, count());
    int rest = placement;
    for (int split = 0; ; split++) {
      int count = splits.get(split).placements();
      if (rest < count) {
        return split;
      }
      rest -= count;
    }
  }

  /** @return 割り当てごと・席ごとのスロットの番号（[割り当て][席]。呼ぶたびに新しく作る） */
  public int[][] slotTable() {
    int[][] table = new int[count()][variant.seats()];
    int placement = 0;
    for (Split split : splits) {
      for (int local = 0; local < split.placements(); local++) {
        split.slotsOf(local, table[placement++]);
      }
    }
    return table;
  }
}
