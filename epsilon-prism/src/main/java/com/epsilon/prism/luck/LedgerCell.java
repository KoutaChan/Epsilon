package com.epsilon.prism.luck;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.DecisionClass;
import java.util.EnumMap;
import java.util.Map;

/**
 * 判断時の局の段階と自席の順位で帳簿の係数を選ぶ区分。区分はルールから決まり、番号は 段階 × 席の数 ＋ 順位（0 始まり）。
 *
 * <p>局の段階は 3 つで、東場、南場の最後の局より前、南場の最後の局から後（西場を含む）。区切りは場風の境と
 * {@link Variant#lastRound()} から取る。四人打ちの段階の名前は {@code EAST}・{@code SOUTH_1_3}・{@code SOUTH_4_PLUS}、三人打ちは
 * {@code EAST}・{@code SOUTH_1_2}・{@code SOUTH_3_PLUS}。
 */
public final class LedgerCell {
  private static final int STAGES = 3;

  private static final Map<Variant, LedgerCell> BY_VARIANT = new EnumMap<>(Variant.class);

  static {
    for (Variant variant : Variant.values()) BY_VARIANT.put(variant, new LedgerCell(variant));
  }

  private final Variant variant;
  private final int seats;
  private final String[] stages;

  /** O を読むリーダーと順位を求める作業領域（読むスレッドごとに 1 つ）。 */
  private final ThreadLocal<Reader> readers;

  private LedgerCell(Variant variant) {
    this.variant = variant;
    seats = variant.seats();
    int south = variant.lastRound() - seats;
    stages = new String[] {"EAST", "SOUTH_1_" + south, "SOUTH_" + (south + 1) + "_PLUS"};
    readers = ThreadLocal.withInitial(() -> new Reader(new Observation(variant), new int[seats]));
  }

  /** ルール {@code variant} の区分。 */
  public static LedgerCell of(Variant variant) {
    return BY_VARIANT.get(variant);
  }

  private record Reader(Observation observation, int[] scores) {}

  /** 区分の数。 */
  public int count() {
    return STAGES * seats;
  }

  /** 観測席からの相対席を絶対席に戻してから、起家順の同点処理で順位を求め、O の判断の区分の番号を返す。 */
  public int of(long[] words, int base) {
    Reader scratch = readers.get();
    Observation o = scratch.observation().bind(words, base);
    for (int seat = 0; seat < seats; seat++) scratch.scores()[o.absoluteSeat(seat)] = o.score(seat);
    return index(o.kyoku(), Placement.rank(scratch.scores(), o.observer()));
  }

  int index(int kyoku, int rank) {
    int stage = kyoku < seats ? 0 : kyoku < variant.lastRound() ? 1 : 2;
    return stage * seats + rank;
  }

  /** 報告に書く区分の名前。順位は 1 始まり。 */
  public String name(DecisionClass group, int cell) {
    return group.name() + "_" + stages[cell / seats] + "_" + (1 + cell % seats);
  }
}
