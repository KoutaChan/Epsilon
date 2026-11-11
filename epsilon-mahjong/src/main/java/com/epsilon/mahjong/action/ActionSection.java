package com.epsilon.mahjong.action;

import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 行動の番号を、種別と牌の選び方ごとに分けたセクション。各セクションは、番号の並び方と幅を持つ。先頭の番号は、列挙した順に
 * 幅を足し上げて一度だけ決める。系譜の記録は行動の番号を持つので、セクションの順、幅、中の並び方は、記録を読む側との契約になる。
 */
public enum ActionSection implements VariantScoped {
  DAHAI(Type.DAHAI, TileSelection.NON_AKA, Slots.KIND),
  DAHAI_TSUMOGIRI(Type.DAHAI, TileSelection.TSUMOGIRI, Slots.KIND),
  DAHAI_AKA(Type.DAHAI, TileSelection.AKA, Slots.RED_FIVE),
  DAHAI_TSUMOGIRI_AKA(Type.DAHAI, TileSelection.TSUMOGIRI_AKA, Slots.RED_FIVE),
  CHI(Type.CHI, TileSelection.NON_AKA, Slots.RUN),
  CHI_AKA(Type.CHI, TileSelection.AKA, Slots.RED_RUN),
  PON(Type.PON, TileSelection.NON_AKA, Slots.KIND),
  PON_AKA(Type.PON, TileSelection.AKA, Slots.RED_FIVE),
  DAIMINKAN(Type.DAIMINKAN, TileSelection.NOT_APPLICABLE, Slots.KIND),
  ANKAN(Type.ANKAN, TileSelection.NOT_APPLICABLE, Slots.KIND),
  KAKAN(Type.KAKAN, TileSelection.NOT_APPLICABLE, Slots.KIND),
  RIICHI_DAHAI(Type.RIICHI_DAHAI, TileSelection.NON_AKA, Slots.KIND),
  RIICHI_TSUMOGIRI(Type.RIICHI_DAHAI, TileSelection.TSUMOGIRI, Slots.KIND),
  RIICHI_AKA(Type.RIICHI_DAHAI, TileSelection.AKA, Slots.RED_FIVE),
  RIICHI_TSUMOGIRI_AKA(Type.RIICHI_DAHAI, TileSelection.TSUMOGIRI_AKA, Slots.RED_FIVE),
  TSUMO_AGARI(Type.TSUMO_AGARI, TileSelection.NOT_APPLICABLE, Slots.SINGLE),
  RON_AGARI(Type.RON_AGARI, TileSelection.NOT_APPLICABLE, Slots.SINGLE),
  PASS(Type.PASS, TileSelection.NOT_APPLICABLE, Slots.SINGLE),
  KYUSHU_KYUHAI(Type.KYUSHU_KYUHAI, TileSelection.NOT_APPLICABLE, Slots.SINGLE),
  NUKI(Type.NUKI, TileSelection.NOT_APPLICABLE, Slots.SINGLE);

  /** セクションの中の番号の並び方。幅は並ぶ番号の数。 */
  enum Slots {
    /** 牌の種類の順。 */
    KIND(Tiles.KINDS),
    /** 赤の 5 の色の順（萬・筒・索）。 */
    RED_FIVE(Tiles.NUMBER_SUITS),
    /** 色、順子の最初の数字、鳴いた位置の順。 */
    RUN(Tiles.NUMBER_SUITS * Meld.RUN_STARTS * Meld.RUN),
    /**
     * {@link #RUN} のうち、手の赤の 5 を使えるチーだけを並べる（順子が 5 を含み、鳴いた牌が 5 でないもの）。5 を含む順子は 1 色に
     * 3 通りあり、鳴いた位置は 5 以外の 2 通り。並び方は {@link #RUN} と同じ。
     */
    RED_RUN(Tiles.NUMBER_SUITS * Meld.RUN * (Meld.RUN - 1)),
    /** 牌の種類を指さない行動に、番号を 1 つだけ割り当てる。 */
    SINGLE(1);

    private final int width;

    Slots(int width) {
      this.width = width;
    }
  }

  private static final ActionSection[] VALUES = values();

  /** [セクションの ordinal] → 先頭の番号。最後の要素は、最後のセクションの終わり。 */
  private static final int[] STARTS = starts();

  /** どのルールの番号も収まる番号の数（最後のセクションの終わり）。 */
  public static final int SPACE = STARTS[VALUES.length];

  private static final ActionSection[] BY_CODE = byCode();

  /** [種別の ordinal][牌の選び方の ordinal] → セクション。無い組み合わせは null。 */
  private static final ActionSection[][] BY_SELECTION = bySelection();

  private final Type type;
  private final TileSelection selection;
  private final Slots slots;

  ActionSection(Type type, TileSelection selection, Slots slots) {
    this.type = type;
    this.selection = selection;
    this.slots = slots;
  }

  public Type type() {
    return type;
  }

  public TileSelection selection() {
    return selection;
  }

  public int width() {
    return slots.width;
  }

  public int start() {
    return STARTS[ordinal()];
  }

  public int end() {
    return STARTS[ordinal() + 1];
  }

  public boolean contains(int code) {
    return code >= start() && code < end();
  }

  @Override
  public RuleFlag requires() {
    return type.requires();
  }

  Slots slots() {
    return slots;
  }

  public static ActionSection of(int code) {
    if (code < 0 || code >= SPACE) {
      throw new IllegalArgumentException("Action code out of range: " + code);
    }
    return BY_CODE[code];
  }

  /** 種別と牌の選び方に当たるセクションを返す。無い組み合わせなら IllegalArgumentException を投げる。 */
  public static ActionSection of(Type type, TileSelection selection) {
    ActionSection section = BY_SELECTION[type.ordinal()][selection.ordinal()];
    if (section == null) {
      throw new IllegalArgumentException("No action section for " + type + " with " + selection);
    }
    return section;
  }

  /** そのルールで使うセクションのうち、最後のものの終わりを返す。そのルールの行動の番号は、どれもこれより小さい。 */
  public static int space(Variant variant) {
    int space = 0;
    for (ActionSection section : VALUES) {
      if (section.in(variant)) {
        space = Math.max(space, section.end());
      }
    }
    return space;
  }

  private static int[] starts() {
    int[] starts = new int[VALUES.length + 1];
    for (ActionSection section : VALUES) {
      starts[section.ordinal() + 1] = starts[section.ordinal()] + section.width();
    }
    return starts;
  }

  private static ActionSection[] byCode() {
    ActionSection[] sections = new ActionSection[SPACE];
    for (ActionSection section : VALUES) {
      Arrays.fill(sections, section.start(), section.end(), section);
    }
    return sections;
  }

  private static ActionSection[][] bySelection() {
    ActionSection[][] sections = new ActionSection[Type.values().length][TileSelection.values().length];
    for (ActionSection section : VALUES) {
      sections[section.type.ordinal()][section.selection.ordinal()] = section;
    }
    return sections;
  }
}
