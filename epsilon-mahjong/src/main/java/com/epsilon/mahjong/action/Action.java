package com.epsilon.mahjong.action;

import com.epsilon.mahjong.action.ActionSection.Slots;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 打ち手の行動 1 つ。不変の値で、番号から表で引く。同じ番号には、いつも同じインスタンスを返す。合法手は、行動の番号を short の
 * 配列で持つ。番号の並び方は {@link ActionSection} が決める。
 */
public final class Action {

  public enum Type implements VariantScoped {
    /** 立直を宣言しない打牌。 */
    DAHAI,
    CHI(RuleFlag.CHI),
    PON,
    DAIMINKAN,
    ANKAN,
    KAKAN,
    RIICHI_DAHAI,
    TSUMO_AGARI,
    /** 槍槓と、抜いた北へのロンを含む。 */
    RON_AGARI,
    PASS,
    KYUSHU_KYUHAI,
    NUKI(RuleFlag.NUKI);

    private final RuleFlag requires;

    Type() {
      this(null);
    }

    Type(RuleFlag requires) {
      this.requires = requires;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  /**
   * どの物理牌を使うか。打牌では、赤の 5 を打つかと、ツモ切りかを表す。チー・ポンでは、手の赤の 5 を使うかを表す。槓と、
   * 牌の種類を指さない行動は {@code NOT_APPLICABLE}。
   */
  public enum TileSelection {
    NOT_APPLICABLE,
    NON_AKA,
    AKA,
    TSUMOGIRI,
    TSUMOGIRI_AKA;

    /** 赤の 5 を打つか、手の赤の 5 を使うなら true。 */
    public boolean red() {
      return this == AKA || this == TSUMOGIRI_AKA;
    }

    public boolean tsumogiri() {
      return this == TSUMOGIRI || this == TSUMOGIRI_AKA;
    }

    /** 打牌（立直の宣言牌を含む）での牌の選び方を返す。 */
    public static TileSelection discard(boolean red, boolean tsumogiri) {
      if (tsumogiri) {
        return red ? TSUMOGIRI_AKA : TSUMOGIRI;
      }
      return red ? AKA : NON_AKA;
    }
  }

  private static final Action[] ACTIONS = decodeAll();

  /**
   * [セクションの ordinal][牌の種類] → 番号。行があるのは、牌の種類ごとに番号を持つセクション（種類の順か、赤の 5 の色の順に
   * 並ぶもの）だけ。番号の無い種類は −1。
   */
  private static final int[][] KIND_CODES = kindCodes();

  /** [順子の最小の種類 × 3 ＋ 鳴いた位置] → 番号。無い組み合わせは −1。 */
  private static final int[] CHI_CODES = chiCodes(ActionSection.CHI);

  private static final int[] RED_CHI_CODES = chiCodes(ActionSection.CHI_AKA);

  private final int code;
  private final ActionSection section;
  private final int tile;
  private final int base;

  private Action(int code, ActionSection section, int tile, int base) {
    this.code = code;
    this.section = section;
    this.tile = tile;
    this.base = base;
  }

  public static Action of(int code) {
    if (code < 0 || code >= ACTIONS.length) {
      throw new IllegalArgumentException("Action code out of range: " + code);
    }
    return ACTIONS[code];
  }

  /**
   * 牌の種類を 1 つ指す行動（打牌・立直の打牌・ポン・槓）の番号を返す。打牌と立直の打牌では、牌の選び方で赤の 5 かとツモ切りかを
   * 指定する。ポンでは、手の赤の 5 を使うかを指定する。槓には {@code NOT_APPLICABLE} を渡す。番号の無い組み合わせなら
   * IllegalArgumentException を投げる。
   */
  public static int code(Type type, int kind, TileSelection selection) {
    int[] codes = KIND_CODES[ActionSection.of(type, selection).ordinal()];
    int code = codes == null || kind < 0 || kind >= Tiles.KINDS ? -1 : codes[kind];
    if (code < 0) {
      throw new IllegalArgumentException("No action code for " + type + " " + selection + " of kind " + kind);
    }
    return code;
  }

  /**
   * 牌の種類を指さない行動（ツモ和了・ロン・見送り・九種九牌・北抜き）の番号を返す。種類が要る行動なら
   * IllegalArgumentException を投げる。
   */
  public static int code(Type type) {
    ActionSection section = ActionSection.of(type, TileSelection.NOT_APPLICABLE);
    if (section.slots() != Slots.SINGLE) {
      throw new IllegalArgumentException(type + " needs a tile kind");
    }
    return section.start();
  }

  /**
   * 打牌の番号を返す。
   *
   * @param kind 打つ牌の種類
   * @param red 打つ牌が赤の 5 なら true
   * @param tsumogiri ツモ切りなら true
   * @param riichi 立直の宣言牌なら true
   */
  public static int discard(int kind, boolean red, boolean tsumogiri, boolean riichi) {
    return code(riichi ? Type.RIICHI_DAHAI : Type.DAHAI, kind, TileSelection.discard(red, tsumogiri));
  }

  /**
   * チー・ポン・大明槓の面子から、その鳴きの番号を返す。面子の赤の 5 が手から出した牌なら、手の赤の 5 を使った番号にする。
   * 暗槓・加槓を渡すと IllegalArgumentException を投げる。
   */
  public static int call(Meld meld) {
    boolean handRed = meld.aka() == Meld.AkaSource.CONSUMED_HAND_TILE;
    return switch (meld.type()) {
      case CHI -> chi(meld.base(), meld.called(), handRed);
      case PON -> code(Type.PON, meld.base(), handRed ? TileSelection.AKA : TileSelection.NON_AKA);
      case DAIMINKAN -> code(Type.DAIMINKAN, meld.base(), TileSelection.NOT_APPLICABLE);
      case ANKAN, KAKAN -> throw new IllegalArgumentException("Not a call: " + meld);
    };
  }

  /**
   * チーの番号を返す。番号の無い組み合わせなら IllegalArgumentException を投げる。
   *
   * @param base 順子の最小の種類
   * @param called 鳴いた牌の種類
   * @param red 手の赤の 5 を使うなら true
   */
  public static int chi(int base, int called, boolean red) {
    int position = called - base;
    int code =
        base < 0 || base >= Tiles.KINDS || position < 0 || position >= Meld.RUN
            ? -1
            : (red ? RED_CHI_CODES : CHI_CODES)[base * Meld.RUN + position];
    if (code < 0) {
      throw new IllegalArgumentException("No chi code for base " + base + " called " + called + " red " + red);
    }
    return code;
  }

  public int code() {
    return code;
  }

  public ActionSection section() {
    return section;
  }

  public Type type() {
    return section.type();
  }

  public TileSelection selection() {
    return section.selection();
  }

  /** 打つ牌、鳴く牌、槓にする牌の種類（チーは鳴いた牌の種類）。牌の種類を指さない行動は −1。 */
  public int tile() {
    return tile;
  }

  /** 作る面子の最小の種類。チーは順子の最初の牌の種類で、ほかは {@link #tile} と同じ。 */
  public int base() {
    return base;
  }

  /** 番号をそのまま返す。オブジェクトごとの既定のハッシュは実行のたびに変わり、HashMap などの並び順が揺れるので使わない。 */
  @Override
  public int hashCode() {
    return code;
  }

  @Override
  public boolean equals(Object other) {
    return this == other;
  }

  @Override
  public String toString() {
    if (tile < 0) {
      return section.name();
    }
    StringBuilder text = new StringBuilder(section.name()).append('(');
    if (type() != Type.CHI) {
      return text.append(Face.of(tile, selection().red()).mjai()).append(')').toString();
    }
    for (int kind = base; kind < base + Meld.RUN; kind++) {
      text.append(Face.of(kind, false).mjai());
    }
    return text.append(' ').append(Face.of(tile, false).mjai()).append(')').toString();
  }

  private static Action[] decodeAll() {
    Action[] actions = new Action[ActionSection.SPACE];
    for (ActionSection section : ActionSection.values()) {
      int code = section.start();
      switch (section.slots()) {
        case KIND -> {
          for (int kind = 0; kind < Tiles.KINDS; kind++) {
            actions[code] = new Action(code, section, kind, kind);
            code++;
          }
        }
        case RED_FIVE -> {
          for (Tiles.Group suit : Tiles.Group.values()) {
            if (suit.suited()) {
              int five = suit.first() + Tiles.FIVE;
              actions[code] = new Action(code, section, five, five);
              code++;
            }
          }
        }
        case RUN, RED_RUN -> {
          for (Tiles.Group suit : Tiles.Group.values()) {
            if (!suit.suited()) {
              continue;
            }
            for (int start = 0; start < Meld.RUN_STARTS; start++) {
              for (int position = 0; position < Meld.RUN; position++) {
                if (section.slots() == Slots.RED_RUN && !Meld.redFromHand(start, position)) {
                  continue;
                }
                int base = suit.first() + start;
                actions[code] = new Action(code, section, base + position, base);
                code++;
              }
            }
          }
        }
        case SINGLE -> {
          actions[code] = new Action(code, section, -1, -1);
          code++;
        }
      }
      if (code != section.end()) {
        throw new IllegalStateException(
            section + " lists " + (code - section.start()) + " codes but is " + section.width() + " wide");
      }
    }
    return actions;
  }

  private static int[][] kindCodes() {
    int[][] codes = new int[ActionSection.values().length][];
    for (Action action : ACTIONS) {
      Slots slots = action.section.slots();
      if (slots != Slots.KIND && slots != Slots.RED_FIVE) {
        continue;
      }
      int[] row = codes[action.section.ordinal()];
      if (row == null) {
        row = new int[Tiles.KINDS];
        Arrays.fill(row, -1);
        codes[action.section.ordinal()] = row;
      }
      row[action.tile] = action.code;
    }
    return codes;
  }

  private static int[] chiCodes(ActionSection section) {
    int[] codes = new int[Tiles.KINDS * Meld.RUN];
    Arrays.fill(codes, -1);
    for (int code = section.start(); code < section.end(); code++) {
      Action action = ACTIONS[code];
      codes[action.base * Meld.RUN + action.tile - action.base] = code;
    }
    return codes;
  }
}
