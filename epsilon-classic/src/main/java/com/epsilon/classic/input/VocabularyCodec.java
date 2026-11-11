package com.epsilon.classic.input;

import com.epsilon.classic.input.ClassicVocabulary.ActionBlock;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.ClassicVocabulary.KanKind;
import com.epsilon.classic.input.ClassicVocabulary.MeldAkaSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldSource;
import com.epsilon.classic.input.ClassicVocabulary.MeldType;
import com.epsilon.classic.input.ClassicVocabulary.RiichiStatus;
import com.epsilon.classic.input.ClassicVocabulary.SettlementAssumption;
import com.epsilon.classic.input.ClassicVocabulary.TileSelection;
import com.epsilon.classic.input.ClassicVocabulary.Yaku;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.tile.Tiles;
import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * エンジンの値を classic の語彙へ変換する。この変換をするのはこのクラスだけ。enum は定数の名前で対応させ、手は classic の手の番号
 * （{@link ActionBlock}）にする。四人打ちのルールにあるエンジンの定数には、どれも classic の語彙に同じ名前の定数がある。無ければ、
 * クラスを読み込むときに IllegalStateException を投げる。四人打ちに無い定数（抜き）を変換すると IllegalArgumentException を投げる。
 * どのメソッドもメモリを確保しない。
 */
public final class VocabularyCodec {

  /** classic の語彙が使うルール（Nano・Major は四人打ちだけ）。 */
  private static final Variant VARIANT = Variant.TENHOU_4P;

  private static final ActionType[] ACTION_TYPES = byName(Action.Type.class, ActionType.class);
  private static final TileSelection[] SELECTIONS = byName(Action.TileSelection.class, TileSelection.class);
  private static final MeldType[] MELD_TYPES = byName(Meld.Type.class, MeldType.class);
  private static final MeldSource[] MELD_SOURCES = byName(Meld.RelativeSource.class, MeldSource.class);
  private static final MeldAkaSource[] AKA_SOURCES = byName(Meld.AkaSource.class, MeldAkaSource.class);
  private static final RiichiStatus[] RIICHI = byName(RiichiState.class, RiichiStatus.class);
  private static final KanKind[] KAN_KINDS = byName(TurnEvent.KanKind.class, KanKind.class);
  private static final SettlementAssumption[] SETTLEMENTS =
      byName(WinSettlementProjection.SettlementAssumption.class, SettlementAssumption.class);
  private static final Yaku[] YAKU = byName(ScoringYaku.class, Yaku.class);

  /** 手の赤の 5 を使うチーの、色の中での番号。添字は（順子の始まり × 3 ＋ 鳴いた牌の位置）。手の赤を使えない組み合わせは −1。 */
  private static final int[] RED_RUNS = redRuns();

  private VocabularyCodec() {}

  public static ActionType actionType(Action.Type type) {
    return require(ACTION_TYPES, type);
  }

  public static TileSelection tileSelection(Action.TileSelection selection) {
    return require(SELECTIONS, selection);
  }

  public static MeldType meldType(Meld.Type type) {
    return require(MELD_TYPES, type);
  }

  public static MeldSource meldSource(Meld.RelativeSource source) {
    return require(MELD_SOURCES, source);
  }

  public static MeldAkaSource meldAkaSource(Meld.AkaSource source) {
    return require(AKA_SOURCES, source);
  }

  public static RiichiStatus riichiStatus(RiichiState state) {
    return require(RIICHI, state);
  }

  public static KanKind kanKind(TurnEvent.KanKind kind) {
    return require(KAN_KINDS, kind);
  }

  public static SettlementAssumption settlement(WinSettlementProjection.SettlementAssumption assumption) {
    return require(SETTLEMENTS, assumption);
  }

  /**
   * 役の集合を classic の役の bit に変換する。
   *
   * @param yaku エンジンの役の集合（bit k が ordinal k の役）
   * @return classic の役の集合（bit k が ordinal k の役）
   */
  public static long yaku(long yaku) {
    long mapped = 0;
    for (long rest = yaku; rest != 0; rest &= rest - 1) {
      mapped |= 1L << YAKU[Long.numberOfTrailingZeros(rest)].ordinal();
    }
    return mapped;
  }

  /**
   * classic の手の番号。
   *
   * @return 0 以上 372 未満
   * @throws IllegalArgumentException 四人打ちに無い手、赤を選んだのに赤の 5 を使えない手、打ち方（赤か、ツモ切りか）の無い打牌のとき
   */
  public static int actionIndex(Action action) {
    int tile = action.tile();
    boolean red = action.selection().red();
    return switch (actionType(action.type())) {
      case DAHAI -> identity(tile, action.selection());
      case RIICHI_DAHAI -> ActionBlock.RIICHI.start() - ActionBlock.DAHAI.start() + identity(tile, action.selection());
      case CHI -> red ? ActionBlock.CHI_AKA.start() + redRun(action) : ActionBlock.CHI.start() + run(action);
      case PON -> red ? ActionBlock.PON_AKA.start() + suit(tile) : ActionBlock.PON.start() + tile;
      case DAIMINKAN -> ActionBlock.DAIMINKAN.start() + tile;
      case ANKAN -> ActionBlock.ANKAN.start() + tile;
      case KAKAN -> ActionBlock.KAKAN.start() + tile;
      case TSUMO_AGARI -> ActionBlock.TSUMO.start();
      case RON_AGARI -> ActionBlock.RON.start();
      case PASS -> ActionBlock.PASS.start();
      case KYUSHU_KYUHAI -> ActionBlock.KYUSHU_KYUHAI.start();
    };
  }

  /**
   * 打牌と立直の打牌が共有する、物理の打牌の番号（打牌の 4 つのブロックを続けた中での位置）。同じ牌を同じ打ち方（赤か、ツモ切りか）で
   * 打つなら、打牌と立直の打牌は同じ番号になる。
   *
   * @return 0 以上 74 未満。打牌でなければ −1
   */
  public static int discardIdentity(Action action) {
    ActionType type = actionType(action.type());
    return type == ActionType.DAHAI || type == ActionType.RIICHI_DAHAI
        ? identity(action.tile(), action.selection())
        : -1;
  }

  /** 打牌の 4 つのブロックを続けた中での位置。 */
  private static int identity(int kind, Action.TileSelection selection) {
    return switch (selection) {
      case NON_AKA -> ActionBlock.DAHAI.start() + kind;
      case TSUMOGIRI -> ActionBlock.DAHAI_TSUMOGIRI.start() + kind;
      case AKA -> ActionBlock.DAHAI_AKA.start() + suit(kind);
      case TSUMOGIRI_AKA -> ActionBlock.DAHAI_TSUMOGIRI_AKA.start() + suit(kind);
      case NOT_APPLICABLE -> throw new IllegalArgumentException("A discard needs a tile selection");
    };
  }

  /** 赤の 5 の色の番号（萬 0・筒 1・索 2）。5 でなければ IllegalArgumentException を投げる。 */
  private static int suit(int kind) {
    Tiles.Group group = Tiles.Group.of(kind);
    if (!group.suited() || kind != group.first() + Tiles.FIVE) {
      throw new IllegalArgumentException("Only a five has a red copy: kind " + kind);
    }
    return group.ordinal();
  }

  /** チーのブロックの中での位置。色 × 21 ＋ 順子の始まり × 3 ＋ 鳴いた牌の位置。 */
  private static int run(Action chi) {
    int base = chi.base();
    return (Tiles.Group.of(base).ordinal() * ClassicVocabulary.RUN_STARTS + Tiles.number(base)) * ClassicVocabulary.RUN
        + chi.tile()
        - base;
  }

  /** 手の赤の 5 を使うチーのブロックの中での位置。色 × 6 ＋ 色の中での番号。 */
  private static int redRun(Action chi) {
    int base = chi.base();
    int pattern = RED_RUNS[Tiles.number(base) * ClassicVocabulary.RUN + chi.tile() - base];
    if (pattern < 0) {
      throw new IllegalArgumentException(chi + " cannot take the red five from the hand");
    }
    int perSuit = ActionBlock.CHI_AKA.width() / Tiles.NUMBER_SUITS;
    return Tiles.Group.of(base).ordinal() * perSuit + pattern;
  }

  private static int[] redRuns() {
    int[] patterns = new int[ClassicVocabulary.RUN_STARTS * ClassicVocabulary.RUN];
    Arrays.fill(patterns, -1);
    int next = 0;
    for (int start = 0; start < ClassicVocabulary.RUN_STARTS; start++) {
      for (int position = 0; position < ClassicVocabulary.RUN; position++) {
        boolean holdsFive = start <= Tiles.FIVE && Tiles.FIVE < start + ClassicVocabulary.RUN;
        if (holdsFive && start + position != Tiles.FIVE) {
          patterns[start * ClassicVocabulary.RUN + position] = next++;
        }
      }
    }
    return patterns;
  }

  private static <E extends Enum<E>, C extends Enum<C>> C require(C[] mapped, E constant) {
    C value = mapped[constant.ordinal()];
    if (value == null) {
      throw new IllegalArgumentException("Classic inputs have no " + constant.getDeclaringClass().getSimpleName()
          + " " + constant);
    }
    return value;
  }

  /**
   * エンジンの ordinal の順に、同じ名前の classic の定数を並べた配列。四人打ちのルールにあるエンジンの定数に、同じ名前の classic の
   * 定数が無ければ IllegalStateException を投げる。
   */
  private static <E extends Enum<E>, C extends Enum<C>> C[] byName(Class<E> engine, Class<C> classic) {
    E[] constants = engine.getEnumConstants();
    @SuppressWarnings("unchecked")
    C[] mapped = (C[]) Array.newInstance(classic, constants.length);
    for (E constant : constants) {
      for (C candidate : classic.getEnumConstants()) {
        if (candidate.name().equals(constant.name())) {
          mapped[constant.ordinal()] = candidate;
        }
      }
      boolean inVariant = !(constant instanceof VariantScoped scoped) || scoped.in(VARIANT);
      if (mapped[constant.ordinal()] == null && inVariant) {
        throw new IllegalStateException(classic.getSimpleName() + " has no " + engine.getSimpleName() + " " + constant);
      }
    }
    return mapped;
  }
}
