package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 向聴・和了牌・有効牌・受け入れ枚数を数える。不変で、スレッドの間で共有できる。どのメソッドもメモリを確保しない。
 *
 * <p>「足りない枚数」は、ある形を作るのに手へ足す牌の最小の枚数。形は 3 つある。通常の形（副露・暗槓と合わせて面子 4 つと雀頭
 * 1 つ）、七対子（違う種類の対子 7 つ）、国士無双（么九牌 13 種類と、そのどれかの対子）。七対子と国士無双は面子の無い手でだけ作れる。
 * 形を作るときは、各種類をその種類の牌の数（ルールで決まり、4 枚か 0 枚）まで使える。向聴は、3 つの形の足りない枚数のうち最も
 * 小さいものから 1 を引いた値で、和了の形なら −1。
 *
 * <p>ルールで使わない種類（三人打ちの 2〜8 萬）と、手に 4 枚ある種類は引けないので、和了牌にも有効牌にも入れない。三人打ちの萬子は
 * 1m と 9m だけで順子を作れないので、萬子のキーは、順子にならない 2 種類の表（{@code ShapeSection.TERMINAL_MAN_KEY}）から引く。
 */
public final class ShapeAnalyzer {

  /** 作れない形について、足りない枚数の代わりに使う値。 */
  private static final int IMPOSSIBLE = Integer.MAX_VALUE;

  private static final long MAN_KINDS = Tiles.range(Tiles.M1, Tiles.M9);
  private static final long PIN_KINDS = Tiles.range(Tiles.P1, Tiles.P9);
  private static final long SOU_KINDS = Tiles.range(Tiles.S1, Tiles.S9);
  private static final long HONOR_KINDS = Tiles.range(Tiles.TON, Tiles.CHUN);
  private static final long TERMINAL_MAN_KINDS = Tiles.bit(Tiles.M1) | Tiles.bit(Tiles.M9);

  /** 順子の先頭になれる種類（各色の 1〜7）。 */
  private static final long RUN_STARTS = runStarts();

  private final TileSet tiles;
  private final long kinds;
  private final boolean terminalMan;

  public ShapeAnalyzer(Variant variant) {
    tiles = variant.tiles();
    kinds = tiles.kinds();
    long man = kinds & MAN_KINDS;
    terminalMan = man == TERMINAL_MAN_KINDS;
    if (!terminalMan && man != MAN_KINDS) {
      throw new IllegalArgumentException("No shape table for the man kinds of " + tiles);
    }
  }

  /** 3 つの形のうち、最も小さい向聴。 */
  public int shanten(HandShape hand) {
    return Math.min(standardDistance(hand), Math.min(chiitoitsuDistance(hand), kokushiDistance(hand))) - 1;
  }

  public int standardShanten(HandShape hand) {
    return standardDistance(hand) - 1;
  }

  /** 七対子の向聴。面子のある手は七対子にならないので IllegalArgumentException を投げる。 */
  public static int chiitoitsuShanten(HandShape hand) {
    requireConcealed(hand);
    return chiitoitsuDistance(hand) - 1;
  }

  /** 国士無双の向聴。面子のある手は国士無双にならないので IllegalArgumentException を投げる。 */
  public static int kokushiShanten(HandShape hand) {
    requireConcealed(hand);
    return kokushiDistance(hand) - 1;
  }

  /**
   * 13 枚の手の和了牌を、形だけを見て返す（役・振聴・残りの枚数は見ない）。聴牌でなければ 0。手の枚数が 13 でなければ
   * IllegalArgumentException を投げる。
   */
  public long waits(HandShape hand) {
    if (hand.size() != HandShape.MAX_SIZE - 1) {
      throw new IllegalArgumentException("Waits need a hand of 13 tiles: " + hand);
    }
    return improving(hand, true, null);
  }

  /** 引くと向聴が下がる種類（有効牌）。残りの枚数は見ない。手の枚数が 14 なら IllegalArgumentException を投げる。 */
  public long improving(HandShape hand) {
    requireDrawable(hand);
    return improving(hand, false, null);
  }

  /**
   * 向聴（最も小さいものと、3 つの形それぞれ）と有効牌を、形の表を 1 回だけ引いてまとめて求める。値は {@link #shanten}・
   * {@link #standardShanten}・{@link #chiitoitsuShanten}・{@link #kokushiShanten}・{@link #improving} と同じ。手の枚数が
   * 14 なら IllegalArgumentException を投げる。
   *
   * @param hand 手
   * @param out 結果を書く先（呼び出し元が持って使い回す）
   */
  public void analyze(HandShape hand, Analysis out) {
    requireDrawable(hand);
    out.improving = improving(hand, false, out);
  }

  /**
   * 引くと足りない枚数が減る種類。聴牌の手では、これが和了牌になる。
   *
   * @param tenpaiOnly true なら、聴牌（足りない枚数が 1）でない手では候補の牌を調べずに 0 を返す
   * @param out null でなければ、向聴を書く先
   */
  private long improving(HandShape hand, boolean tenpaiOnly, Analysis out) {
    int melds = Variant.MAX_MELDS - hand.melds();
    int man = manKey(hand.code(Block.MAN));
    int pin = ShapeTables.suitKey(hand.code(Block.PIN));
    int sou = ShapeTables.suitKey(hand.code(Block.SOU));
    int honor = ShapeTables.honorKey(hand.code(Block.HONOR));
    int manPin = ShapeTables.twoSuits(man, pin);
    int suits = ShapeTables.threeSuits(manPin, sou);
    int standard = ShapeTables.distance(suits, honor, melds);
    int chiitoitsu = chiitoitsuDistance(hand);
    int kokushi = kokushiDistance(hand);
    int best = Math.min(standard, Math.min(chiitoitsu, kokushi));
    if (out != null) {
      out.shanten = best - 1;
      out.standard = standard - 1;
      out.chiitoitsu = chiitoitsu - 1;
      out.kokushi = kokushi - 1;
      out.concealed = hand.melds() == 0;
    }
    if (tenpaiOnly && best != 1) {
      return 0;
    }
    long drawable = kinds & ~hand.full();
    long improving = 0;
    if (chiitoitsu == best) {
      improving |= hand.kinds() & ~hand.pairs();
      if (Long.bitCount(hand.kinds()) < HandShape.SEVEN_PAIRS) {
        improving |= ~hand.kinds();
      }
    }
    if (kokushi == best) {
      boolean pair = (hand.pairs() & Tiles.TERMINALS_AND_HONORS) != 0;
      improving |= pair ? Tiles.TERMINALS_AND_HONORS & ~hand.kinds() : Tiles.TERMINALS_AND_HONORS;
    }
    if (standard == best) {
      long candidates = drawable & ~improving;
      if (best == 1) {
        candidates &= completions(hand.kinds());
      }
      improving |= standardImproving(hand, candidates, best, melds, man, pin, sou, honor);
    }
    return improving & drawable;
  }

  /**
   * 候補の牌のうち、引くと通常の形の足りない枚数が {@code best} 未満になる種類。引き直すのは引いた牌のブロックのキーだけで、
   * ほかのブロックのキーは先に合わせておく。
   */
  private long standardImproving(
      HandShape hand, long candidates, int best, int melds, int man, int pin, int sou, int honor) {
    long improving = 0;
    long manCandidates = candidates & MAN_KINDS;
    if (manCandidates != 0) {
      int pinSou = ShapeTables.twoSuits(pin, sou);
      int code = hand.code(Block.MAN);
      for (long rest = manCandidates; rest != 0; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        int suits = ShapeTables.threeSuits(pinSou, manKey(code + Block.weight(kind)));
        if (ShapeTables.distance(suits, honor, melds) < best) {
          improving |= Tiles.bit(kind);
        }
      }
    }
    long pinCandidates = candidates & PIN_KINDS;
    if (pinCandidates != 0) {
      int manSou = ShapeTables.twoSuits(man, sou);
      int code = hand.code(Block.PIN);
      for (long rest = pinCandidates; rest != 0; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        int suits = ShapeTables.threeSuits(manSou, ShapeTables.suitKey(code + Block.weight(kind)));
        if (ShapeTables.distance(suits, honor, melds) < best) {
          improving |= Tiles.bit(kind);
        }
      }
    }
    int manPin = ShapeTables.twoSuits(man, pin);
    long souCandidates = candidates & SOU_KINDS;
    if (souCandidates != 0) {
      int code = hand.code(Block.SOU);
      for (long rest = souCandidates; rest != 0; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        int suits = ShapeTables.threeSuits(manPin, ShapeTables.suitKey(code + Block.weight(kind)));
        if (ShapeTables.distance(suits, honor, melds) < best) {
          improving |= Tiles.bit(kind);
        }
      }
    }
    long honorCandidates = candidates & HONOR_KINDS;
    if (honorCandidates != 0) {
      int suits = ShapeTables.threeSuits(manPin, sou);
      int code = hand.code(Block.HONOR);
      for (long rest = honorCandidates; rest != 0; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        if (ShapeTables.distance(suits, ShapeTables.honorKey(code + Block.weight(kind)), melds) < best) {
          improving |= Tiles.bit(kind);
        }
      }
    }
    return improving;
  }

  /**
   * 受け入れ枚数（有効牌の残りの枚数の和）。
   *
   * @param visible 種類ごとに見えている枚数（河・副露・表示牌・抜いた北）
   */
  public int acceptance(HandShape hand, byte[] visible) {
    return remaining(hand, visible, improving(hand));
  }

  /**
   * 指定した種類の、残りの枚数の和。1 種類の残りの枚数は、その種類の牌の数 − 見えている枚数 − 純手牌にある枚数。ルールで使わない
   * 種類は数えない。
   *
   * @param visible 種類ごとに見えている枚数
   * @param selected 数える種類の集合
   */
  public int remaining(HandShape hand, byte[] visible, long selected) {
    int remaining = 0;
    for (long rest = selected & kinds; rest != 0; rest &= rest - 1) {
      int kind = Long.numberOfTrailingZeros(rest);
      remaining += tiles.copies(kind) - visible[kind] - hand.count(kind);
    }
    return remaining;
  }

  private int standardDistance(HandShape hand) {
    int man = manKey(hand.code(Block.MAN));
    int pin = ShapeTables.suitKey(hand.code(Block.PIN));
    int sou = ShapeTables.suitKey(hand.code(Block.SOU));
    int suits = ShapeTables.threeSuits(ShapeTables.twoSuits(man, pin), sou);
    int honor = ShapeTables.honorKey(hand.code(Block.HONOR));
    return ShapeTables.distance(suits, honor, Variant.MAX_MELDS - hand.melds());
  }

  private int manKey(int code) {
    return terminalMan ? ShapeTables.terminalManKey(code) : ShapeTables.suitKey(code);
  }

  /** 七対子に足りない枚数。1 枚だけの種類には 1 枚ずつ足し、種類が 7 つに満たなければ新しい種類を 2 枚ずつ足す。 */
  private static int chiitoitsuDistance(HandShape hand) {
    if (hand.melds() > 0) {
      return IMPOSSIBLE;
    }
    int missingKinds = Math.max(0, HandShape.SEVEN_PAIRS - Long.bitCount(hand.kinds()));
    return HandShape.SEVEN_PAIRS - Long.bitCount(hand.pairs()) + missingKinds;
  }

  private static int kokushiDistance(HandShape hand) {
    if (hand.melds() > 0) {
      return IMPOSSIBLE;
    }
    int orphans = Long.bitCount(hand.kinds() & Tiles.TERMINALS_AND_HONORS);
    int pair = (hand.pairs() & Tiles.TERMINALS_AND_HONORS) != 0 ? 1 : 0;
    return HandShape.MAX_SIZE - orphans - pair;
  }

  private static void requireDrawable(HandShape hand) {
    if (hand.size() >= HandShape.MAX_SIZE) {
      throw new IllegalArgumentException("A hand of 14 tiles cannot draw: " + hand);
    }
  }

  private static void requireConcealed(HandShape hand) {
    if (hand.melds() > 0) {
      throw new IllegalArgumentException("A hand with melds has no seven pairs or thirteen orphans shape: " + hand);
    }
  }

  /** 引くと和了の形になりうる種類。手にある種類と、手にある 2 種類と組んで順子になる種類。 */
  private static long completions(long present) {
    long below = (present >>> 1) & (present >>> 2) & RUN_STARTS;
    long middle = (present & (present >>> 2) & RUN_STARTS) << 1;
    long above = (present & (present >>> 1) & RUN_STARTS) << 2;
    return present | below | middle | above;
  }

  private static long runStarts() {
    long starts = 0;
    for (Tiles.Group group : Tiles.Group.values()) {
      if (group.suited()) {
        starts |= Tiles.range(group.first(), group.last() - (HandShape.MELD_SIZE - 1));
      }
    }
    return starts;
  }

  /** {@link #analyze} が書く、1 つの手の向聴と有効牌。呼び出し元が 1 つ持って使い回す。同時に使えるのは 1 スレッドだけ。 */
  public static final class Analysis {

    private int shanten;
    private int standard;
    private int chiitoitsu;
    private int kokushi;
    private boolean concealed;
    private long improving;

    /** 3 つの形のうち、最も小さい向聴。 */
    public int shanten() {
      return shanten;
    }

    public int standardShanten() {
      return standard;
    }

    /** 七対子の向聴。面子のある手は七対子にならないので IllegalStateException を投げる。 */
    public int chiitoitsuShanten() {
      requireConcealed();
      return chiitoitsu;
    }

    /** 国士無双の向聴。面子のある手は国士無双にならないので IllegalStateException を投げる。 */
    public int kokushiShanten() {
      requireConcealed();
      return kokushi;
    }

    /** 引くと向聴が下がる種類（有効牌）。残りの枚数は見ない。 */
    public long improving() {
      return improving;
    }

    private void requireConcealed() {
      if (!concealed) {
        throw new IllegalStateException("A hand with melds has no seven pairs or thirteen orphans shape");
      }
    }
  }
}
