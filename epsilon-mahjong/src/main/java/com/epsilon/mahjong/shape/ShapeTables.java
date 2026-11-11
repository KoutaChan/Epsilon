package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.shape.ShapeSection.Element;
import com.epsilon.mahjong.tile.Tiles;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.zip.GZIPInputStream;

/**
 * 形の表（{@link ShapeSection}）。クラスの初期化のときにリソースを 1 回だけ読む。メソッドは引数の範囲を確かめないので、渡す
 * コードは {@link HandShape} が持つ手のコードに限る。
 */
final class ShapeTables {

  private static final Object[] SECTIONS = read();

  private static final byte[] SUIT_KEY = (byte[]) section(ShapeSection.SUIT_KEY);
  private static final byte[] HONOR_KEY = (byte[]) section(ShapeSection.HONOR_KEY);
  private static final byte[] TERMINAL_MAN_KEY = (byte[]) section(ShapeSection.TERMINAL_MAN_KEY);
  private static final byte[] TWO_SUITS = (byte[]) section(ShapeSection.TWO_SUITS);
  private static final byte[] THREE_SUITS = (byte[]) section(ShapeSection.THREE_SUITS);
  private static final byte[] DISTANCE = (byte[]) section(ShapeSection.DISTANCE);
  private static final char[] SUIT_STATE = (char[]) section(ShapeSection.SUIT_STATE);
  private static final char[] HONOR_STATE = (char[]) section(ShapeSection.HONOR_STATE);
  private static final int[] FIRST = (int[]) section(ShapeSection.FIRST);
  private static final byte[] PAIR = (byte[]) section(ShapeSection.PAIR);
  private static final byte[] MELD = (byte[]) section(ShapeSection.MELD);

  /** 数牌のキーの数（2 色・3 色の表の行の幅）。 */
  private static final int SUIT_KEYS = Math.max(keys(SUIT_KEY), keys(TERMINAL_MAN_KEY));

  private static final int HONOR_KEYS = keys(HONOR_KEY);

  /** 萬子のコードをこれで割ると、5 × 9m の枚数になる。1m の枚数は 5 未満なので、割ったときの切り捨てで消える。 */
  private static final int NINE_MAN_DIVISOR = Block.weight(Tiles.M9) / Block.RADIX;

  static {
    require(ShapeSection.SUIT_KEY, SUIT_KEY.length, Block.MAN.codes());
    require(ShapeSection.HONOR_KEY, HONOR_KEY.length, Block.HONOR.codes());
    require(ShapeSection.TERMINAL_MAN_KEY, TERMINAL_MAN_KEY.length, Block.RADIX * Block.RADIX);
    require(ShapeSection.TWO_SUITS, TWO_SUITS.length, SUIT_KEYS * SUIT_KEYS);
    require(ShapeSection.THREE_SUITS, THREE_SUITS.length, keys(TWO_SUITS) * SUIT_KEYS);
    require(
        ShapeSection.DISTANCE, DISTANCE.length, keys(THREE_SUITS) * HONOR_KEYS * ShapeSection.MELD_TARGETS);
    require(ShapeSection.SUIT_STATE, SUIT_STATE.length, Block.MAN.codes());
    require(ShapeSection.HONOR_STATE, HONOR_STATE.length, Block.HONOR.codes());
    require(ShapeSection.FIRST, FIRST.length, Math.max(states(SUIT_STATE), states(HONOR_STATE)) + 1);
    require(ShapeSection.PAIR, PAIR.length, FIRST[FIRST.length - 1]);
    require(ShapeSection.MELD, MELD.length, PAIR.length * Variant.MAX_MELDS);
  }

  private ShapeTables() {}

  static int suitKey(int code) {
    return Byte.toUnsignedInt(SUIT_KEY[code]);
  }

  static int honorKey(int code) {
    return Byte.toUnsignedInt(HONOR_KEY[code]);
  }

  /** 萬子が 1m・9m だけのルールで、萬子のコードから数牌のキーを引く。 */
  static int terminalManKey(int manCode) {
    return Byte.toUnsignedInt(TERMINAL_MAN_KEY[manCode % Block.RADIX + manCode / NINE_MAN_DIVISOR]);
  }

  static int twoSuits(int key, int other) {
    return Byte.toUnsignedInt(TWO_SUITS[key * SUIT_KEYS + other]);
  }

  static int threeSuits(int twoSuits, int key) {
    return Byte.toUnsignedInt(THREE_SUITS[twoSuits * SUIT_KEYS + key]);
  }

  /**
   * 面子と雀頭 1 つを作るのに足りない最小の枚数。
   *
   * @param threeSuits 3 色のキー
   * @param honorKey 字牌のキー
   * @param melds 作る面子の数
   */
  static int distance(int threeSuits, int honorKey, int melds) {
    return DISTANCE[(threeSuits * HONOR_KEYS + honorKey) * ShapeSection.MELD_TARGETS + melds];
  }

  /** ブロックのコードから分解の状態を引く。0 は分解が無いことを表す。 */
  static int state(Block block, int code) {
    return block.runs() ? SUIT_STATE[code] : HONOR_STATE[code];
  }

  /** その状態で最初の分解の番号。最後の分解の次の番号は、次の状態の最初の番号と同じ。 */
  static int first(int state) {
    return FIRST[state];
  }

  /** 分解に雀頭があるか。雀頭の有無はブロックの枚数を 3 で割った余りで決まるので、同じ状態の分解ではどれも同じ。 */
  static boolean hasPair(int decomposition) {
    return PAIR[decomposition] >= 0;
  }

  /** 分解の雀頭の、ブロックの中での番号。雀頭のある分解でだけ使う。 */
  static int pair(int decomposition) {
    return PAIR[decomposition];
  }

  /**
   * 分解の面子の、ブロックの中での番号（順子なら先頭の番号）。空きは −1 で、空きより後ろも全て空き。
   *
   * @param slot 何番目の面子か
   */
  static int meldStart(int decomposition, int slot) {
    int meld = MELD[decomposition * Variant.MAX_MELDS + slot];
    return meld >= ShapeSection.RUN ? meld - ShapeSection.RUN : meld;
  }

  /** 分解の面子が順子か。空きなら false。 */
  static boolean meldRun(int decomposition, int slot) {
    return MELD[decomposition * Variant.MAX_MELDS + slot] >= ShapeSection.RUN;
  }

  private static Object[] read() {
    try (InputStream resource = ShapeTables.class.getResourceAsStream(ShapeSection.RESOURCE)) {
      if (resource == null) {
        throw new IllegalStateException("Missing shape table resource " + ShapeSection.RESOURCE);
      }
      ByteBuffer data;
      try (GZIPInputStream unzipped = new GZIPInputStream(resource)) {
        data = ByteBuffer.wrap(unzipped.readAllBytes());
      }
      ShapeSection[] sections = ShapeSection.values();
      Object[] arrays = new Object[sections.length];
      for (ShapeSection section : sections) {
        arrays[section.ordinal()] = readSection(data, section.element());
      }
      if (data.hasRemaining()) {
        throw new IllegalStateException("Shape table has " + data.remaining() + " bytes after the last section");
      }
      return arrays;
    } catch (IOException e) {
      throw new UncheckedIOException("Cannot read the shape table " + ShapeSection.RESOURCE, e);
    }
  }

  private static Object readSection(ByteBuffer data, Element element) {
    int length = data.getInt();
    ByteBuffer body = data.slice(data.position(), length * element.bytes());
    data.position(data.position() + body.capacity());
    return switch (element) {
      case BYTE -> {
        byte[] values = new byte[length];
        body.get(values);
        yield values;
      }
      case CHAR -> {
        char[] values = new char[length];
        body.asCharBuffer().get(values);
        yield values;
      }
      case INT -> {
        int[] values = new int[length];
        body.asIntBuffer().get(values);
        yield values;
      }
    };
  }

  private static Object section(ShapeSection section) {
    return SECTIONS[section.ordinal()];
  }

  /** キーの数（最大のキー ＋ 1）。 */
  private static int keys(byte[] keys) {
    int max = 0;
    for (byte key : keys) {
      max = Math.max(max, Byte.toUnsignedInt(key));
    }
    return max + 1;
  }

  /** 状態の数（最大の状態 ＋ 1）。 */
  private static int states(char[] states) {
    int max = 0;
    for (char state : states) {
      max = Math.max(max, state);
    }
    return max + 1;
  }

  private static void require(ShapeSection section, int length, int expected) {
    if (length != expected) {
      throw new IllegalStateException(
          "Shape table section " + section + " holds " + length + " values, expected " + expected);
    }
  }
}
