package com.epsilon.agent.observe.codec;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * {@link BitField} の enum から、記録の中のフィールドの位置を一度だけ決める。位置は、ルールごとのフィールドの範囲から宣言の順に
 * 決める。ビットを詰める処理と取り出す処理は、この型だけが持つ。
 *
 * <p>記録は、単位（8・16・32・64 ビット）を {@link #units()} 個並べたもので、{@code long[]} の中に隙間なく続けて置く。r 番目の
 * 記録は、{@code words[base]} の最下位のビットから数えて r × units × unitBits ビット目から始まる。フィールドは宣言の順に下位の
 * ビットから詰め、単位をまたがない（入らなければ次の単位の先頭に置く）。繰り返しのあるフィールドは、繰り返しの番号の順に同じやり方で
 * 詰める。単位の幅は 64 を割り切るので、フィールドが {@code long} をまたぐことも無い。
 *
 * <p>フィールドには 値 − min を書くので、0 で埋めた記録のフィールドは min として読める。値がフィールドの範囲の外なら、切り捨てずに
 * 例外にする。繰り返しのあるフィールドは番号を取るメソッドで読み書きする。番号を取らないメソッドは、繰り返しの無いフィールドにだけ
 * 使える。読み書きでメモリは確保しない。
 *
 * <p>共通の先頭部分の後に種類ごとのフィールドが続く記録は、先頭部分の配置に {@link #following} で続けた配置を種類ごとに作る。記録の
 * 幅が 64 を割り切るなら、記録を {@code long} 1 つとして丸ごと読み書きできる（{@link #record}・{@link #putRecord}）。そのときは、
 * {@code long} 1 つを扱うメソッド（{@link #get(long, Enum)}・{@link #with}）でフィールドを取り出したり詰めたりする。
 */
public final class BitLayout<E extends Enum<E> & BitField> {

  private static final int WORD_SHIFT = 6;
  private static final int WORD_MASK = Long.SIZE - 1;

  /** フィールドの値の幅（max − min）を表すビット数の上限。値は {@code int} で読むので、符号のビットを残す。 */
  private static final int MAX_FIELD_BITS = Integer.SIZE - 1;

  private final Class<E> type;
  private final Variant variant;
  private final E[] fields;
  private final int unitBits;
  private final int units;
  private final int recordBits;
  private final long recordMask;
  private final int endBit;

  /** フィールドごとに、最初の繰り返しが入るスロットの番号（フィールドの ordinal の順）。 */
  private final int[] first;

  private final int[] count;

  /** スロットごとに、記録の先頭からのビットの位置。 */
  private final int[] position;

  private final int[] width;
  private final long[] mask;
  private final int[] min;
  private final int[] max;

  private BitLayout(Class<E> type, Variant variant, int unitBits, int startBit) {
    if (unitBits != Byte.SIZE && unitBits != Short.SIZE && unitBits != Integer.SIZE && unitBits != Long.SIZE) {
      throw new IllegalArgumentException("Unit must be 8, 16, 32 or 64 bits: " + unitBits);
    }
    this.type = Objects.requireNonNull(type, "type");
    this.variant = Objects.requireNonNull(variant, "variant");
    this.unitBits = unitBits;
    fields = type.getEnumConstants();
    first = new int[fields.length];
    count = new int[fields.length];
    width = new int[fields.length];
    mask = new long[fields.length];
    min = new int[fields.length];
    max = new int[fields.length];
    int slots = 0;
    for (E field : fields) {
      int f = field.ordinal();
      min[f] = field.min(variant);
      max[f] = field.max(variant);
      count[f] = field.count(variant);
      long span = (long) max[f] - min[f];
      int bits = Long.SIZE - Long.numberOfLeadingZeros(span);
      if (span < 1 || bits > MAX_FIELD_BITS || bits > unitBits) {
        throw new IllegalArgumentException("Invalid range " + min[f] + ".." + max[f] + " for " + name(f));
      }
      if (count[f] < 1) {
        throw new IllegalArgumentException("Invalid count " + count[f] + " for " + name(f));
      }
      first[f] = slots;
      width[f] = bits;
      mask[f] = (1L << bits) - 1;
      slots += count[f];
    }
    position = new int[slots];
    int bit = startBit;
    for (E field : fields) {
      int f = field.ordinal();
      for (int repeat = 0; repeat < count[f]; repeat++) {
        int unitEnd = (bit / unitBits + 1) * unitBits;
        if (bit + width[f] > unitEnd) {
          bit = unitEnd;
        }
        position[first[f] + repeat] = bit;
        bit += width[f];
      }
    }
    endBit = bit;
    units = Math.max(1, (bit + unitBits - 1) / unitBits);
    recordBits = units * unitBits;
    recordMask = recordBits >= Long.SIZE ? -1L : (1L << recordBits) - 1;
  }

  /**
   * 指定したルールの範囲で、フィールドの enum を記録の先頭から詰めた配置を作る。単位の幅、フィールドの範囲、繰り返しの数のどれかが
   * 不正なら IllegalArgumentException を投げる。
   */
  public static <E extends Enum<E> & BitField> BitLayout<E> of(Class<E> type, int unitBits, Variant variant) {
    return new BitLayout<>(type, variant, unitBits, 0);
  }

  /** 共通の先頭部分に続けて、種類ごとのフィールドを詰めた配置を作る。ルールと単位の幅は、先頭部分の配置と同じ。 */
  public static <E extends Enum<E> & BitField> BitLayout<E> following(BitLayout<?> prefix, Class<E> type) {
    return new BitLayout<>(type, prefix.variant, prefix.unitBits, prefix.endBit);
  }

  /**
   * 指定した記録の、繰り返しの無いフィールドを読む。
   *
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号
   */
  public int get(long[] words, int base, int record, E field) {
    int f = single(field);
    return read(words, base, record * recordBits + position[first[f]], f);
  }

  /** 指定した記録のフィールドのうち、指定した番号の繰り返しを読む。 */
  public int get(long[] words, int base, int record, E field, int index) {
    int f = field.ordinal();
    int slot = first[f] + Objects.checkIndex(index, count[f]);
    return read(words, base, record * recordBits + position[slot], f);
  }

  /** 指定した記録の繰り返しのあるフィールドを、繰り返しの番号の順に全て読み、渡した配列の先頭から書く。 */
  public void getAll(long[] words, int base, int record, E field, int[] values) {
    int f = field.ordinal();
    int repeats = count[f];
    Objects.checkFromIndexSize(0, repeats, values.length);
    int start = record * recordBits;
    for (int repeat = 0, slot = first[f]; repeat < repeats; repeat++, slot++) {
      values[repeat] = read(words, base, start + position[slot], f);
    }
  }

  /** 指定した記録の全てのビットを、下位に寄せた {@code long} で返す。記録の幅が 64 を割り切らなければ IllegalStateException を投げる。 */
  public long record(long[] words, int base, int record) {
    requireAlignedWord();
    int bit = record * recordBits;
    return (words[base + (bit >>> WORD_SHIFT)] >>> (bit & WORD_MASK)) & recordMask;
  }

  /**
   * 指定した記録の全てのビットを、{@link #with} で詰めた {@code long} で置き換える。ほかの記録はそのまま残す。記録の幅が 64 を
   * 割り切らなければ IllegalStateException を、渡した {@code long} が記録の幅の外にビットを持てば IllegalArgumentException を投げる。
   */
  public void putRecord(long[] words, int base, int record, long word) {
    requireAlignedWord();
    if ((word & ~recordMask) != 0L) {
      throw new IllegalArgumentException(type.getSimpleName() + " record word has bits beyond " + recordBits + ": " + word);
    }
    int bit = record * recordBits;
    int index = base + (bit >>> WORD_SHIFT);
    int shift = bit & WORD_MASK;
    words[index] = (words[index] & ~(recordMask << shift)) | (word << shift);
  }

  /** 指定した記録の、繰り返しの無いフィールドに書く。ほかのフィールドはそのまま残す。 */
  public void put(long[] words, int base, int record, E field, int value) {
    int f = single(field);
    write(words, base, record * recordBits + position[first[f]], f, value);
  }

  /** 指定した記録のフィールドのうち、指定した番号の繰り返しに書く。ほかのフィールドはそのまま残す。 */
  public void put(long[] words, int base, int record, E field, int index, int value) {
    int f = field.ordinal();
    int slot = first[f] + Objects.checkIndex(index, count[f]);
    write(words, base, record * recordBits + position[slot], f, value);
  }

  /** {@code long} 1 つに収まる記録の、繰り返しの無いフィールドを読む。 */
  public int get(long word, E field) {
    requireSingleWord();
    int f = single(field);
    return (int) ((word >>> position[first[f]]) & mask[f]) + min[f];
  }

  /** {@code long} 1 つに収まる記録の、繰り返しの無いフィールドに書いた結果を返す。ほかのフィールドはそのまま残す。 */
  public long with(long word, E field, int value) {
    requireSingleWord();
    int f = single(field);
    int shift = position[first[f]];
    return (word & ~(mask[f] << shift)) | (stored(f, value) << shift);
  }

  /** 記録 1 つに使う単位の数。 */
  public int units() {
    return units;
  }

  /** 指定した数の記録を置くのに要る {@code long} の数。 */
  public int longs(int records) {
    return (int) (((long) records * recordBits + WORD_MASK) >>> WORD_SHIFT);
  }

  /** フィールドの幅（ビットの数）。 */
  public int bits(E field) {
    return width[field.ordinal()];
  }

  public int count(E field) {
    return count[field.ordinal()];
  }

  /** 書ける最小の値。0 で埋めた記録のフィールドは、この値として読める。 */
  public int min(E field) {
    return min[field.ordinal()];
  }

  public int max(E field) {
    return max[field.ordinal()];
  }

  /** フィンガープリントに入れる、配置の記述。型の名前、単位、フィールドごとの名前・範囲・繰り返し・位置を書く。 */
  public String describe() {
    StringBuilder out = new StringBuilder(SchemaDigest.name(type));
    out.append("{unit=").append(unitBits).append(",units=").append(units);
    for (E field : fields) {
      int f = field.ordinal();
      out.append(',').append(field.name()).append(':').append(min[f]).append("..").append(max[f]);
      if (count[f] > 1) {
        out.append('x').append(count[f]);
      }
      out.append('@');
      for (int repeat = 0; repeat < count[f]; repeat++) {
        if (repeat > 0) {
          out.append('/');
        }
        out.append(position[first[f] + repeat]);
      }
    }
    return out.append('}').toString();
  }

  private int read(long[] words, int base, int bit, int f) {
    return (int) ((words[base + (bit >>> WORD_SHIFT)] >>> (bit & WORD_MASK)) & mask[f]) + min[f];
  }

  private void write(long[] words, int base, int bit, int f, int value) {
    long stored = stored(f, value);
    int word = base + (bit >>> WORD_SHIFT);
    int shift = bit & WORD_MASK;
    words[word] = (words[word] & ~(mask[f] << shift)) | (stored << shift);
  }

  /** フィールドに書くビット（値 − min）。 */
  private long stored(int f, int value) {
    if (value < min[f] || value > max[f]) {
      throw new IllegalArgumentException(name(f) + " is outside " + min[f] + ".." + max[f] + ": " + value);
    }
    return (long) value - min[f];
  }

  private int single(E field) {
    int f = field.ordinal();
    if (count[f] != 1) {
      throw new IllegalArgumentException(name(f) + " repeats " + count[f] + " times");
    }
    return f;
  }

  private void requireSingleWord() {
    if (recordBits > Long.SIZE) {
      throw new IllegalStateException(type.getSimpleName() + " spans " + units + " units");
    }
  }

  private void requireAlignedWord() {
    if (Long.SIZE % recordBits != 0) {
      throw new IllegalStateException(type.getSimpleName() + " records of " + recordBits + " bits can straddle words");
    }
  }

  private String name(int f) {
    return type.getSimpleName() + "." + fields[f].name();
  }
}
