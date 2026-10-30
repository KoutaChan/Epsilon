package com.epsilon.prism.codec;

import java.util.Objects;

/**
 * {@link BitField} の enum から、記録の中の欄の位置を宣言の順に一度だけ作る。ビットの詰めと解きは、この型だけが行う。
 *
 * <p>記録は {@code unitBits}（8・16・32・64）の単位を {@link #units()} 個並べたもので、{@code long[]} の中に隙間なく続けて置く。
 * 記録 {@code r} は {@code words[base]} の先頭のビットから数えて {@code r × units() × unitBits} ビット目から始まる。欄は宣言の順に
 * 下位のビットから詰め、単位をまたがない（入らなければ次の単位の先頭に置く）。繰り返しのある欄は、繰り返しの番号の順に同じ規則で
 * 詰める。単位の幅は 64 を割り切るので、欄が {@code long} をまたぐこともない。
 *
 * <p>欄には値 − min を、max − min が入る最小の幅で書く。0 で埋めた記録の欄は min を読む。値が欄の範囲の外なら英語の例外を投げ、
 * 切り捨てはしない。繰り返しのある欄は番号つきの口で読み書きし、番号なしの口は繰り返しのない欄にだけ使える。読み書きは割り当てをしない。
 *
 * <p>共通の頭に種類ごとの欄が続く記録は、頭の配置に {@link #following} で続けた配置を種類ごとに作る。
 *
 * <p>記録が {@code long} の境をまたがない幅（64 を割り切る幅）なら、記録を 1 語として丸ごと読み書きできる（{@link #record}・
 * {@link #putRecord}）。欄は語の口（{@link #get(long, Enum)}・{@link #with}）で解いて詰めるので、欄ごとに配列を読み直さない。
 *
 * @param <E> 欄の enum
 */
public final class BitLayout<E extends Enum<E> & BitField> {

  private final Class<E> type;
  private final E[] fields;
  private final int unitBits;
  private final int units;
  private final int recordBits;
  private final long recordMask;
  private final int endBit;
  private final int[] first;
  private final int[] count;
  private final int[] position;
  private final int[] width;
  private final long[] mask;
  private final int[] min;
  private final int[] max;

  private BitLayout(Class<E> type, int unitBits, int startBit) {
    if (unitBits != 8 && unitBits != 16 && unitBits != 32 && unitBits != 64) {
      throw new IllegalArgumentException("unit must be 8, 16, 32 or 64 bits: " + unitBits);
    }
    this.type = Objects.requireNonNull(type, "type");
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
      long span = (long) field.max() - field.min();
      int bits = Long.SIZE - Long.numberOfLeadingZeros(span);
      if (span < 1 || bits > 31 || bits > unitBits) {
        throw new IllegalArgumentException(
            "invalid range " + field.min() + ".." + field.max() + " for " + name(f));
      }
      if (field.count() < 1) {
        throw new IllegalArgumentException("invalid count " + field.count() + " for " + name(f));
      }
      first[f] = slots;
      count[f] = field.count();
      width[f] = bits;
      mask[f] = (1L << bits) - 1;
      min[f] = field.min();
      max[f] = field.max();
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
   * 欄の enum を、記録の先頭から詰めた配置を作る。
   *
   * @param type 欄の enum
   * @param unitBits 単位の幅（8・16・32・64）
   * @param <E> 欄の enum
   * @return 配置
   * @throws IllegalArgumentException 単位の幅か、欄の範囲・繰り返しが不正なとき
   */
  public static <E extends Enum<E> & BitField> BitLayout<E> of(Class<E> type, int unitBits) {
    return new BitLayout<>(type, unitBits, 0);
  }

  /**
   * 共通の頭の配置に続けて、種類ごとの欄を詰めた配置を作る。単位の幅は頭と同じ。
   *
   * @param prefix 共通の頭の配置
   * @param type 種類ごとの欄の enum
   * @param <E> 欄の enum
   * @return 頭の最後の欄の後から詰めた配置
   */
  public static <E extends Enum<E> & BitField> BitLayout<E> following(
      BitLayout<?> prefix, Class<E> type) {
    return new BitLayout<>(type, prefix.unitBits, prefix.endBit);
  }

  /**
   * 記録 {@code record} の繰り返しのない欄を読む。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param field 欄
   * @return 値
   * @throws IllegalArgumentException 欄に繰り返しがあるとき
   */
  public int get(long[] words, int base, int record, E field) {
    int f = single(field);
    return read(words, base, record * recordBits + position[first[f]], f);
  }

  /**
   * 記録 {@code record} の欄の {@code index} 番目の繰り返しを読む。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param field 欄
   * @param index 繰り返しの番号（0 始まり）
   * @return 値
   */
  public int get(long[] words, int base, int record, E field, int index) {
    int f = field.ordinal();
    int slot = first[f] + Objects.checkIndex(index, count[f]);
    return read(words, base, record * recordBits + position[slot], f);
  }

  /**
   * 記録 {@code record} の繰り返しのある欄を、繰り返しの番号の順に全て読む。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param field 欄
   * @param values 読んだ値を入れる配列（先頭から {@link #count(Enum)} 個）
   * @throws IndexOutOfBoundsException 配列が繰り返しの数より短いとき
   */
  public void getAll(long[] words, int base, int record, E field, int[] values) {
    int f = field.ordinal();
    int repeats = count[f];
    Objects.checkFromIndexSize(0, repeats, values.length);
    int start = record * recordBits;
    for (int repeat = 0, slot = first[f]; repeat < repeats; repeat++, slot++) {
      values[repeat] = read(words, base, start + position[slot], f);
    }
  }

  /**
   * 記録 {@code record} を 1 語として、全てのビットを下位に寄せて返す。欄は {@link #get(long, Enum)} で解く。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @return 記録の語
   * @throws IllegalStateException 記録の幅が 64 を割り切らない（記録が語の境をまたぎうる）とき
   */
  public long record(long[] words, int base, int record) {
    requireAlignedWord();
    int bit = record * recordBits;
    return (words[base + (bit >>> 6)] >>> (bit & 63)) & recordMask;
  }

  /**
   * 記録 {@code record} の全てのビットを、{@link #with} で詰めた語で置き換える。ほかの記録は保つ。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param word 記録の語（記録の幅の外のビットは 0）
   * @throws IllegalStateException 記録の幅が 64 を割り切らないとき
   * @throws IllegalArgumentException 語が記録の幅の外にビットを持つとき
   */
  public void putRecord(long[] words, int base, int record, long word) {
    requireAlignedWord();
    if ((word & ~recordMask) != 0L) {
      throw new IllegalArgumentException(
          type.getSimpleName() + " record word has bits beyond " + recordBits + ": " + word);
    }
    int bit = record * recordBits;
    int index = base + (bit >>> 6);
    int shift = bit & 63;
    words[index] = (words[index] & ~(recordMask << shift)) | (word << shift);
  }

  /**
   * 記録 {@code record} の繰り返しのない欄に書く。ほかの欄は保つ。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param field 欄
   * @param value 値
   * @throws IllegalArgumentException 欄に繰り返しがあるか、値が欄の範囲の外のとき
   */
  public void put(long[] words, int base, int record, E field, int value) {
    int f = single(field);
    write(words, base, record * recordBits + position[first[f]], f, value);
  }

  /**
   * 記録 {@code record} の欄の {@code index} 番目の繰り返しに書く。ほかの欄は保つ。
   *
   * @param words 記録を並べた配列
   * @param base 最初の記録が始まる {@code long} の添字
   * @param record 記録の番号（0 始まり）
   * @param field 欄
   * @param index 繰り返しの番号（0 始まり）
   * @param value 値
   * @throws IllegalArgumentException 値が欄の範囲の外のとき
   */
  public void put(long[] words, int base, int record, E field, int index, int value) {
    int f = field.ordinal();
    int slot = first[f] + Objects.checkIndex(index, count[f]);
    write(words, base, record * recordBits + position[slot], f, value);
  }

  /**
   * 1 語に収まる記録（{@code units() × unitBits ≤ 64}）の繰り返しのない欄を読む。
   *
   * @param word 記録の語
   * @param field 欄
   * @return 値
   * @throws IllegalArgumentException 欄に繰り返しがあるとき
   */
  public int get(long word, E field) {
    requireSingleWord();
    int f = single(field);
    return (int) ((word >>> position[first[f]]) & mask[f]) + min[f];
  }

  /**
   * 1 語に収まる記録の繰り返しのない欄を書いた語を返す。ほかの欄は保つ。
   *
   * @param word 元の語
   * @param field 欄
   * @param value 値
   * @return 書いた後の語
   * @throws IllegalArgumentException 欄に繰り返しがあるか、値が欄の範囲の外のとき
   */
  public long with(long word, E field, int value) {
    requireSingleWord();
    int f = single(field);
    int shift = position[first[f]];
    return (word & ~(mask[f] << shift)) | (stored(f, value) << shift);
  }

  /**
   * 1 つの記録が使う単位の数を返す。
   *
   * @return 単位の数
   */
  public int units() {
    return units;
  }

  /**
   * 単位の幅を返す。
   *
   * @return ビットの数
   */
  public int unitBits() {
    return unitBits;
  }

  /**
   * {@code records} 個の記録を置くのに要る {@code long} の数を返す。
   *
   * @param records 記録の数
   * @return {@code long} の数
   */
  public int longs(int records) {
    return (int) (((long) records * recordBits + 63) >>> 6);
  }

  /**
   * 欄の幅を返す。
   *
   * @param field 欄
   * @return ビットの数
   */
  public int bits(E field) {
    return width[field.ordinal()];
  }

  /**
   * 欄の繰り返しの数を返す。
   *
   * @param field 欄
   * @return 繰り返しの数
   */
  public int count(E field) {
    return count[field.ordinal()];
  }

  /**
   * 欄の最小の値を返す。0 で埋めた記録の欄はこの値を読む。
   *
   * @param field 欄
   * @return 書ける最小の値
   */
  public int min(E field) {
    return min[field.ordinal()];
  }

  /**
   * 欄の最大の値を返す。
   *
   * @param field 欄
   * @return 書ける最大の値
   */
  public int max(E field) {
    return max[field.ordinal()];
  }

  /**
   * 指紋に入れる配置の記述を返す（欄の名前・範囲・繰り返し・位置と単位）。
   *
   * @return 記述
   */
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
    return (int) ((words[base + (bit >>> 6)] >>> (bit & 63)) & mask[f]) + min[f];
  }

  private void write(long[] words, int base, int bit, int f, int value) {
    long stored = stored(f, value);
    int word = base + (bit >>> 6);
    int shift = bit & 63;
    words[word] = (words[word] & ~(mask[f] << shift)) | (stored << shift);
  }

  /** 欄に書くビット（値 − min）を返す。 */
  private long stored(int f, int value) {
    if (value < min[f] || value > max[f]) {
      throw new IllegalArgumentException(
          name(f) + " is outside " + min[f] + ".." + max[f] + ": " + value);
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
      throw new IllegalStateException(
          type.getSimpleName() + " records of " + recordBits + " bits can straddle words");
    }
  }

  private String name(int f) {
    return type.getSimpleName() + "." + fields[f].name();
  }
}
