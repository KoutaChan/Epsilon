package com.epsilon.calculate.table;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;

/** 読み取り専用の配列を一度だけ読み込む。起動時のインデックス展開や特徴生成は行わない。 */
final class ShapeTableLoader {
  private static final int MAX_ELEMENTS = 32_000_000;
  private static final int[] POWERS = {1, 5, 25, 125, 625, 3125, 15625, 78125, 390625};

  private static final class Holder {
    private static final ShapeTableLoader TABLE = load();
  }

  final byte[] suitKeys;
  final byte[] honorKeys;
  final int suitKeyCount;
  final int honorKeyCount;
  final byte[] mergeTwo;
  final byte[] mergeThree;
  final byte[] distances;
  final char[] suitStates;
  final char[] honorStates;
  final int[] suitRanges;
  final int[] honorRanges;
  final int[] suitPatterns;
  final int[] honorPatterns;
  final long[] suitWaits;
  final long[] honorWaits;
  final long[] suitFeatures;
  final long[] honorFeatures;

  private ShapeTableLoader(DataInputStream in) throws IOException {
    int[] dims = readInts(in, 1, 4);
    for (int n : dims) if (n < 1 || n >= 255) throw new IOException("Invalid key dimensions");
    suitKeys = readBytes(in, 2, ShapeTableFormat.SUIT_CODES);
    honorKeys = readBytes(in, 3, ShapeTableFormat.HONOR_CODES);
    suitKeyCount = dims[0];
    honorKeyCount = dims[1];
    mergeTwo = readBytes(in, 4, Math.multiplyExact(dims[0], dims[0]));
    mergeThree = readBytes(in, 5, Math.multiplyExact(dims[2], dims[0]));
    distances = readBytes(in, 6, Math.multiplyExact(Math.multiplyExact(dims[3], dims[1]), 5));
    suitStates = readChars(in, 7, ShapeTableFormat.SUIT_CODES);
    honorStates = readChars(in, 8, ShapeTableFormat.HONOR_CODES);
    suitRanges = readInts(in, 9, -1);
    honorRanges = readInts(in, 10, -1);
    suitPatterns = readInts(in, 11, -1);
    honorPatterns = readInts(in, 12, -1);
    suitWaits = readLongs(in, 13, suitPatterns.length);
    honorWaits = readLongs(in, 14, honorPatterns.length);
    suitFeatures = readLongs(in, 15, suitPatterns.length);
    honorFeatures = readLongs(in, 16, honorPatterns.length);
    validateKeys(suitKeys, dims[0], true);
    validateKeys(honorKeys, dims[1], true);
    validateKeys(mergeTwo, dims[2], false);
    validateKeys(mergeThree, dims[3], false);
    validateKeys(suitStates, suitRanges.length, false);
    validateKeys(honorStates, honorRanges.length, false);
    validateRanges(suitRanges, suitPatterns.length);
    validateRanges(honorRanges, honorPatterns.length);
    for (byte distance : distances)
      if (distance < 0 || distance > 14) throw new IOException("Invalid distance");
  }

  static ShapeTableLoader table() {
    return Holder.TABLE;
  }

  /** ファイルの読み込み時に検証する。テーブルの生成処理から独立させ、破損した配列を返さない。 */
  static ShapeTableLoader read(InputStream source) throws IOException {
    CRC32 checksum = new CRC32();
    try (DataInputStream in =
        new DataInputStream(new CheckedInputStream(new BufferedInputStream(source), checksum))) {
      if (in.readInt() != ShapeTableFormat.MAGIC
          || in.readInt() != ShapeTableFormat.VERSION
          || in.readInt() != ShapeTableFormat.SECTION_COUNT)
        throw new IOException("Unsupported shape table header");
      ShapeTableLoader result;
      try {
        result = new ShapeTableLoader(in);
      } catch (ArithmeticException e) {
        throw new IOException("Oversized shape table dimensions", e);
      }
      long actual = checksum.getValue();
      if (in.readLong() != actual) throw new IOException("Shape table checksum mismatch");
      if (in.read() != -1) throw new IOException("Trailing shape table data");
      return result;
    }
  }

  private static ShapeTableLoader load() {
    InputStream source = ShapeTableLoader.class.getResourceAsStream(ShapeTableFormat.RESOURCE_PATH);
    if (source == null)
      throw new IllegalStateException("Missing shape table: " + ShapeTableFormat.RESOURCE_PATH);
    try {
      return read(source);
    } catch (IOException e) {
      throw new IllegalStateException("Cannot load shape table", e);
    }
  }

  static int suitCountCode(int[] counts, int offset) {
    return countCode(counts, offset, 9);
  }

  static int honorCountCode(int[] counts) {
    return countCode(counts, 27, 7);
  }

  static int suitKey(int[] counts, int offset) {
    return suitKey(suitCountCode(counts, offset));
  }

  static int honorKey(int[] counts) {
    return honorKey(honorCountCode(counts));
  }

  static int suitKey(int code) {
    return checkedKey(table().suitKeys, code);
  }

  static int honorKey(int code) {
    return checkedKey(table().honorKeys, code);
  }

  static int suitKeyUnchecked(int code) {
    return table().suitKeys[code] & 0xff;
  }

  static int honorKeyUnchecked(int code) {
    return table().honorKeys[code] & 0xff;
  }

  static int suitKeyAfterAdding(int code, int rank) {
    return suitKeyUnchecked(code + POWERS[rank]);
  }

  static int honorKeyAfterAdding(int code, int rank) {
    return honorKeyUnchecked(code + POWERS[rank]);
  }

  static int standardShanten(int man, int pin, int sou, int honor, int meldTarget) {
    ShapeTableLoader t = table();
    int two = t.mergeTwo[man * t.suitKeyCount + pin] & 0xff;
    int three = t.mergeThree[two * t.suitKeyCount + sou] & 0xff;
    return t.distances[(three * t.honorKeyCount + honor) * 5 + meldTarget] - 1;
  }

  static char[] suitPatternStatesByCountCode() {
    return table().suitStates;
  }

  static char[] honorPatternStatesByCountCode() {
    return table().honorStates;
  }

  static int[] suitPatternRangesByState() {
    return table().suitRanges;
  }

  static int[] honorPatternRangesByState() {
    return table().honorRanges;
  }

  static int[] suitPatterns() {
    return table().suitPatterns;
  }

  static int[] honorPatterns() {
    return table().honorPatterns;
  }

  static long[] suitMachiTypeMatrices() {
    return table().suitWaits;
  }

  static long[] honorMachiTypeMatrices() {
    return table().honorWaits;
  }

  static long[] suitFeaturePackets() {
    return table().suitFeatures;
  }

  static long[] honorFeaturePackets() {
    return table().honorFeatures;
  }

  private static int countCode(int[] counts, int offset, int size) {
    if (counts == null || offset < 0 || offset + size > counts.length)
      throw new IllegalArgumentException("Invalid count array");
    int code = 0;
    for (int rank = 0; rank < size; rank++) {
      int count = counts[offset + rank];
      if (count < 0 || count > 4)
        throw new IllegalArgumentException("Invalid tile count: " + count);
      code += count * POWERS[rank];
    }
    return code;
  }

  private static int checkedKey(byte[] keys, int code) {
    if (code < 0 || code >= keys.length || (keys[code] & 0xff) == 0xff)
      throw new IllegalArgumentException("Invalid count code: " + code);
    return keys[code] & 0xff;
  }

  private static void validateKeys(byte[] keys, int bound, boolean invalidAllowed)
      throws IOException {
    for (byte value : keys) {
      int key = value & 0xff;
      if (key >= bound && (!invalidAllowed || key != 0xff))
        throw new IOException("Shape table index outside target array");
    }
  }

  private static void validateKeys(char[] keys, int bound, boolean invalidAllowed)
      throws IOException {
    for (char key : keys)
      if (key >= bound && (!invalidAllowed || key != 0xff))
        throw new IOException("Shape table index outside target array");
  }

  private static void validateRanges(int[] ranges, int patternCount) throws IOException {
    if (ranges.length == 0 || ranges[0] != 0) throw new IOException("Invalid empty pattern range");
    for (int i = 1; i < ranges.length; i++) {
      int count = ranges[i] & 31;
      int start = (ranges[i] & 0x7fffffff) >>> 5;
      if (count == 0 || start > patternCount - count)
        throw new IOException("Invalid decomposition range");
    }
  }

  private static int section(DataInputStream in, int id, int width, int expected)
      throws IOException {
    if (in.readInt() != id || in.readInt() != width)
      throw new IOException("Unexpected shape table section " + id);
    int n = in.readInt();
    if (n < 0 || n > MAX_ELEMENTS || expected >= 0 && n != expected)
      throw new IOException("Invalid section length " + id);
    return n;
  }

  private static char[] readChars(DataInputStream in, int id, int n) throws IOException {
    char[] data = new char[section(in, id, 2, n)];
    for (int i = 0; i < data.length; i++) data[i] = in.readChar();
    return data;
  }

  private static int[] readInts(DataInputStream in, int id, int n) throws IOException {
    int[] data = new int[section(in, id, 4, n)];
    for (int i = 0; i < data.length; i++) data[i] = in.readInt();
    return data;
  }

  private static long[] readLongs(DataInputStream in, int id, int n) throws IOException {
    long[] data = new long[section(in, id, 8, n)];
    for (int i = 0; i < data.length; i++) data[i] = in.readLong();
    return data;
  }

  private static byte[] readBytes(DataInputStream in, int id, int n) throws IOException {
    byte[] data = new byte[section(in, id, 1, n)];
    in.readFully(data);
    return data;
  }
}
