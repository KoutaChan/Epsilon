package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.tile.Tiles;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

/**
 * 手の形の表（ShapeSection の各セクション）を作る。
 *
 * <p>ブロックの d(m, p) は、ブロックの牌に足して面子 m 個と雀頭 p 個を作るのに要る最小の枚数。作る形では、どの種類も 4 枚
 * までしか使わない。d は、種類を小さい方から 1 つずつ決める動的計画で求める。状態は（1 つ前の種類から始めた順子の数、
 * 2 つ前の種類から始めた順子の数、置いた面子の数、雀頭の数）で、値はそこまでに足した枚数。ブロックの全てのコードは、
 * 小さい種類の桁から決める深さ優先でたどり、前の桁までの計算を使い回す。d が同じコードは 1 つのキーにまとめる。
 * ブロックどうしの d は (min, +) の畳み込みで合わせる。
 */
final class ShapeTableGenerator {

  /** 作れない (m, p) に入れる値。作れる形なら、足す枚数は 3 × 4 ＋ 2 = 14 枚以下に収まる。 */
  static final int IMPOSSIBLE = HandShape.MAX_SIZE + 1;

  private static final int PAIR_SIZE = 2;
  private static final int MAX_PAIRS = 1;

  /** 1 つの種類から始める順子は 0〜4 個。 */
  private static final int RUN_COUNTS = Tiles.COPIES + 1;

  private static final int STATES = RUN_COUNTS * RUN_COUNTS * ShapeSection.MELD_TARGETS * (MAX_PAIRS + 1);

  private ShapeTableGenerator() {}

  /** 表の各セクションを作り、ShapeSection の順に並べた配列を返す。 */
  static Object[] generate() {
    Object[] sections = new Object[ShapeSection.values().length];
    Keys suits = new Keys();
    sections[ShapeSection.SUIT_KEY.ordinal()] = keys(blockDistances(Block.MAN), suits);
    byte[] terminalMan = new byte[Block.RADIX * Block.RADIX];
    for (int nine = 0; nine < Block.RADIX; nine++) {
      for (int one = 0; one < Block.RADIX; one++) {
        terminalMan[one + Block.RADIX * nine] = unsigned(suits.key(distances(new int[] {one, nine}, false)));
      }
    }
    sections[ShapeSection.TERMINAL_MAN_KEY.ordinal()] = terminalMan;
    Keys honors = new Keys();
    sections[ShapeSection.HONOR_KEY.ordinal()] = keys(blockDistances(Block.HONOR), honors);
    Keys two = new Keys();
    sections[ShapeSection.TWO_SUITS.ordinal()] = merge(suits, suits, two);
    Keys three = new Keys();
    sections[ShapeSection.THREE_SUITS.ordinal()] = merge(two, suits, three);
    sections[ShapeSection.DISTANCE.ordinal()] = distance(three, honors);
    Decompositions decompositions = new Decompositions();
    sections[ShapeSection.SUIT_STATE.ordinal()] = decompositions.states(Block.MAN);
    sections[ShapeSection.HONOR_STATE.ordinal()] = decompositions.states(Block.HONOR);
    sections[ShapeSection.FIRST.ordinal()] = decompositions.first();
    sections[ShapeSection.PAIR.ordinal()] = decompositions.pairs();
    sections[ShapeSection.MELD.ordinal()] = decompositions.melds();
    return sections;
  }

  /** セクションを順に書いたバイト列を返す（gzip で圧縮する前）。 */
  static byte[] encode(Object[] sections) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    try (DataOutputStream out = new DataOutputStream(bytes)) {
      for (ShapeSection section : ShapeSection.values()) {
        Object values = sections[section.ordinal()];
        switch (section.element()) {
          case BYTE -> {
            byte[] array = (byte[]) values;
            out.writeInt(array.length);
            out.write(array);
          }
          case CHAR -> {
            char[] array = (char[]) values;
            out.writeInt(array.length);
            for (char value : array) {
              out.writeChar(value);
            }
          }
          case INT -> {
            int[] array = (int[]) values;
            out.writeInt(array.length);
            for (int value : array) {
              out.writeInt(value);
            }
          }
        }
      }
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    return bytes.toByteArray();
  }

  /** セクションのバイト列を gzip で圧縮してファイルに書く。 */
  static void write(byte[] encoded, Path file) throws IOException {
    Files.createDirectories(file.toAbsolutePath().getParent());
    try (OutputStream out = new GZIPOutputStream(Files.newOutputStream(file))) {
      out.write(encoded);
    }
  }

  /** ブロックの全てのコードについて d を求める（Distances の形）。15 枚以上のコードには −1 を入れる。 */
  static long[] blockDistances(Block block) {
    long[] distances = new long[block.codes()];
    Arrays.fill(distances, -1);
    int[][] states = new int[block.size() + 1][STATES];
    start(states[0]);
    visit(block, states, 0, 0, 0, distances);
    return distances;
  }

  /**
   * 1 つのブロックの d を求める（Distances の形）。
   *
   * @param counts ブロックの種類ごとの枚数
   * @param runs 隣り合う種類で順子を作れるか
   * @return d を詰めた値
   */
  static long distances(int[] counts, boolean runs) {
    int[] from = new int[STATES];
    int[] to = new int[STATES];
    start(from);
    for (int position = 0; position < counts.length; position++) {
      step(from, counts[position], runs && position + HandShape.MELD_SIZE <= counts.length, to);
      int[] swap = from;
      from = to;
      to = swap;
    }
    return Distances.of(from);
  }

  /**
   * ブロックのコードを、種類ごとの枚数に戻す。
   *
   * @param code ブロックのコード
   * @param digits 種類ごとの枚数を書く先（小さい種類から）
   * @return 枚数の和
   */
  static int digits(int code, int[] digits) {
    int tiles = 0;
    for (int position = 0; position < digits.length; position++) {
      digits[position] = code % Block.RADIX;
      tiles += digits[position];
      code /= Block.RADIX;
    }
    return tiles;
  }

  private static void visit(Block block, int[][] states, int position, int code, int tiles, long[] distances) {
    if (position == block.size()) {
      distances[code] = Distances.of(states[position]);
      return;
    }
    boolean runs = block.runs() && position + HandShape.MELD_SIZE <= block.size();
    int weight = Block.weight(block.first() + position);
    for (int count = 0; count <= Tiles.COPIES && tiles + count <= HandShape.MAX_SIZE; count++) {
      step(states[position], count, runs, states[position + 1]);
      visit(block, states, position + 1, code + count * weight, tiles + count, distances);
    }
  }

  private static void start(int[] states) {
    Arrays.fill(states, IMPOSSIBLE);
    states[state(0, 0, 0, 0)] = 0;
  }

  /**
   * 状態の番号を返す。
   *
   * @param previous 1 つ前の種類から始めた順子の数
   * @param before 2 つ前の種類から始めた順子の数
   * @param melds 置いた面子の数（作りかけの順子も数える）
   * @param pairs 雀頭の数
   */
  private static int state(int previous, int before, int melds, int pairs) {
    return ((previous * RUN_COUNTS + before) * ShapeSection.MELD_TARGETS + melds) * (MAX_PAIRS + 1) + pairs;
  }

  /**
   * 動的計画を 1 つの種類だけ進める。この種類では、順子をいくつ始めるか（始められるときだけ）と、刻子・雀頭をそれぞれ置くか
   * どうかを選ぶ。1 つ前と 2 つ前の種類から始めた順子の分と合わせて、この種類を 4 枚以下しか使わない組み合わせを全て試す。
   * 手に無い分の枚数は、足す枚数に加える。
   *
   * @param from この種類を決める前の、状態ごとに足した枚数
   * @param count 手にあるこの種類の枚数
   * @param runs この種類から順子を始められるか
   * @param to この種類を決めた後の、状態ごとに足した枚数を書く先
   */
  private static void step(int[] from, int count, boolean runs, int[] to) {
    Arrays.fill(to, IMPOSSIBLE);
    for (int previous = 0; previous < RUN_COUNTS; previous++) {
      for (int before = 0; previous + before <= Tiles.COPIES; before++) {
        for (int melds = 0; melds <= Variant.MAX_MELDS; melds++) {
          for (int pairs = 0; pairs <= MAX_PAIRS; pairs++) {
            int added = from[state(previous, before, melds, pairs)];
            if (added >= IMPOSSIBLE) {
              continue;
            }
            int maxStarted = runs ? Variant.MAX_MELDS - melds : 0;
            for (int started = 0; started <= maxStarted; started++) {
              for (int triplet = 0; triplet <= 1 && melds + started + triplet <= Variant.MAX_MELDS; triplet++) {
                for (int pair = 0; pairs + pair <= MAX_PAIRS; pair++) {
                  int used = started + previous + before + HandShape.MELD_SIZE * triplet + PAIR_SIZE * pair;
                  if (used > Tiles.COPIES) {
                    continue;
                  }
                  int next = state(started, previous, melds + started + triplet, pairs + pair);
                  to[next] = Math.min(to[next], added + Math.max(0, used - count));
                }
              }
            }
          }
        }
      }
    }
  }

  /** コードごとに、d のキーを並べた表を返す。15 枚以上のコードは 0 にする。 */
  private static byte[] keys(long[] distances, Keys keys) {
    byte[] codes = new byte[distances.length];
    for (int code = 0; code < distances.length; code++) {
      codes[code] = distances[code] < 0 ? 0 : unsigned(keys.key(distances[code]));
    }
    return codes;
  }

  /** 行のキーと列のキーの全ての組で 2 つの d を合わせ、合わせた d のキーを並べた表を返す。表の添字は（行のキー）×（列のキーの数）＋（列のキー）。 */
  private static byte[] merge(Keys rows, Keys columns, Keys merged) {
    byte[] table = new byte[rows.size() * columns.size()];
    for (int row = 0; row < rows.size(); row++) {
      for (int column = 0; column < columns.size(); column++) {
        long distances = Distances.merge(rows.value(row), columns.value(column));
        table[row * columns.size() + column] = unsigned(merged.key(distances));
      }
    }
    return table;
  }

  private static byte[] distance(Keys suits, Keys honors) {
    byte[] table = new byte[suits.size() * honors.size() * ShapeSection.MELD_TARGETS];
    for (int suit = 0; suit < suits.size(); suit++) {
      for (int honor = 0; honor < honors.size(); honor++) {
        long distances = Distances.merge(suits.value(suit), honors.value(honor));
        for (int melds = 0; melds < ShapeSection.MELD_TARGETS; melds++) {
          int value = Distances.get(distances, melds, MAX_PAIRS);
          if (value >= IMPOSSIBLE) {
            throw new IllegalStateException("No hand of " + melds + " melds and a pair for keys " + suit + ", " + honor);
          }
          table[(suit * honors.size() + honor) * ShapeSection.MELD_TARGETS + melds] = (byte) value;
        }
      }
    }
    return table;
  }

  private static byte unsigned(int key) {
    if (key > 0xFF) {
      throw new IllegalStateException("Key " + key + " does not fit a byte");
    }
    return (byte) key;
  }

  /** d の値ごとのキー。キーは初めて現れた順に振る。 */
  private static final class Keys {

    private final Map<Long, Integer> keys = new HashMap<>();
    private final List<Long> values = new ArrayList<>();

    int key(long distances) {
      Integer key = keys.get(distances);
      if (key == null) {
        key = values.size();
        keys.put(distances, key);
        values.add(distances);
      }
      return key;
    }

    long value(int key) {
      return values.get(key);
    }

    int size() {
      return values.size();
    }
  }

  /** 面子の数と雀頭の数の組ごとの d（値は 0〜{@code IMPOSSIBLE}）を、1 つの long に 4 bit ずつ詰める。 */
  static final class Distances {

    private static final int BITS = 4;
    private static final long MASK = (1L << BITS) - 1;

    private Distances() {}

    static int get(long distances, int melds, int pairs) {
      return (int) (distances >>> shift(melds, pairs) & MASK);
    }

    /** 2 つのブロックの d を (min, +) の畳み込みで合わせる。面子と雀頭を 2 つのブロックに分ける全ての分け方で、d の和の最小を取る。 */
    static long merge(long first, long second) {
      long merged = 0;
      for (int melds = 0; melds <= Variant.MAX_MELDS; melds++) {
        for (int pairs = 0; pairs <= MAX_PAIRS; pairs++) {
          int best = IMPOSSIBLE;
          for (int left = 0; left <= melds; left++) {
            for (int leftPairs = 0; leftPairs <= pairs; leftPairs++) {
              best = Math.min(best, get(first, left, leftPairs) + get(second, melds - left, pairs - leftPairs));
            }
          }
          merged |= (long) best << shift(melds, pairs);
        }
      }
      return merged;
    }

    /** 動的計画の最後の状態のうち、作りかけの順子が残っていない状態から d を作る。 */
    private static long of(int[] states) {
      long distances = 0;
      for (int melds = 0; melds <= Variant.MAX_MELDS; melds++) {
        for (int pairs = 0; pairs <= MAX_PAIRS; pairs++) {
          long value = Math.min(states[state(0, 0, melds, pairs)], IMPOSSIBLE);
          distances |= value << shift(melds, pairs);
        }
      }
      return distances;
    }

    private static int shift(int melds, int pairs) {
      return (melds * (MAX_PAIRS + 1) + pairs) * BITS;
    }
  }

  /** ブロックのコードごとに、面子と雀頭への分け方（分解）を集める。状態 0 は分解が無いことを表す。 */
  private static final class Decompositions {

    private final List<Integer> first = new ArrayList<>(List.of(0, 0));
    private final ByteArrayOutputStream pairOut = new ByteArrayOutputStream();
    private final ByteArrayOutputStream meldOut = new ByteArrayOutputStream();
    private final int[] melds = new int[Variant.MAX_MELDS];
    private int count;

    /** ブロックのコードごとに、分解の状態を返す。分解はコードの昇順に、状態ごとにまとめて並べる。 */
    char[] states(Block block) {
      char[] states = new char[block.codes()];
      int[] rest = new int[block.size()];
      for (int code = 0; code < states.length; code++) {
        int tiles = digits(code, rest);
        if (tiles > HandShape.MAX_SIZE || tiles % HandShape.MELD_SIZE == 1) {
          continue;
        }
        int before = count;
        decompose(block, rest, 0, -1, tiles % HandShape.MELD_SIZE == PAIR_SIZE, 0);
        if (count > before) {
          int state = first.size() - 1;
          if (state > Character.MAX_VALUE) {
            throw new IllegalStateException("State " + state + " does not fit a char");
          }
          states[code] = (char) state;
          first.add(count);
        }
      }
      return states;
    }

    int[] first() {
      return first.stream().mapToInt(Integer::intValue).toArray();
    }

    byte[] pairs() {
      return pairOut.toByteArray();
    }

    byte[] melds() {
      return meldOut.toByteArray();
    }

    /**
     * 分け始める種類から先の残りの牌を、面子と雀頭 1 つ（要るときだけ）にちょうど分ける分解を全て書く。各種類の枚数は、
     * 雀頭・刻子と、その種類から始める順子だけで使い切る。だから同じ分解を 2 度書くことはない。
     *
     * @param block 分けるブロック
     * @param rest 種類ごとの残りの枚数
     * @param position 分け始める種類の位置
     * @param pair 置いた雀頭の種類の位置（まだ置いていなければ −1）
     * @param needsPair 雀頭が要るか
     * @param meldCount 置いた面子の数
     */
    private void decompose(Block block, int[] rest, int position, int pair, boolean needsPair, int meldCount) {
      if (position == rest.length) {
        if (needsPair == (pair >= 0)) {
          write(pair, meldCount);
        }
        return;
      }
      boolean runs = block.runs() && position + HandShape.MELD_SIZE <= rest.length;
      int maxPairs = needsPair && pair < 0 ? MAX_PAIRS : 0;
      for (int pairs = 0; pairs <= maxPairs; pairs++) {
        for (int triplet = 0; triplet <= 1; triplet++) {
          int started = rest[position] - PAIR_SIZE * pairs - HandShape.MELD_SIZE * triplet;
          if (started < 0 || meldCount + triplet + started > Variant.MAX_MELDS) {
            continue;
          }
          if (started > 0 && (!runs || rest[position + 1] < started || rest[position + 2] < started)) {
            continue;
          }
          int placed = meldCount;
          if (triplet == 1) {
            melds[placed++] = position;
          }
          for (int run = 0; run < started; run++) {
            melds[placed++] = ShapeSection.RUN + position;
          }
          if (started > 0) {
            rest[position + 1] -= started;
            rest[position + 2] -= started;
          }
          decompose(block, rest, position + 1, pairs == 1 ? position : pair, needsPair, placed);
          if (started > 0) {
            rest[position + 1] += started;
            rest[position + 2] += started;
          }
        }
      }
    }

    private void write(int pair, int meldCount) {
      pairOut.write(pair);
      for (int slot = 0; slot < Variant.MAX_MELDS; slot++) {
        meldOut.write(slot < meldCount ? melds[slot] : -1);
      }
      count++;
    }
  }
}
