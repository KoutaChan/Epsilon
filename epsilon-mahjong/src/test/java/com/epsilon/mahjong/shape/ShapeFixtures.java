package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Random;

/** テスト用の手。表記は数字の後に種類の文字を置く（m・p・s。字牌は z で、1z〜7z が東南西北白發中）。 */
public final class ShapeFixtures {

  private ShapeFixtures() {}

  /** 表記に書いた牌の種類を、書いた順に返す。 */
  public static int[] tiles(String text) {
    int[] kinds = new int[text.length()];
    int size = 0;
    int pending = 0;
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      if (Character.isDigit(c)) {
        pending++;
        continue;
      }
      int first =
          switch (c) {
            case 'm' -> Tiles.M1;
            case 'p' -> Tiles.P1;
            case 's' -> Tiles.S1;
            case 'z' -> Tiles.TON;
            default -> throw new IllegalArgumentException("Unknown suit " + c + " in " + text);
          };
      for (int digit = i - pending; digit < i; digit++) {
        kinds[size++] = first + text.charAt(digit) - '1';
      }
      pending = 0;
    }
    if (pending > 0) {
      throw new IllegalArgumentException("Digits without a suit in " + text);
    }
    return Arrays.copyOf(kinds, size);
  }

  /** ちょうど 1 枚の牌を書いた表記から、その牌の種類を返す。 */
  public static int tile(String text) {
    int[] kinds = tiles(text);
    if (kinds.length != 1) {
      throw new IllegalArgumentException("Not one tile: " + text);
    }
    return kinds[0];
  }

  static int[] counts(String text) {
    int[] counts = new int[Tiles.KINDS];
    for (int kind : tiles(text)) {
      counts[kind]++;
    }
    return counts;
  }

  static HandShape shape(String text, int melds) {
    return shape(counts(text), melds);
  }

  public static HandShape shape(int[] counts, int melds) {
    HandShape hand = new HandShape();
    for (int meld = 0; meld < melds; meld++) {
      hand.addMeld();
    }
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      for (int copy = 0; copy < counts[kind]; copy++) {
        hand.add(kind);
      }
    }
    return hand;
  }

  static long kinds(String text) {
    long kinds = 0;
    for (int kind : tiles(text)) {
      kinds |= Tiles.bit(kind);
    }
    return kinds;
  }

  /**
   * 牌の集合から、ランダムに牌を選ぶ。どの種類も、集合にある枚数を超えては選ばない。
   *
   * @param tiles 選ぶ枚数
   * @return 種類ごとの枚数
   */
  static int[] randomTiles(Random random, TileSet set, int tiles) {
    int[] counts = new int[Tiles.KINDS];
    for (int placed = 0; placed < tiles; ) {
      int kind = random.nextInt(Tiles.KINDS);
      if (counts[kind] < set.copies(kind)) {
        counts[kind]++;
        placed++;
      }
    }
    return counts;
  }

  /**
   * 牌の集合にある牌で、ランダムな和了の形（面子と雀頭 1 つ）を作る。
   *
   * @param melds 面子の数
   * @return 種類ごとの枚数
   */
  public static int[] randomComplete(Random random, TileSet set, int melds) {
    while (true) {
      int[] counts = new int[Tiles.KINDS];
      counts[randomKind(random, set)] += 2;
      for (int meld = 0; meld < melds; meld++) {
        int kind = randomKind(random, set);
        boolean run =
            random.nextBoolean()
                && Tiles.Group.of(kind).suited()
                && Tiles.number(kind) + 2 < Tiles.SUIT_SIZE
                && set.copies(kind + 1) > 0
                && set.copies(kind + 2) > 0;
        if (run) {
          counts[kind]++;
          counts[kind + 1]++;
          counts[kind + 2]++;
        } else {
          counts[kind] += 3;
        }
      }
      if (fits(counts, set)) {
        return counts;
      }
    }
  }

  /** 和了の形から 1 枚を除き、2 回に 1 回はさらに 1 枚をランダムな牌に替えた形を返す（聴牌か 1 向聴が多い）。 */
  static int[] randomNearComplete(Random random, TileSet set, int melds) {
    int[] counts = randomComplete(random, set, melds);
    counts[randomHeld(random, counts)]--;
    if (random.nextBoolean()) {
      counts[randomHeld(random, counts)]--;
      addRandom(random, set, counts);
    }
    return counts;
  }

  /** 牌の集合にある種類から、一様に 1 つ選ぶ。 */
  public static int randomKind(Random random, TileSet set) {
    while (true) {
      int kind = random.nextInt(Tiles.KINDS);
      if (set.copies(kind) > 0) {
        return kind;
      }
    }
  }

  /** 1 枚以上ある種類から、一様に 1 つ選ぶ（枚数で重みを付けない）。 */
  public static int randomHeld(Random random, int[] counts) {
    while (true) {
      int kind = random.nextInt(Tiles.KINDS);
      if (counts[kind] > 0) {
        return kind;
      }
    }
  }

  private static void addRandom(Random random, TileSet set, int[] counts) {
    while (true) {
      int kind = randomKind(random, set);
      if (counts[kind] < Tiles.COPIES) {
        counts[kind]++;
        return;
      }
    }
  }

  private static boolean fits(int[] counts, TileSet set) {
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (counts[kind] > set.copies(kind)) {
        return false;
      }
    }
    return true;
  }
}
