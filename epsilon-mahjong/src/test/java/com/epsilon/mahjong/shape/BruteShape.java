package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 形を総当たりで調べるテスト用のヘルパー。作る形（面子と雀頭の多重集合）を全て並べ、手に足りない枚数の最小を求める。解析器の表と
 * その生成器は使わない。作る形では、各種類をルールで決まる枚数まで使える。通常形をブロックごとに数えて (min, +) で合わせてよいのは、
 * 違うブロックの形が同じ種類を使わず、足りない枚数がブロックごとの和になるから。
 */
final class BruteShape {

  /** 作れない形に付ける、足りない枚数の値。4 つのブロックの和にしても溢れない。 */
  static final int IMPOSSIBLE = 1_000;

  private static final Block[] BLOCKS = Block.values();

  private final TileSet tiles;

  /** ブロックごとの、作れる面子の列（刻子は同じ種類 3 枚、順子は続いた 3 種類）。 */
  private final int[][][] melds = new int[BLOCKS.length][][];

  BruteShape(TileSet tiles) {
    this.tiles = tiles;
    for (Block block : BLOCKS) {
      List<int[]> list = new ArrayList<>();
      for (int kind = block.first(); kind <= block.last(); kind++) {
        if (tiles.copies(kind) >= HandShape.MELD_SIZE) {
          list.add(new int[] {kind, kind, kind});
        }
        if (block.runs()
            && kind + 2 <= block.last()
            && tiles.copies(kind) > 0
            && tiles.copies(kind + 1) > 0
            && tiles.copies(kind + 2) > 0) {
          list.add(new int[] {kind, kind + 1, kind + 2});
        }
      }
      melds[block.ordinal()] = list.toArray(new int[0][]);
    }
  }

  /** ブロックの牌から各形を作るのに足りない枚数の表を返す。添字は「面子の数 × 2 ＋ 雀頭の数」で、値はその形を作るのに足りない最小の枚数。 */
  int[] block(int[] counts, Block block) {
    int[] best = new int[ShapeSection.MELD_TARGETS * 2];
    Arrays.fill(best, IMPOSSIBLE);
    search(counts, block, 0, 0, 0, new int[Tiles.KINDS], best);
    return best;
  }

  /** 3 つの形（通常形・七対子・国士）のうち、足りない枚数の最小。 */
  int distance(int[] counts, int handMelds) {
    return Math.min(standard(counts, handMelds), Math.min(chiitoitsu(counts, handMelds), kokushi(counts, handMelds)));
  }

  int standard(int[] counts, int handMelds) {
    int[][] blocks = new int[BLOCKS.length][];
    for (Block block : BLOCKS) {
      blocks[block.ordinal()] = block(counts, block);
    }
    return combine(blocks, Variant.MAX_MELDS - handMelds);
  }

  int chiitoitsu(int[] counts, int handMelds) {
    if (handMelds > 0) {
      return IMPOSSIBLE;
    }
    List<Integer> covered = new ArrayList<>();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (tiles.copies(kind) >= 2) {
        covered.add(Math.min(2, counts[kind]));
      }
    }
    covered.sort((a, b) -> b - a);
    int sum = 0;
    for (int i = 0; i < HandShape.SEVEN_PAIRS; i++) {
      sum += covered.get(i);
    }
    return 2 * HandShape.SEVEN_PAIRS - sum;
  }

  int kokushi(int[] counts, int handMelds) {
    if (handMelds > 0) {
      return IMPOSSIBLE;
    }
    int best = IMPOSSIBLE;
    for (int pair = 0; pair < Tiles.KINDS; pair++) {
      if (!Tiles.contains(Tiles.TERMINALS_AND_HONORS, pair)) {
        continue;
      }
      int covered = 0;
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        if (Tiles.contains(Tiles.TERMINALS_AND_HONORS, kind)) {
          covered += Math.min(kind == pair ? 2 : 1, counts[kind]);
        }
      }
      best = Math.min(best, HandShape.MAX_SIZE - covered);
    }
    return best;
  }

  /** 引くと、3 つの形の足りない枚数の最小が減る種類のマスク（ルールで使う種類のうち、手に 4 枚無いもの）。 */
  long improving(int[] counts, int handMelds) {
    int[][] blocks = new int[BLOCKS.length][];
    for (Block block : BLOCKS) {
      blocks[block.ordinal()] = block(counts, block);
    }
    int target = Variant.MAX_MELDS - handMelds;
    int now = Math.min(combine(blocks, target), Math.min(chiitoitsu(counts, handMelds), kokushi(counts, handMelds)));
    long improving = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      if (tiles.copies(kind) == 0 || counts[kind] == Tiles.COPIES) {
        continue;
      }
      Block block = Block.of(kind);
      int[] saved = blocks[block.ordinal()];
      counts[kind]++;
      blocks[block.ordinal()] = block(counts, block);
      int after = Math.min(combine(blocks, target), Math.min(chiitoitsu(counts, handMelds), kokushi(counts, handMelds)));
      counts[kind]--;
      blocks[block.ordinal()] = saved;
      if (after < now) {
        improving |= Tiles.bit(kind);
      }
    }
    return improving;
  }

  /** 引くと和了の形になる種類のマスク。 */
  long waits(int[] counts, int handMelds) {
    return distance(counts, handMelds) == 1 ? improving(counts, handMelds) : 0;
  }

  /**
   * 決めた数の面子をブロックに割り振り、雀頭を 1 つのブロックにだけ置く。全ての分け方について、ブロックごとの足りない枚数を足し、
   * その最小を返す。
   *
   * @param target 作る面子の数
   */
  private static int combine(int[][] blocks, int target) {
    int best = IMPOSSIBLE;
    for (int pairBlock = 0; pairBlock < BLOCKS.length; pairBlock++) {
      for (int man = 0; man <= target; man++) {
        for (int pin = 0; man + pin <= target; pin++) {
          for (int sou = 0; man + pin + sou <= target; sou++) {
            int[] split = {man, pin, sou, target - man - pin - sou};
            int sum = 0;
            for (int b = 0; b < BLOCKS.length; b++) {
              sum += blocks[b][split[b] * 2 + (b == pairBlock ? 1 : 0)];
            }
            best = Math.min(best, sum);
          }
        }
      }
    }
    return best;
  }

  /**
   * 面子を列の順に（同じ面子を重ねてもよい）選び、面子の多重集合を全てたどる。たどった形ごとに、雀頭を足した形も数える。
   *
   * @param from 選べる最初の面子の、列での位置
   * @param used 作る形が使う、種類ごとの枚数
   */
  private void search(int[] counts, Block block, int from, int meldCount, int added, int[] used, int[] best) {
    best[meldCount * 2] = Math.min(best[meldCount * 2], added);
    for (int kind = block.first(); kind <= block.last(); kind++) {
      if (used[kind] + 2 <= tiles.copies(kind)) {
        int extra = Math.max(0, used[kind] + 2 - counts[kind]) - Math.max(0, used[kind] - counts[kind]);
        best[meldCount * 2 + 1] = Math.min(best[meldCount * 2 + 1], added + extra);
      }
    }
    if (meldCount == Variant.MAX_MELDS) {
      return;
    }
    int[][] types = melds[block.ordinal()];
    for (int type = from; type < types.length; type++) {
      boolean fits = true;
      int extra = 0;
      for (int kind : types[type]) {
        used[kind]++;
        fits &= used[kind] <= tiles.copies(kind);
        extra += used[kind] > counts[kind] ? 1 : 0;
      }
      if (fits) {
        search(counts, block, type, meldCount + 1, added + extra, used, best);
      }
      for (int kind : types[type]) {
        used[kind]--;
      }
    }
  }
}
